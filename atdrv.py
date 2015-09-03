#!/usr/bin/python

import Adafruit_BBIO.PWM as PWM
import time
import math
import numpy as np
import random 

#Vehicle Geometry
#The length of car as a whole is a+b
global a
a = 0.6
global b
b = 0.4

def path(src,dst):
    #Taking up the Bezier path planner    
    
    #specify the length of tangent
    len = 1.0
    P0 = [src[0],src[1]]
    P1 = [0.0,0.0]
    P1[0] = P0[0] + len*math.cos(src[2])
    P1[1] = P0[1] + len*math.sin(src[2])

    P3 = [dst[0],dst[1]]
    P2 = [0.0,0.0]
    P2[0] = P3[0] - 20*len*math.cos(dst[2])
    P2[1] = P3[1] - 20*len*math.sin(dst[2])

    Gx = [P0[0],P1[0],P2[0],P3[0]]
    Gy = [P0[1],P1[1],P2[1],P3[1]]
 
    N = np.array([[1,-3,3,-1],[0,3,-6,3],[0,0,3,-3],[0,0,0,1]],float)

    i=0

    num = 10.0
    pathpar = np.zeros((2,num+1),dtype = float)

    for t in range(0,int(num)):
        
        T = np.array([[1],[t/num],[pow((t/num),2)],[pow((t/num),3)]],float)
        Px=np.dot(Gx,np.dot(N,T))
        Py=np.dot(Gy,np.dot(N,T))
        pathpar[0,i]=Px
        pathpar[1,i]=Py
        i=i+1
    
    pathpar[0,i]=P3[0]
    pathpar[1,i]=P3[1]

    return pathpar

    
def errcal(strt,end):
    #Function to calculate error on given particle
    #Error is Euclidean distance between desired and attained

    err = math.sqrt(pow((end[0] - strt[0]),2) + pow((end[1] - strt[1]),2))
    return err

def fwdkin(intl,jntvar):
    #Forward kinematic solver for a car
    
    #Vehicle Geometry
    #a and b should be nowhere else used
    global a
    global b
    l = a+b
    
    #Initial co:ordinates
    Xi = intl[0]
    Yi = intl[1]
    Thetai = intl[2]

    d = jntvar[0]
    psi = jntvar[1]
    R = l/math.tan(psi)
        
    if(R<500):
        Betta = (d*math.tan(psi))/l
        Thetaf = Thetai + Betta
        Xf = Xi + (R*math.sin(Thetaf)) - (R*math.sin(Thetai)) + (b*math.cos(Thetaf)) - (b*math.cos(Thetai))
        Yf = Yi - (R*math.cos(Thetaf)) + (R*math.cos(Thetai)) + (b*math.sin(Thetaf)) - (b*math.sin(Thetai))
    else:
        Thetaf = Thetai
        Xf = Xi + d*math.cos(Thetaf)
        Yf = Yi + d*math.sin(Thetaf)
    
    attnd = np.zeros(3,dtype = float)
    attnd[0] = Xf
    attnd[1] = Yf
    attnd[2] = Thetaf
    
    return attnd

def Gaussian(x,sigma):
    #Function to calculate the probability of x 
    #for 1-dim Gaussian with mean 0 and var. sigma
        
     wgt = np.exp(-(pow(x,2))/(2.0*(pow(sigma,2))))
     wgt = wgt/np.sqrt(2.0*3.1415*(pow(sigma,2)))
 
     return wgt

def invkin2( intl,dsrd ):
    #Inverse kinematic solver
    #Initial task space co:ordinates in intl.
    #Desired task space co:ordinates in dsrd.
    #Refer IK4CAR for more details.
    
    #set limits
    stlinedist = errcal(dsrd,[intl[0],intl[1]])
    d_min = stlinedist
    d_max = stlinedist*3.1415
    psi_min = -(3.1415/3.0)
    psi_max = (3.1415/3.0)

    #set std dev. of gaussian prob func.
    stderr = 0.1
    #set spread while resampling
    ressprd = 1.0

    #Lets populate
    #Set number of particles
    N = 100
    rnge = 10.0
    
    particles = np.random.uniform(0.0,rnge,(2,N))

    #Setting weights on each particle
    wgt = np.zeros((N,1),dtype = float)
    
    for pno in range(0,N):
    
        #modify to original parameters
        modfy = np.zeros((2),dtype = float)
        modfy[0] =  (particles[0,pno]*((d_max - d_min)/rnge)) + d_min
        modfy[1] =  (particles[1,pno]*((psi_max - psi_min)/rnge)) + psi_min
        
        #Calculate frwd kinematic of particle
        attnd = fwdkin(intl,modfy)
        
        #Calculate error of particle
        err = errcal(dsrd,[attnd[0],attnd[1]])
        
        #Calculate weight of particle
        wgt[pno] = Gaussian(err,stderr)
    
    #Resampling
    while(np.sum(np.std(particles,1)) > 0.1):
    
        newparticles = np.zeros((2,N),dtype = float)
        newwgt = np.zeros((N,1),dtype = float)
        idx = int(random.random()*N)
        beta = 0.0
        mw = max(wgt)
        
        for i in range(N):
            beta = beta + 2.0*mw*random.random()
            
            while (beta >= wgt[idx]):
                beta = beta - wgt[idx]
                idx = (idx+1)%N   
            newparticles[0][i] = particles[0][idx]
            newparticles[1][i] = particles[1][idx]
            newwgt[i] = wgt[idx]
        
        particles = newparticles
        wgt = newwgt
    
    #Repopulate
    for pno in range(0,N):
        particles[0][pno] = particles[0][pno] + ressprd*np.random.normal(0,0.1)
        particles[1][pno] = particles[1][pno] + ressprd*np.random.normal(0,0.5)
    
    #Setting weights on each particle
    for pno in range(0,N):
        
        #modify to original parameters
        modfy[0] =  (particles[0][pno]*((d_max - d_min)/rnge)) + d_min
        modfy[1] =  (particles[1][pno]*((psi_max - psi_min)/rnge)) + psi_min
        
        #Calculate frwd kinematic of particle
        attnd = fwdkin(intl,modfy)
        
        #Calculate error of particle
        err = errcal(dsrd,[attnd[0],attnd[1]])
        
        #Calculate weight of particle
        wgt[pno] = Gaussian(err,stderr)
    
    #Resampling
    while(np.sum(np.std(particles,1)) > 0.1):

        idx = int(random.random()*N)
        beta = 0.0
        mw = max(wgt)
        
        for i in range(0,N):
            beta = beta + 2.0*mw*random.random()
            
            while (beta >= wgt[idx]):
                beta = beta - wgt[idx]
                idx = (idx+1)%N                                           
            
            newparticles[0][i] = particles[0][idx]
            newparticles[1][i] = particles[1][idx]
            newwgt[i] = wgt[idx]
            
        particles = newparticles
        wgt = newwgt
    
    #solution
    sol = np.zeros((2),dtype = float)
    sol[0] = (particles[0][pno]*((d_max - d_min)/rnge)) + d_min
    sol[1] = (particles[1][pno]*((psi_max - psi_min)/rnge)) + psi_min
    
    return sol

def srvdrv(angl):
    #Code to drive the servo motor using PWM 
    #Min angle at pwm = 8
    #Max angle at pwm = 12
    #Frequency set at 60Hz

    angl += (3.1415/3)
    angl *= 4.0
    angl /= (2*(3.1415/3))
    angl += 8.0
    
    PWM.start("P8_19",angl,60)
    time.sleep(0.2)
    PWM.stop("P8_19")
    
def motodrv(dist,speed = 100):
    #Code to drive the Dc motor using PWM for speed control
    #Postion control is done by driving the motor for desired time only
    #Min speed at pwm = 20
    #Max at 100
    #setting default speed at 45
    #Frequency set at 40kHz
    
    PWM.start("P8_13",speed,40000)
    time.sleep(dist*2.0)
    PWM.stop("P8_13")
    
def main():
    
    #print 'Enter source co:ordinates \nEnter Xi:'
    #Xi = input()
    Xi = 0.0
    #print 'Enter Yi:'
    #Yi = input()
    Yi = 0.0
    #print 'Enter Thetai:'
    #Thetai = input()
    Thetai = 0.0
    print 'Intital steering orientation taken to be zero'
    psi = 0.0
    intl = [Xi,Yi,Thetai]
    
    #print 'Enter destination co:ordinates \nEnter Xf:'
    #Xf = input()
    Xf = 7.0
    #print 'Enter Yf:'
    #Yf = input()
    Yf = 7.0
    #print 'Enter Thetaf:'
    #Thetaf = input()
    Thetaf = 3.14
    dsrd = [Xf,Yf,Thetaf]
    
    print 'Creating trajecory\n'
    conpoints = path(intl,dsrd)

    print 'Auto Drive engaged:::\n'
    prev = intl
    for mov in range(1,np.size(conpoints,1)):

        print '\nCon.point :',conpoints[0,mov],' : ',conpoints[1,mov]

        jntvar = invkin2(prev,[conpoints[0,mov],conpoints[1,mov]])
        attnd = fwdkin(prev,jntvar)
        print 'Attained :',attnd[0],' : ',attnd[1]

        srvdrv(jntvar[1])
        motodrv(jntvar[0])
        
        prev = attnd
    
    PWM.cleanup()   

    return
    
main()
