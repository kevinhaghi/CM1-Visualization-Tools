#!/usr/bin/env python

#########################################################################
#########################################################################
#####                                                               #####
#    Generate Isolated Soliton                                          #
#    ----------------------------------------------                     #
#    Takes a slice of a simulation, exponentially let the ends decay    #
#    back to the original background state, and places it into a restart#
#    file to be rerun.  Can reduce magnitude of spliced wave            #
#                                                                       #
#    Created by: Kevin Haghi                                            #
#    Date      : 4/5/19                                                 #
#    Updated   : 1/20/20                                                #
#                                                                       #
#    1) Filename and folder creation, Declarations                      #
#    2) Data Input and Preparation                                      #
#    3) Slice wave signal                                               #
#    4) Add decaying ends to signal                                     #
#    5) Place signal in restart file
#####                                                               #####
#########################################################################
#########################################################################


#Import libraries
from CM1_library import *
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
import sys


#################################################
######## 1) Filename and Folder Creation ########
#################################################

run2                    = 'solitary'
runtime                 = '000051'
runtime2                = '000001_environment'
outputfile              = '60402_jet_N2_0002_rdalpha_1.6'
rstfile                 = 'cm1out_rst_000001.nc'
directory               = '/Users/Kevin/Desktop/cm1r19/cm1output/'+outputfile+'/'
fname                   = 'cm1out_'+runtime+'.nc'
fname2                  = 'cm1out_'+runtime2+'.nc'
savedir                 = '/Users/Kevin/Desktop/cm1r19/images/'

ibound1                 = 800
ibound2                 = 2400
kbound                  = 351
multiplier              = .10           #Value to use if reducing magnitude of wave


############################################################
############## 2) Data Input and Preparation ###############
############################################################
    
dirname         = directory+fname
fbackground     = directory+fname2
frst            = directory+rstfile

time            = getdata2d(dirname,'time')
x               = getdata2d(dirname,'xh') #km
x               = getdata2d(dirname,'xf') #km
z               = getdata2d(dirname,'z')
za              = getdata2d(dirname,'zf') #km

Ppert           = getdata2d(dirname,'prspert') #pa
Pinit           = getdata2d(fbackground,'prs') #pa

THpert          = getdata2d(dirname,'thpert') #K
TH              = getdata2d(dirname,'th') #K
THinit          = getdata2d(fbackground,'th') #K

Upert           = getdata2d(dirname,'upert') #m/s
uinit           = getdata2d(fbackground,'u') #m/s
print(uinit[:,-1:])

w               = getdata2d(dirname,'w') #m/s
rho             = getdata2d(dirname,'rho')

#################### Generate background profile #########################
uprof           = uinit[:,-1:]*np.ones((kbound,ibound2-ibound1))
thprof          = THinit[:,-1:]*np.ones((kbound,ibound2-ibound1))
prsprof         = Pinit[:,-1:]*np.ones((kbound,ibound2-ibound1))

############################################################
############## 3) Slice out wave signal      ###############
############################################################

uwave           = multiplier*Upert[:,ibound1:ibound2]
wwave           = multiplier*w[0:kbound,ibound1:ibound2]
thwave          = multiplier*THpert[:,ibound1:ibound2]
prswave         = multiplier*Ppert[:,ibound1:ibound2]


deltax          = 400                            #set value to determine length of exponential decay area
deltacount      = 400.0                      #set numeric vaue for length of exponential decay area
xdir            = ibound2-ibound1 + deltax*2
zdir            = kbound

uwaveblend      = multiplier*Upert[:,ibound2:ibound2+deltax]
wwaveblend      = multiplier*w[0:kbound,ibound2:ibound2+deltax]
thwaveblend     = multiplier*THpert[:,ibound2:ibound2+deltax]
prswaveblend    = multiplier*Ppert[:,ibound2:ibound2+deltax]

umodel          = np.zeros((zdir,xdir))
wmodel          = np.zeros((zdir,xdir))
thmodel         = np.zeros((zdir,xdir))
thpmodel        = np.zeros((zdir,xdir))
prsmodel        = np.zeros((zdir,xdir))

############################################################
######## 4) Add decayig ends to signal             #########
############################################################


for i in range(deltax):
    decay = (np.exp((i-deltacount)/deltacount))**6
    #print(decay,i)
    #print(uprof.shape,uwave[:,0].shape,umodel[:,0].shape)
    umodel[:,i]                             = uprof[:,0]+ uwave[:,0]*decay
    wmodel[:,i]                             = wwave[:,0]*decay
    thmodel[:,i]                            = thwave[:,0]*decay
    prsmodel[:,i]                           = prsprof[:,0] + prswave[:,0]*decay

umodel[:,deltax:ibound2-ibound1+deltax]     = uprof + uwave
wmodel[:,deltax:ibound2-ibound1+deltax]     = wwave
thmodel[:,deltax:ibound2-ibound1+deltax]    = thwave
prsmodel[:,deltax:ibound2-ibound1+deltax]   = prsprof + prswave


for i in range(deltax):
    decay                                   = (np.exp(-i/deltacount))**3  #Generates exponentially decaying tails

    umodel[:,i+ibound2-ibound1+deltax]      =  uprof[:,0] + uwaveblend[:,i]*decay #Adds tails to signal
    wmodel[:,i+ibound2-ibound1+deltax]      = wwaveblend[:,i]*decay
    thmodel[:,i+ibound2-ibound1+deltax]     =  thwaveblend[:,i]*decay
    prsmodel[:,i+ibound2-ibound1+deltax]    = prsprof[:,0] + prswaveblend[:,i]*decay

#################################################
######## 5) Place signal in resart file  ########
#################################################


rstxpos         = 400                           #Set the horizontal grid where to input the wave signal

frst            = Dataset(directory+rstfile,'r+')
frst.variables['prs'][0,:,0,rstxpos:xdir+rstxpos]   = prsmodel
frst.variables['wa'][0,0:-1,0,rstxpos:xdir+rstxpos] = wmodel
frst.variables['tha'][0,:,0,rstxpos:xdir+rstxpos]   = thmodel
frst.variables['ua'][0,:,0,rstxpos:xdir+rstxpos]    = umodel
frst.close()


