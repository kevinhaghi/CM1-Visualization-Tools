#!/usr/bin/env python

#########################################################################
#########################################################################
#####                                                               #####
#    Plot Max Turbulent Kinetic Energy vs Max W                         #
#    ----------------------------------------------                     #
#    Calculates the max tke and max w and plots them vs afo time        #
#                                                                       #
#    Created by: Kevin Haghi                                            #
#    Date      : 11/21/19                                               #
#    Updated   : 1/20/20                                                #
#                                                                       #
#    1) Filename and folder creation, Declarations                      #
#    2) Data Input and Preparation                                      #
#    3) Plot TKE vs W                                                   #
#####                                                               #####
#########################################################################
#########################################################################

#Import libraries
from cm1_library import *
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import scipy as sp
from scipy import integrate
#mpl.use('Agg')
#import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from netCDF4 import Dataset
from scipy.signal import find_peaks
import sys
import os
import csv


###############################################################
######## 1) Filename and Folder Creation, Declarations ########
###############################################################
outputfile1             ='60402_redone_coldpool'
outputfile2             ='60402_idealjet_realtheta'
outputfile3             ='60402_idealjet_idealtheta'

directory1              ='/Users/Kevin/Desktop/cm1r19/cm1output/'+outputfile1+'/'
directory2              ='/Users/Kevin/Desktop/cm1r19/cm1output/'+outputfile2+'/'
directory3              ='/Users/Kevin/Desktop/cm1r19/cm1output/'+outputfile3+'/'

timecount               =0  #Sets the variable for incrementing time vectors to 0

time_range                   =[8,34]

#real
plottke1 = np.zeros((time_range[1]-time_range[0]))
plotw1 = np.zeros((time_range[1]-time_range[0]))

#ideal/real
plottke2 = np.zeros((time_range[1]-time_range[0]))
plotw2 = np.zeros((time_range[1]-time_range[0]))

#ideal/ideal
plottke3 = np.zeros((time_range[1]-time_range[0]))
plotw3 = np.zeros((time_range[1]-time_range[0]))

#parameters to set before running
for hour in range(time_range[0],time_range[1]):
    
    runtime             = CM1hour2runtime(hour)
    fname               = 'cm1out_'+runtime+'.nc'
    
    dirname1 = directory1+fname
    dirname2 = directory2+fname
    dirname3 = directory3+fname
    
    ############################################################
    ############## 2) Data Input and Preparation ###############
    ############################################################
    
    #open the file and grab variables you want
    time1                    = getdata2d(dirname1,'time')
    x1                       = getdata2d(dirname1,'xh') #km
    z1                       = getdata2d(dirname1,'z') #km
    tke_inter1               = getdata2d(dirname1,'tke') #m^2s^-2
    w1                       = getdata2d(dirname1,'winterp') #Pa
    Ppert1                   = getdata2d(dirname1,'prspert') #hpa
    Upert1                   = getdata2d(dirname1,'upert') #m/s
    P1                       = getdata2d(dirname1,'prs') #hpa
    
    time2                    = getdata2d(dirname2,'time')
    x2                       = getdata2d(dirname2,'xh') #km
    z2                       = getdata2d(dirname2,'z') #km
    tke_inter2               = getdata2d(dirname2,'tke') #m^2s^-2
    w2                       = getdata2d(dirname2,'winterp') #Pa
    Ppert2                   = getdata2d(dirname2,'prspert') #hpa
    Upert2                   = getdata2d(dirname2,'upert') #m/s
    P2                       = getdata2d(dirname2,'prs') #hpa
    
    time3                    = getdata2d(dirname3,'time')
    x3                       = getdata2d(dirname3,'xh') #km
    z3                       = getdata2d(dirname3,'z') #km
    tke_inter3               = getdata2d(dirname3,'tke') #m^2s^-2
    w3                       = getdata2d(dirname3,'winterp') #Pa
    Ppert3                   = getdata2d(dirname3,'prspert') #hpa
    Upert3                   = getdata2d(dirname3,'upert') #m/s
    P3                       = getdata2d(dirname3,'prs') #hpa
    
    #Unstagger
    Upert1                   = Upert1[:,:-1]
    Upert2                   = Upert2[:,:-1]
    Upert3                   = Upert3[:,:-1]
    tke1                     = var_unstagger_vert(tke_inter1)
    tke2                     = var_unstagger_vert(tke_inter2)
    tke3                     = var_unstagger_vert(tke_inter3)
    
    #Find max values for w and tke
    plotw1[timecount]        = find_max(w1)
    
    plotw2[timecount]        = find_max(w2)
    
    plotw3[timecount]        = find_max(w3)
    
    plottke1[timecount]      = find_max(tke1)
    
    plottke2[timecount]      = find_max(tke2)
    
    plottke3[timecount]      = find_max(tke3)
    
    timecount                += 1

###########################################
#### 3) Plotting Bernoulli Function #######
###########################################

fig                     =plt.figure(figsize=(25,18),frameon=True)
plt.rc('axes', labelsize=40)
plt.rc('xtick', labelsize=32)    # fontsize of the tick labels
plt.rc('ytick', labelsize=32)
plt.ylabel(r'Max Vertical Velocity ($ms^{-1}$)')
plt.xlabel(r'Max TKE ($m^2s^{-2}$)')


IIcolor                 = '#004D40'         #Ideal/Ideal Color
ICcolor                 = '#1E88E5'         #Ideal/Composite Color
CCcolor                 = '#D81B60'         #Composite/Composite Color
mksize                  = 60                #Marker Size

plt.plot(plottke1[0:5],plotw1[0:5],CCcolor,linewidth=8,label='Comp/Comp')
plt.plot(plottke1[4:9],plotw1[4:9],CCcolor,linewidth=8)
plt.plot(plottke1[8:13],plotw1[8:13],CCcolor,linewidth=8)
plt.plot(plottke1[12:17],plotw1[12:17],CCcolor,linewidth=8)
plt.plot(plottke1[16:21],plotw1[16:21],CCcolor,linewidth=8)
plt.plot(plottke1[20:26],plotw1[20:26],CCcolor,linewidth=8)

plt.plot(plottke1[0],plotw1[0],'.g',linewidth=4,markersize=mksize,color=CCcolor)
plt.plot(plottke1[4],plotw1[4],'.g',linewidth=4,markersize=mksize,color=CCcolor)
plt.plot(plottke1[8],plotw1[8],'.g',linewidth=4,markersize=mksize,color=CCcolor)
plt.plot(plottke1[12],plotw1[12],'.g',linewidth=4,markersize=mksize,color=CCcolor)
plt.plot(plottke1[16],plotw1[16],'.g',linewidth=4,markersize=mksize,color=CCcolor)
plt.plot(plottke1[20],plotw1[20],'.g',linewidth=4,markersize=mksize,color=CCcolor)
plt.plot(plottke1[25],plotw1[25],'.g',linewidth=4,markersize=mksize,color=CCcolor)

plt.text(plottke1[0],plotw1[0],'0',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke1[4],plotw1[4],'1',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke1[8],plotw1[8],'2',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke1[12],plotw1[12],'3',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke1[16],plotw1[16],'4',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke1[20],plotw1[20],'5',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke1[25],plotw1[25],'6',horizontalalignment='center',verticalalignment='center',color='white',size=30)


plt.plot(plottke2[0:5],plotw2[0:5],ICcolor,linewidth=8,label='Ideal/Comp')
plt.plot(plottke2[4:9],plotw2[4:9],ICcolor,linewidth=8)
plt.plot(plottke2[8:13],plotw2[8:13],ICcolor,linewidth=8)
plt.plot(plottke2[12:18],plotw2[12:18],ICcolor,linewidth=8)
plt.plot(plottke2[16:21],plotw2[16:21],ICcolor,linewidth=8)
plt.plot(plottke2[20:26],plotw2[20:26],ICcolor,linewidth=8)

plt.plot(plottke2[0],plotw2[0],'.',linewidth=4,markersize=mksize,color=ICcolor)
plt.plot(plottke2[4],plotw2[4],'.',linewidth=4,markersize=mksize,color=ICcolor)
plt.plot(plottke2[8],plotw2[8],'.',linewidth=4,markersize=mksize,color=ICcolor)
plt.plot(plottke2[12],plotw2[12],'.',linewidth=4,markersize=mksize,color=ICcolor)
plt.plot(plottke2[16],plotw2[16],'.',linewidth=4,markersize=mksize,color=ICcolor)
plt.plot(plottke2[20],plotw2[20],'.',linewidth=4,markersize=mksize,color=ICcolor)
plt.plot(plottke2[25],plotw2[25],'.',linewidth=4,markersize=mksize,color=ICcolor)

plt.text(plottke2[0],plotw2[0],'0',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke2[4],plotw2[4],'1',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke2[8],plotw2[8],'2',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke2[12],plotw2[12],'3',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke2[16],plotw2[16],'4',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke2[20],plotw2[20],'5',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke2[25],plotw2[25],'6',horizontalalignment='center',verticalalignment='center',color='white',size=30)

plt.plot(plottke3[0:5],plotw3[0:5],IIcolor,linewidth=8,label='Ideal/Ideal')
plt.plot(plottke3[4:9],plotw3[4:9],IIcolor,linewidth=8)
plt.plot(plottke3[8:13],plotw3[8:13],IIcolor,linewidth=8)
plt.plot(plottke3[12:18],plotw3[12:18],IIcolor,linewidth=8)
plt.plot(plottke3[16:21],plotw3[16:21],IIcolor,linewidth=8)
plt.plot(plottke3[20:26],plotw3[20:26],IIcolor,linewidth=8)

plt.plot(plottke3[0],plotw3[0],'.',linewidth=4,markersize=mksize,color=IIcolor)
plt.plot(plottke3[4],plotw3[4],'.',linewidth=4,markersize=mksize,color=IIcolor)
plt.plot(plottke3[8],plotw3[8],'.',linewidth=4,markersize=mksize,color=IIcolor)
plt.plot(plottke3[12],plotw3[12],'.',linewidth=4,markersize=mksize,color=IIcolor)
plt.plot(plottke3[16],plotw3[16],'.',linewidth=4,markersize=mksize,color=IIcolor)
plt.plot(plottke3[20],plotw3[20],'.',linewidth=4,markersize=mksize,color=IIcolor)
plt.plot(plottke3[25],plotw3[25],'.',linewidth=4,markersize=mksize,color=IIcolor)

plt.text(plottke3[0],plotw3[0],'0',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke3[4],plotw3[4],'1',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke3[8],plotw3[8],'2',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke3[12],plotw3[12],'3',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke3[16],plotw3[16],'4',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke3[20],plotw3[20],'5',horizontalalignment='center',verticalalignment='center',color='white',size=30)
plt.text(plottke3[25],plotw3[25],'6',horizontalalignment='center',verticalalignment='center',color='white',size=30)


plt.ylim(0.5,3.)
plt.xlim(10,22)
plt.legend(framealpha=1, frameon=True,prop={'size': 30},loc=2);

fig.savefig('/Users/Kevin/Desktop/cm1r19/images/publication_figures/tkevsw.pdf',fontsize=30)
plt.close()


