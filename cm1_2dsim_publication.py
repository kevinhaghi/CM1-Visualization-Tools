#!/usr/bin/env python

###### Import libraries #######
from cm1_library import *
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import scipy as sp
#mpl.use('Agg')
#import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import sys
import os
import csv


##### Initial Declarations #######
VAR='w' #ref, prspert, thpert, th, upert, w, Rich                       #Determine the variable to colorfill for plots
outputfile      ='60402_idealjet_idealtheta'                                 #Set name of input folder for CM1 files
directory       ='/Users/Kevin/Desktop/cm1r19/cm1output/'+outputfile+'/'      #Set directory name where output folders are located

savedir         = dirname(VAR,outputfile)                                       #Name the file of the movie

os.system('mkdir /Users/Kevin/Desktop/cm1r19/images/'+outputfile)       #Make directories for images and movies
os.system('mkdir '+ savedir)

for hour in range(3,14):                                               #Loop over output files by output number
    runtime         = CM1hour2runtime(hour)
    fname           = 'cm1out_'+runtime+'.nc'
    
    #Get data from netcdf file
    dirname         = directory+fname
    time            = getdata2d(dirname,'time')
    x               = getdata2d(dirname,'xh') #km
    z               = getdata2d(dirname,'z') #km
    P               = getdata2d(dirname,'prs') #Pa
    psfc            = getdata2d(dirname,'psfc') #Pa
    rho             = getdata2d(dirname,'rho')
    Ppert           = getdata2d(dirname,'prspert') #Pa
    TH              = getdata2d(dirname,'th') #K
    THpert          = getdata2d(dirname,'thpert') #K
    u               = getdata2d(dirname,'uinterp') #m/s
    w               = getdata2d(dirname,'winterp') #m/s
    nm              = getdata2d(dirname,'nm')
    Upert           = getdata2d(dirname,'upert') #m/s
    tke_inter       = getdata2d(dirname,'tke') #m^2/s^2
    cpc             = getdata2d(dirname,'cpc')
    cph             = getdata2d(dirname,'cph')
    
    #Scale the variables
    Ppert           = Ppert/100.0
    cph             =cph/1000.
    
    #Make u flow-relative
    #0, 22, 11.11 for ideal jet ideal theta
    denspeed        = 0
    jetspeed        = 22
    borespeed       = 4
    u               = makerelative(u,denspeed)
    
    #Unstagger
    tke             = var_unstagger_vert(tke_inter)
    Upert           = Upert[:,:-1]
    
    #Create mask for bore to plot two colored contours on the same plot
    threshold       = 2
    wmask           = makemask(w,tke,threshold)
    

    #Find boundaries to center image around
    kbound          = 41
    extent          = 75                                #Controls the width of the image.  Grid size dependent
    lcush           = int(extent)
    rcush           = int(extent)
    ibound1,ibound2 = center_figure(tke,lcush,rcush)

    #Calculate Richardson #
    utot            = Upert + u
    thtot           = TH + THpert
    zgrad           = z*1000;
    len_x           = len(x)
    Rich            = Richardson(utot,thtot,len_x,zgrad)
    
    #####PLOT SECTION #####
    
    #Assign variables for plotting
    #assign variables to dictionary to make plotting more streamlined
    vardict         ={'upert':Upert,'thpert':THpert,'prspert':Ppert,'th':TH,'w':w,'cpc':cpc,'cph':cph,'u':u,'Rich':Rich,'tke':tke}
    
    var             = vardict[VAR]
    clevdict        = CM1_plot_clevdict(VAR)
    cmapdict        = CM1_plot_cmapdict(VAR)
    unitdict        = CM1_plot_unitdict(VAR)
    title           = VAR
    extend          = 'neither'
    x,z             = np.meshgrid(x,z)
    axes            = 20
    font            = 20
    xtick           = 22
    ytick           = 22
    
    fig=plt.figure(figsize=(12,10),frameon=True)#Set size of figure
    plt_scale_figs(fig,x,z,var,tke,wmask,TH,VAR,clevdict,cmapdict,extend,title,unitdict,font,axes,xtick,ytick,ibound1,ibound2,kbound)

    fig.savefig(savedir+VAR+runtime+'.pdf')                                     #Save file to directory









