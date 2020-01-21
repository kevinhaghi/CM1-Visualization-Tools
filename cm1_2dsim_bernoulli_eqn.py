#!/usr/bin/env python

#########################################################################
#########################################################################
#####                                                               #####
#    Bernoulli function analysis along trajectories                     #
#    ----------------------------------------------                     #
#    Calculates the Bernoulli function and its components along         #
#    trajectories that begin at a distance ahead of a bore and          #
#    end at a designated distance behind.  In a frame of reference      #
#    with the bore.                                                     #
#                                                                       #
#    Created by: Kevin Haghi                                            #
#    Date      : 9/21/19                                                #
#    Updated   : 1/20/20                                                #
#                                                                       #
#    1) Filename and folder creation                                    #
#    2) Data Input and Preparation                                      #
#    3) Calculation Bernoulli Values                                    #
#    4) Interpolate values to Bernoulli Surfaces                        #
#    5) Calculate Trajectories                                          #
#    6) Plot Bernoulli Functions                                        #
#####                                                               #####
#########################################################################
#########################################################################

#Import libraries
from matplotlib.colors import LinearSegmentedColormap
from pylab import rcParams
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
import sys
import os
import csv

#################################################
######## 1) Filename and Folder Creation ########
#################################################

VAR             ='tke' #ref, prspert, thpert, th, upert, w, Rich
outputfile      ='60402_redone_coldpool'
directory       ='/Users/Kevin/Desktop/cm1r19/cm1output/'+outputfile+'/'
os.system('mkdir /Users/Kevin/Desktop/cm1r19/images/'+outputfile)

savedir         = dirname(VAR,outputfile)
os.system('mkdir '+ savedir)

#################################################

for hour in range(26,27):           #Select range of outputs on which to perform analysis
    
    runtime         = CM1hour2runtime(hour)
    fname           ='cm1out_'+runtime+'.nc'

    ############################################################
    ############## 2) Data Input and Preparation ###############
    ############################################################
    
    dirname         = directory+fname
    time            = getdata2d(dirname,'time')
    x               = getdata2d(dirname,'xh') #km
    z               = getdata2d(dirname,'z') #km
    P               = getdata2d(dirname,'prs') #Pa
    psfc            = getdata2d(dirname,'psfc') #Pa
    rho             = getdata2d(dirname,'rho')
    rhopert         = getdata2d(dirname,'rhopert')
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
    cph             = cph/1000.
    
    #Unstagger
    Upert           = Upert[:,:-1]
    tke             = var_unstagger_vert(tke_inter)
    
    #Determine dx and dz
    dx              = round((x[1]-x[0])*1000.,1)
    dz              = round((z[1]-z[0])*1000.,1)

    ###############################################
    ######## 3) Calculate Bernoulli Values ########
    ###############################################
    
    #Calculate advective terms
    ubar            = u-Upert
    dudx            = np.gradient(Upert,dx,axis=1)
    uadvx           = ubar*dudx
    
    dwdx            = np.gradient(w,dx,axis=1)
    wadvx           = ubar*dwdx
    
    dudz            = np.gradient(ubar,dz,axis=0)
    uadvz           = w*dudz
    
    #Pressure gradient term
    rhobar          = rho-rhopert
    dpdx            = np.gradient(Ppert,dx,axis=1)
    dpdz            = np.gradient(Ppert,dz,axis=0)
    
    presx           = dpdx/rhobar
    presz           = dpdz/rhobar
    
    T               = TH*(P/100000.0)**(287/1004.0)
    
    #gravity term
    b               = -9.81*rhopert/rhobar
    
    #frame of reference
    #denspeed = 0
    #jetspeed = 22
    borespeed       = 14.16   #14.16 for ideal_ideal case, 15.83 for ideal_real
    cdudx           = -borespeed*dudx
    cdwdx           = -borespeed*dwdx
    
    #Bernoulli Eqn
    Bern            = uadvx+wadvx+uadvz+presx+presz+b
    Bernref         = Bern + cdudx+cdwdx
    Bernbnorm       = Bernref/b

    #Bernoulli Terms
    gterm           = np.zeros(T.shape)
    tterm           = T*1004
    uterm           = 0.5*((u-borespeed)**2+w**2)

    for i in range(len(x)):
        gterm[:,i] = 9.81*z*1000
    
    Bernoulliterm = tterm+uterm+gterm

    #Define calculation range                                         ##### MUST SET kbound ######
    kbound          = 41                                              #These are independently controlled.
    xt              = int(35*500.0/dx)                                #Controls the width of the image.  Grid size dependent
    extender        = int(100*500.0/dx)                               #Set your own values if the range is not desirable.
    lcush           = int(xt+extender)
    rcush           = int(xt)
    ibound1,ibound2 = center_figure(tke,lcush,rcush)

    ##################################################
    ### 4) Interpolate Bernoulli to Theta surfaces ###
    ##################################################

    thetagrid       = np.linspace(303,315,13)                           #Sets the range to contour
    zi              = np.zeros((len(thetagrid),len(x)))
    THi             = np.zeros((len(thetagrid),len(x)))
    berni           = np.zeros((len(thetagrid),len(x)))
    gtermi          = np.zeros((len(thetagrid),len(x)))
    ttermi          = np.zeros((len(thetagrid),len(x)))
    utermi          = np.zeros((len(thetagrid),len(x)))
    for i in range(len(x)):
        zi[:,i]         = np.interp(thetagrid,TH[:,i],z)
        THi[:,i]        = np.interp(zi[:,i],z,TH[:,i])
        berni[:,i]      = np.interp(zi[:,i],z,Bernoulliterm[:,i])
        gtermi[:,i]     = np.interp(zi[:,i],z,gterm[:,i])
        ttermi[:,i]     = np.interp(zi[:,i],z,tterm[:,i])
        utermi[:,i]     = np.interp(zi[:,i],z,uterm[:,i])

    ###################################################
    ######### 5) Calculate trajectories ###############
    ###################################################

    ct              = int(1000000/abs(u[6,ibound2]))                #Arbitrarily set counter for time, long enough for parcels to reach the end of their trajectory lengths
    ph              = [1,3,5,7,15,25]                               #Levels of parcels in model
    ds              = 7                                             # number of parcels in ph +1
    trajend         = 200000 #in meters                             #Location the parcels end in meters


    trajx           = np.zeros((ds,ct+1))
    trajz           = np.zeros((ds,ct+1))
    Berntraj        = np.zeros((ds,ct+1))
    ttermtraj       = np.zeros((ds,ct+1))
    utermtraj       = np.zeros((ds,ct+1))
    gtermtraj       = np.zeros((ds,ct+1))

    xmax            = np.max(x)*1000                                #Change to meters for trajectory calculations
    zmax            = np.max(z)*1000                                #Change to meters for trajectory calculations
    xo              = np.arange(0, xmax, dx)
    zo              = np.arange(0, zmax, dz)
    trajx[:,0]      = xo[ibound2]                                                         #Define initial position of trajectories in x direction
    trajz[:,0]      = [10,zo[ph[0]],zo[ph[1]],zo[ph[2]],zo[ph[3]],zo[ph[4]],zo[ph[5]]]    #Define initial position of trajectories in z direction
    
    
    # Find initial tractory locations and values for Bernoulli terms
    xp1             = find_nearest(xo,trajx[0,0])                   #Find the 4 points on the CM1 grid surrounding the tractory in the x direction at time 0
    zp1             = find_nearest(zo,trajz[0,0])                   #Find the 4 points on the CM1 grid surrounding the tractory in the z direction at time 0
    if (xp1-trajx[0,0]) >= 0:
        xp2         = find_nearest(xo,trajx[0,0]-dx)
    else:
        xp2         = find_nearest(xo,trajx[0,0]+dx)
    if (zp1-trajx[0,0]) >= 0:
        zp2         = find_nearest(zo,trajx[0,0]-dz)
    else:
        zp2         = find_nearest(zo,trajx[0,0]+dz)

    points          = [(xp1,zp1,Bernoulliterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,Bernoulliterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,Bernoulliterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,Bernoulliterm[np.where(zp2==zo),np.where(xp2==xo)])]
    Bernbi          = bilinear_interpolation(trajx[0,0], trajz[0,0], points) #Find value of interpolated w

    points          = [(xp1,zp1,tterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,tterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,tterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,tterm[np.where(zp2==zo),np.where(xp2==xo)])]
    ttermbi         = bilinear_interpolation(trajx[0,0], trajz[0,0], points) #Find value of interpolated w
    
    points          = [(xp1,zp1,uterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,uterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,uterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,uterm[np.where(zp2==zo),np.where(xp2==xo)])]
    utermbi         = bilinear_interpolation(trajx[0,0], trajz[0,0], points) #Find value of interpolated w
    
    points          = [(xp1,zp1,gterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,gterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,gterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,gterm[np.where(zp2==zo),np.where(xp2==xo)])]
    gtermbi         = bilinear_interpolation(trajx[0,0], trajz[0,0], points) #Find value of interpolated w

    # Assign values to Bernoulli terms according to the trajectories
    Berntraj[0,0]   = Bernbi
    ttermtraj[0,0]  = ttermbi
    utermtraj[0,0]  = utermbi
    gtermtraj[0,0]  = gtermbi

    for ht in range(1,ds):
        Berntraj[ht,0]   = Bernoulliterm[ph[ht-1],ibound2]
        ttermtraj[ht,0]  = tterm[ph[ht-1],ibound2]
        utermtraj[ht,0]  = uterm[ph[ht-1],ibound2]
        gtermtraj[ht,0]  = gterm[ph[ht-1],ibound2]

    # Calculate the Bernoulli values and position of the parcel along the trajectory
    for p in range(ds):
        t = 0
        while trajx[p,t] > trajend:
            #for t in range(ct):
            #First part finds the 4 points on the numerical grid that surround the location of the trajectory
            xp1              = find_nearest(xo,trajx[p,t])
            zp1              = find_nearest(zo,trajz[p,t])
            if (xp1-trajx[p,t]) >= 0:
                xp2              = find_nearest(xo,trajx[p,t]-dx)
            else:
                xp2              = find_nearest(xo,trajx[p,t]+dx)
            if (zp1-trajz[p,t]) >= 0:
                zp2              = find_nearest(zo,trajz[p,t]-dz)
            else:
                zp2              = find_nearest(zo,trajz[p,t]+dz)
            #print(xp1,xp2,zp1,zp2,xo[np.where(xp1==xo)])
            points           = [(xp1,zp1,u[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,u[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,u[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,u[np.where(zp2==zo),np.where(xp2==xo)])]
            ubi              = bilinear_interpolation(trajx[p,t], trajz[p,t], points) #Find value of interpolated u
            #print('ubi is ', ubi)
            points           = [(xp1,zp1,w[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,w[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,w[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,w[np.where(zp2==zo),np.where(xp2==xo)])]
            wbi              = bilinear_interpolation(trajx[p,t], trajz[p,t], points) #Find value of interpolated w
            
            points           = [(xp1,zp1,Bernoulliterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,Bernoulliterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,Bernoulliterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,Bernoulliterm[np.where(zp2==zo),np.where(xp2==xo)])]
            Bernbi           = bilinear_interpolation(trajx[p,t], trajz[p,t], points) #Find value of interpolated Bernoulli
            
            points           = [(xp1,zp1,tterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,tterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,tterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,tterm[np.where(zp2==zo),np.where(xp2==xo)])]
            ttermbi          = bilinear_interpolation(trajx[p,t], trajz[p,t], points) #Find value of interpolated Bernoulli
            
            points           = [(xp1,zp1,uterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,uterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,uterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,uterm[np.where(zp2==zo),np.where(xp2==xo)])]
            utermbi          = bilinear_interpolation(trajx[p,t], trajz[p,t], points) #Find value of interpolated Bernoulli
            
            points           = [(xp1,zp1,gterm[np.where(zp1==zo),np.where(xp1==xo)]),(xp2,zp1,gterm[np.where(zp1==zo),np.where(xp2==xo)]),(xp1,zp2,gterm[np.where(zp2==zo),np.where(xp1==xo)]),(xp2,zp2,gterm[np.where(zp2==zo),np.where(xp2==xo)])]
            gtermbi          = bilinear_interpolation(trajx[p,t], trajz[p,t], points) #Find value of interpolated Bernoulli

            # Assign values of calculated trajectories to arrays
            trajx[p,t+1]     = trajx[p,t] + ubi - borespeed
            trajz[p,t+1]     = trajz[p,t] + wbi
            Berntraj[p,t+1]  = Bernbi
            ttermtraj[p,t+1] = ttermbi
            utermtraj[p,t+1] = utermbi
            gtermtraj[p,t+1] = gtermbi
            t                = t+1     #increase count by 1
            tcount           = t-1     #use for placement of trajectory values in array Bernfinal
    Bernfinal       = sorted([Berntraj[1,tcount-1],Berntraj[2,tcount-1],Berntraj[3,tcount-1],Berntraj[4,tcount-1],Berntraj[5,tcount-1]])
    
    ###########################################
    #### 6) Plotting Bernoulli Function #######
    ###########################################

    #Change trajectories into km
    trajx           = trajx/1000.0
    trajz           = trajz/1000.0
    xplot,zplot     = np.meshgrid(x,z)

    #Create colormap for TKE
    tkecolormap     = CM1_plot_cmapdict('tke')
    interval        = np.hstack([np.linspace(0.2, 1)])
    colors          = mpl.cm.YlOrBr(interval)
    tkecmap         = LinearSegmentedColormap.from_list('name', colors)
    palette         = ['#064B9D','#1A0DE0','#B97BFE','#D722DC','#EC165C']
    textstr         = ['300m  ','500m  ','700m  ','1500m','2500m']

    #Designate label sizes
    plt.rc('font', size=20)
    plt.rc('axes', labelsize=30)
    plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=24)

    #Initiate figure
    fig             = plt.figure(figsize=(25,28),frameon=True)

    #Create subfigures in plot
    ax0             = plt.subplot2grid((1160,260),(0,0),colspan=247,rowspan=320)  #Designate size of subfigures
    ax1             = plt.subplot2grid((1160,260),(350,0),colspan=240,rowspan=160)
    ax2             = plt.subplot2grid((1160,260),(540,0),colspan=240,rowspan=160)
    ax3             = plt.subplot2grid((1160,260),(730,0),colspan=240,rowspan=160)
    ax4             = plt.subplot2grid((1160,260),(920,0),colspan=240,rowspan=160)
    borelines       = {'levels':np.arange(303,350,2), 'colors':[(0.0/255.0,0.0/255.0,0.0/255.0)]}
    thetalines      = ax0.contour(xplot[0:35,ibound1:ibound2],zplot[0:35,ibound1:ibound2],TH[0:35,ibound1:ibound2],linewidths =1, **borelines)
    tkecont         = ax0.contourf(xplot[0:30,ibound1:ibound2],zplot[0:30,ibound1:ibound2],tke[0:30,ibound1:ibound2],[2,4,6,8,10,12,14,16,18,20],cmap=tkecmap) #np.arange(0,25,1)

    divider         = make_axes_locatable(ax0)
    cax             = divider.append_axes('right', size='2%', pad=0.15)

    fig.colorbar(tkecont,cax=cax,orientation="vertical")

    #Plot trajectories
    for i in [1,2,3,4,5]:
        ax0.plot(trajx[i,:],trajz[i,:],linewidth = 4,color=palette[i-1])
        ax0.tick_params(
                axis                = 'x',          # changes apply to the x-axis
                which               = 'both',      # both major and minor ticks are affected
                bottom              = False,      # ticks along the bottom edge are off
                top                 = False,         # ticks along the top edge are off
                labelbottom         = False) # labels along the bottom edge are off
        ax1.plot(trajx[i,:],Berntraj[i,:]-Berntraj[i,0],linewidth = 4,color=palette[i-1])
        ax1.plot([ibound1/2,ibound2/2],[0,0.],'k-.')
        ax1.tick_params(
                axis                = 'x',          # changes apply to the x-axis
                which               = 'both',      # both major and minor ticks are affected
                bottom              = False,      # ticks along the bottom edge are off
                top                 = False,         # ticks along the top edge are off
                labelbottom         = False) # labels along the bottom edge are off
        ax3.plot(trajx[i,:],gtermtraj[i,:]-gtermtraj[i,0],linewidth = 4,color=palette[i-1])
        ax3.plot([ibound1/2,ibound2/2],[0,0.],'k-.')
        ax3.tick_params(
                axis                = 'x',          # changes apply to the x-axis
                which               = 'both',      # both major and minor ticks are affected
                bottom              = False,      # ticks along the bottom edge are off
                top                 = False,         # ticks along the top edge are off
                labelbottom         = False) # labels along the bottom edge are off
        ax2.plot(trajx[i,:],utermtraj[i,:]-utermtraj[i,0],linewidth = 4,color=palette[i-1])
        ax2.plot([ibound1/2,ibound2/2],[0,0.],'k-.')
        ax2.tick_params(
                axis                = 'x',          # changes apply to the x-axis
                which               = 'both',      # both major and minor ticks are affected
                bottom              = False,      # ticks along the bottom edge are off
                top                 = False,         # ticks along the top edge are off
                labelbottom         = False) # labels along the bottom edge are off
        
        ax4.plot(trajx[i,:],ttermtraj[i,:]-ttermtraj[i,0],linewidth = 4,color=palette[i-1])
        ax4.plot([ibound1/2,ibound2/2],[0,0.],'k-.')
        
        props           = dict(boxstyle='square', facecolor=palette[i-1], alpha=0.5)   #Controls the color on boxes...
        ax1.text(1.01,-2+(i-1)*.4, textstr[i-1], transform=ax1.transAxes, fontsize=20, #Used to display text coordinating trajectory color combinations to box color
            verticalalignment='top', bbox=props)

    # Make z labels for all plots
    fig.text(.065,.4,r'Energy (m$^\mathrm{2}$s$^\mathrm{-2}$)',va='center', ha='center', rotation='vertical', fontsize=rcParams['axes.labelsize'])
    fig.text(.065,.8,'Height (km)',va='center', ha='center', rotation='vertical', fontsize=rcParams['axes.labelsize'])

    # Make x label
    ax4.set_xlabel('Distance (km)')

    # Set x limits for x direction
    ax0.set_xlim([x[ibound1], x[ibound2]])
    ax1.set_xlim([x[ibound1], x[ibound2]])
    ax2.set_xlim([x[ibound1], x[ibound2]])
    ax3.set_xlim([x[ibound1], x[ibound2]])
    ax4.set_xlim([x[ibound1], x[ibound2]])

    #Set y limits for z direction
    ax0.set_ylim([0,2.5])
    ax1.set_ylim([-1500,1500])
    ax2.set_ylim([-1500,1500])
    ax3.set_ylim([-2500,6501])
    ax4.set_ylim([-6501,2500])

    #Set y ticks manually
    ax3.set_yticks([-2500,0,2500,5000])
    ax4.set_yticks([-5000,-2500,0,2500])

    #Text labels for plots
    ax0.text(ibound2*(dx/1000.)-10,2.300,r'Trajectories',fontsize=25)
    ax1.text(ibound2*(dx/1000.)-3,-1300,r'$\Delta B$',fontsize=25)
    ax4.text(ibound2*(dx/1000.)-4,-4500,r'$\Delta IE$',fontsize=25)
    ax2.text(ibound2*(dx/1000.)-4,-1300,r'$\Delta KE$',fontsize=25)
    ax3.text(ibound2*(dx/1000.)-4,-2000,r'$\Delta PE$',fontsize=25)

    textlabel           = dict(boxstyle='square', facecolor='gray', alpha=0.5)
    ax0.text(ibound1*(dx/1000.)+.85,2.350,r'a)',fontsize=25,bbox=textlabel)
    ax1.text(ibound1*(dx/1000.)+.85,1100,r'b)',fontsize=25,bbox=textlabel)
    ax4.text(ibound1*(dx/1000.)+.85,1400,r'e)',fontsize=25,bbox=textlabel)
    ax2.text(ibound1*(dx/1000.)+.85,1125,r'c)',fontsize=25,bbox=textlabel)
    ax3.text(ibound1*(dx/1000.)+.85,5350,r'd)',fontsize=25,bbox=textlabel)

    fig.savefig(savedir+'Trajectories'+str(hour)+'upper_trajectories.pdf',fontsize=30) #'_hour_'+str(p)+

