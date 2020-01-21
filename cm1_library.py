#!/usr/bin/env python

###### Import libraries #######
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

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
        add an arrow to a line.
        
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.
        """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    
    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.
        
        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.
        
        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0
        
        '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
    
    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')
    
    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1) + 0.0)


def CM1_plot_unitdict(var):
    #Below are dictionarys I use to make life easier. Pre-define units, contour levels, color maps and more
    unitdict={'upert':'m/s','ref':'dBz','rain':'mm','th':'K','thpert':'K','u':'m/s','prspert':'hPa','qv':'g/kg','w':r'vertical velocity ($ms^{-1}$)','cpc':'','cph':'km','u':'m/s','Rich':''}
    return unitdict[var]

def CM1_plot_clevdict(var):
    clevdict={'tke':np.array([2,4,6,8,10,12]),'upert':np.arange(-20.,20.1,2.0),'ref':np.linspace(-5,75,17),'rain':np.array([0, 1, 2.5, 5, 10, 25, 50, 100, 150, 200, 400]),'cpc':np.arange(1,36,5),'cph':np.arange(.25,4,.25),
    'th':np.arange(295,800,3.0),'thpert':np.arange(-1.,1.1,.10),'prspert':np.arange(-5.,5.1,.5),
'qv':np.array([.1,.5,1,2,4,6,8,10,12,14,16,18,20]),'wtrap':np.arange(-2.,2.1,.8),'wt':[-.3,-.2,-.18,-.16,-.14,-.12,-.10,-.08,-.06,-.04,-.02,.02,.04,.06,.08,.10,.12,.14,.16,.18,.2,.31],'w':[-2.,-1.8,-1.6,-1.4,-1.2,-1.0,-.8,-.6,-.4,-.2,.2,.4,.6,.8,1.0,1.2,1.4,1.6,1.8,2.1],'wt':[-.2,-.18,-.16,-.14,-.12,-.10,-.08,-.06,-.04,-.02,.02,.04,.06,.08,.10,.12,.14,.16,.18,.20],'w2':[-1.,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,.2,.3,.4,.5,.6,.7,.8,.9,1.],'wo':[-.1,-.09,-.08,-.07,-.06,-.05,-.04,-.03,-.02,.02,.03,.04,.05,.06,.07,.08,.09,.1],'Rich':np.arange(0,1,.1),'Rich':np.arange(0,1,.1)} #np.arange(-.20,.21,.020)
    
    print(var)
    return clevdict[var]

def CM1_plot_cmapdict(var):
    cmapdict={'tke':mpl.cm.YlOrBr,'upert':mpl.cm.PuOr_r,'th':mpl.cm.PRGn,'thpert':mpl.cm.RdBu_r,
    'prspert':mpl.cm.RdBu_r,'qv':mpl.cm.Greens,'w':mpl.cm.RdBu_r,'cpc':mpl.cm.Blues,'cph':mpl.cm.Greys,'u':mpl.cm.RdBu_r,'prspert':mpl.cm.RdBu_r,'Rich':mpl.cm.RdBu_r}
    return cmapdict[var]

def CM1_plot_levdict(var):
    levdict={'sfc':0,'.5km':4,'1km':8}
    return levdict[var]

def dirname(var,outputfile):
    #Provides filepath where images and movies are to be placed, given the type of variable to be contour filled.
    if var == 'th':
        movie = 'initpicture_41_publication'
    else:
        movie = 'bore_current_relative_41_publication'
    folderstring = var+movie
    return '/Users/Kevin/Desktop/cm1r19/images/'+outputfile+'/'+folderstring+'/'

def makerelative(u,*args):
    #Places U wind in a frame of reference to either the bore, the density current, the jet, or a combination of the three
    u_rel = u
    for speed in args:
        u_rel = u_rel - speed
    return u_rel

def CM1hour2runtime(hour):
    #Defines the numeric representation of the CM1 run
    if hour in range(1,10):
        runtime = '00000%d' %(hour)
    elif hour in range(10,100):
        runtime = '0000%d' %(hour)
    else:
        runtime = '000%d' %(hour)
    return runtime

def Richardson(U,TH,x_len,z):
    #Calculates the Richardson # for the flow
    g=9.81
    gradu = np.zeros(U.shape)
    gradth = np.zeros(TH.shape)
    
    for deriv in range(x_len):
        gradu[:,deriv] = np.array(np.gradient(U[:,deriv],z))
        gradth[:,deriv] = np.array(np.gradient(TH[:,deriv],z))
    
    return ((g/TH)*(gradth))/(gradu*gradu)

def makemask(var1,var2,threshold):
    #Creates a mask in the contour filled variable VAR for the profile of TKE to be superimposed on top
    mask = ma.masked_where(var1,var2>2)
    return ma.masked_array(var1,mask)

def getdata2d(filename,var):
    #Returns data from netcdf output file
    f = Dataset(filename)
    if var == 'time':
        data = f.variables[var][:]
        f.close()
        return data
    else:
        data = np.squeeze(f.variables[var][:])
        f.close()
        return data


def _pltXCVar(fig,x,z,variable,u,w,varName,clevs,cmap,extend,title,units):

        bounds=clevs
        norm=mpl.colors.BoundaryNorm(CM1_plto_clevdict['th'],cmap.N)
        plt.rc('font', size=20)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=22)
        #x,z=np.meshgrid(x,z)
        variable.shape
        ax = fig.gca()
        coldpoolline = {'levels':[302.8], 'colors':[(61.0/255.0,1.0/255.0,70.0/255.0)]}
        cs = ax.contour(x,z,TH[:kbound,ibound1:ibound2],**coldpoolline)
        cs.collections[0].set_linewidth(4)
        pltvar=plt.contourf(x,z,TH[:kbound,ibound1:ibound2],clevdict['th'],cmap=CM1_plot_cmapdict('th'),alpha = 1.0, norm=norm,extend=extend)
        pltline=ax.contour(pltvar,colors='k',alpha=0.5)
        ax.clabel(pltline,inline=1,fontsize=16,fmt='%g')
        pltref=plt.contourf(x,z,w,clevs,cmap=cmap,norm=mpl.colors.BoundaryNorm(clevs,cmap.N))
        ax.contour(pltref,colors='k',alpha=0.7)
        
        divider = make_axes_locatable(ax)
        #plt.ylabel('Height (km)',fontsize=30)
        #pltref=plt.contouf(x,z,w,[-2,-1,-0.5,-.25,.25,0.5,1,2],colors='g')
        grid = plt.grid(True)
        thin=1700
        thin2 = 7
        x_thin,z_thin=x[::thin2,::thin],z[::thin2,::thin]
        #x_thin,z_thin=np.meshgrid(x_thin,z_thin)
        #plt.barbs(x_thin,y_thin,u[0,::thin,::thin],v[0,::thin,::thin],length=5,barbcolor='k',flagcolor='k')
        scale=200.
        #Q=plt.quiver(x_thin,z_thin,u[::thin2,::thin],w[::thin2,::thin],scale=scale,headwidth=2)
        #key=-15.0 #11.5 for 71115 case
        #keylabel=str(key)+' m/s'
        #qk=plt.quiverkey(Q,-0.05,0.1,key, keylabel,labelpos='N')
        #plt.title('%s (%s), %s' %(title,unitdict['w'], tt))
        #ax1=fig.add_axes([0.02, 0.04, 0.9, 0.03])
        #cax = divider.append_axes("bottom", size="5%", pad=0.55)
        #cbar=mpl.colorbar.ColorbarBase(ax=cax,norm=mpl.colors.BoundaryNorm(clevs,cmap.N),cmap=cmap,extend=extend,ticks=bounds[::2],spacing='uniform',orientation='horizontal')
        #cbar.set_label(units,fontsize=20)
        
def plt_scale_figs(fig,x,z,variable,tke,w,TH,varName,clevs,cmap,extend,title,units,font,axes,xtick,ytick,ibound1,ibound2,kbound):
        # 1) contour filled plot of variable VAR
        # 2) contour fills TKE under mask
        # 3) contours TH values
        # 4) adds cosmetic lines over coarse boundary between tke and VAR
    
        plt.rc('font', size=font)
        plt.rc('axes', labelsize=axes)
        plt.rc('xtick', labelsize=xtick)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=ytick)
    
        bounds                  = clevs
        tkeclev                 = CM1_plot_clevdict('tke')
        tkecmap                 = CM1_plot_cmapdict('tke')
        norm                    = mpl.colors.BoundaryNorm(tkeclev,cmap.N)
        
        #x,z=np.meshgrid(x,z)
        ax                      = fig.gca()
        grid                    = plt.grid(True)
        
        boreline1               = {'levels':[1.9], 'colors':[(255./255.0,255./255.0,232.0/255.0)]}
        boreline2               = {'levels':[1.2], 'colors':[(247./255.0,247./255.0,247.0/255.0)]}
        
        pltline                 =ax.contour(x[:kbound,ibound1:ibound2],z[:kbound,ibound1:ibound2],TH[:kbound,ibound1:ibound2],colors='k',alpha=0.5,zorder=3)
        ax.clabel(pltline,inline=1,fontsize=20,fmt='%g')

        pltvar                  =plt.contourf(x[:kbound,ibound1:ibound2],z[:kbound,ibound1:ibound2],tke[:kbound,ibound1:ibound2],tkeclev,cmap=tkecmap,alpha = 1.0, norm=norm,extend=extend)
        pltline1                =ax.contour(pltvar,**boreline1)
        pltline2                =ax.contour(pltvar,**boreline2)
        pltline1.collections[0].set_linewidth(8)
        pltline2.collections[0].set_linewidth(8)

        pltref                  =plt.contourf(x[:kbound,ibound1:ibound2],z[:kbound,ibound1:ibound2],w[:kbound,ibound1:ibound2],clevs,cmap=cmap,norm=mpl.colors.BoundaryNorm(clevs,cmap.N))
        ax.contour(pltref,colors='k',alpha=0.7)
        
        divider                 = make_axes_locatable(ax)
        return

def _pltHov(fig,x,t,variable,varName,clevs,cmap,extend,title,units):
        bounds=clevs
        norm=mpl.colors.BoundaryNorm(bounds,cmap.N)
        xx,tt=np.meshgrid(x,time)
        tt=tt/3600.
        varmax=np.amax(vardict[VAR],axis=1)
        varmin=np.amin(vardict[VAR],axis=1)
        plt.contourf(xx,tt,varmax,clevdict[VAR],cmap=cmapdict[VAR])
        #plt.contourf(xx,tt,varmin,clevdict[VAR],cmap=cmapdict[VAR])
        plt.xlabel('x')
        plt.ylabel('time (h)')
        plt.title('%s (%s)' %(title,units))
        ax1=fig.add_axes([0.05, 0.04, 0.9, 0.03])
        cbar=mpl.colorbar.ColorbarBase(ax=ax1,norm=norm,cmap=cmap,extend=extend,ticks=bounds[::2],spacing='uniform',orientation='horizontal')
        cbar.set_label(units)
        return

def _pltsndings(fig,ax,z,var1,var2,xlab,vartit):
        plt.plot(var1,z,)
        plt.plot(var2,z,'r--')
        ax.set(xlabel=xlab, ylabel='Height(km)', title='vertical crosssection of %s' %vartit) #Velocity (ms^-^1), Theta Pert (K)
        #ax.set_xlim(-1,1)
        ax.grid()
        plt.show()
        return

def center_figure(tkeuse,lcush,rcush):
    #Find the lateral boundaries of the area to integrate for the Energy Budget
    tkemax=np.where(tkeuse==tkeuse.max())
    leftbound = int(tkemax[1]-lcush)
    rightbound = int(tkemax[1]+rcush)
    print(tkemax[1])
    return leftbound,rightbound


def definebdy(mult,hour,left,right):
    #Manually sets the boundaries for plotting
    ibound1=int(left+mult*(hour-1))
    ibound2=int(right+mult*(hour-1))
    iboundx = ibound2-ibound1
    return ibound1,ibound2,iboundx

                                                                # Extra Notes for self #
    #1145+633 and 1345+633 for jet N2 constant 60402
    #Trapped wave 2ms_130degrees: 1400-1800, -54 multiplier
    #mult by 2.5 for finer mesh (200/100)  #11 for 2km and 26 for 4km, 20 for 3km, 11 for 60402 case #-52 for 2/0.5ms trapped wave case 1350-1850; int(1100+24*(hour-1)) for 60402 jet no N2 change
    #1125 for finer mesh

def makemovies(outputfile,folderstring,VAR,movie):
    #Making videos of images #
    string = 'ffmpeg -r 4 -start_number 1 -i /Users/Kevin/Desktop/cm1r19/images/'+outputfile+'/'+folderstring+'/'+VAR+'%06d_resolution_plots.png -vcodec mpeg4 -y  -filter:v fps=fps=10 /Users/Kevin/Desktop/cm1r19/images/'+outputfile+'/'+folderstring+'/'+VAR+movie+'.mp4'

    os.system(string)
    return

def var_unstagger_vert(fld):
    return 0.5*(fld[1:,:] + fld[:-1,:])

def var_unstagger_u(fld):
    return 0.5*(fld[:,1:] + fld[:,:-1])











