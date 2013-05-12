import numpy
import pylab
import matplotlib as mpl
mpl.rcParams['legend.fontsize']='small'
from matplotlib.ticker import MultipleLocator

def markersize(p,x):
    ax = pylab.gca()
    axis = ax.xaxis
    yaxis = ax.yaxis    
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(p)

    for line in axis.get_ticklines(minor=True) + yaxis.get_ticklines(minor=True):
        line.set_markersize(x)
    return 

def minor_ticks():
    ax = pylab.gca()
    x=ax.xaxis.get_ticklocs()
    y=ax.yaxis.get_ticklocs()    
    spacing_x = x[1]-x[0]
    spacing_y = y[1]-y[0]
    minor_loc_x = spacing_x/5
    minor_loc_y = spacing_y/5
    ax.xaxis.set_minor_locator(MultipleLocator(minor_loc_x))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_loc_y))
    return

def sci_not_y():
    ax = pylab.gca()
    ax.yaxis.major.formatter.set_powerlimits((0,0))
    return


