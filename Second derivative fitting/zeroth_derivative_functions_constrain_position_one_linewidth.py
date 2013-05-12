import numpy
import pylab
import os
import glob
#from interactivity import MouseMonitor
from savitzky_golay import savitzky_golay
from scipy import stats
import matplotlib as mpl
mpl.rcParams['legend.fontsize']='small'
from matplotlib.ticker import MultipleLocator
from scipy.optimize import leastsq
from time import time
from scipy.stats import *
from scipy.interpolate import interp1d


########################################################################
############################## FUNCTIONS ###############################
########################################################################

########################################################################
#################### ZEROTH DERIVATIVE FUNCTIONS #######################

# Define functions #

def Lorentzian3(x,p,pos):
    return p[0]/(1 + (((x-pos[0])/(p[1]))**2)) +  p[2]/(1 + (((x-pos[1])/(p[3]))**2)) + p[4]/(1 + (((x-pos[2])/(p[5]))**2))

def residuals_lor3(p,y,x,pos):
    err = y - Lorentzian3(x,p,pos)
    return err

def Lorentzian4(x,p,pos):
    return p[0]/(1 + (((x-pos[0])/(p[1]))**2)) +  p[2]/(1 + (((x-pos[1])/(p[3]))**2)) + p[4]/(1 + (((x-pos[2])/(p[5]))**2)) + p[6]/(1 + (((x-pos[3])/(p[7]))**2)) 

def residuals_lor4(p,y,x,pos):
    err = y - Lorentzian4(x,p,pos)
    return err

def Lorentzian5(x,p,pos):
    return p[0]/(1 + (((x-pos[0])/(p[1]))**2)) +  p[2]/(1 + (((x-pos[1])/(p[3]))**2)) + p[4]/(1 + (((x-pos[2])/(p[5]))**2))+ p[6]/(1 + (((x-pos[3])/(p[7]))**2)) + p[8]/(1 + (((x-pos[4])/(p[9]))**2))

def residuals_lor5(p,y,x,pos):
    err = y - Lorentzian5(x,p,pos)
    return err
 
def residuals_lor35(p,y,x,pos):
    err = y - Lorentzian35(x,p,pos)
    return err
#########################################################################


def Lorentzian(x,p):
    return p[0]/(1 + (((x-p[1])/(p[2]))**2))

def draw_lorentzian(x,positions,intensities,linewidth,i,vspace):
    for j in range(len(intensities)):
        p = [intensities[j],positions[j],linewidth[j]]
        print p
        lor = Lorentzian(x,p)
        pylab.fill(x,lor+(i*vspace),'b',alpha=0.1)
    return 

########################################################################

def fitting_z_d1(x, y, intensities, positions): 
    if len(intensities) == 3:      
        p = list(intensities)
        pbest = leastsq(residuals_lor3,p,args=(y,x, positions),full_output=1)
        bestparams = pbest[0]
        datafit = Lorentzian3(numpy.array(x), bestparams, positions)

    if len(intensities) == 4:      
        p = list(intensities)
        pbest = leastsq(residuals_lor4,p,args=(y,x, positions),full_output=1)
        bestparams = pbest[0]
        datafit = Lorentzian4(numpy.array(x), bestparams, positions)

    if len(intensities) == 5:      
        p = list(intensities)
        pbest = leastsq(residuals_lor5,p,args=(y,x, positions),full_output=1)
        bestparams = pbest[0]
        datafit = Lorentzian5(numpy.array(x), bestparams, positions)
    return datafit, bestparams


