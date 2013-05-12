import numpy
import pylab
import os
import glob
from scipy import stats
import matplotlib as mpl
mpl.rcParams['legend.fontsize']='small'
from matplotlib.ticker import MultipleLocator
from scipy.optimize import leastsq
from time import time
from scipy.stats import *
from scipy.interpolate import interp1d, splrep, splev
import sys

## import scripts ## 
from interactivity import MouseMonitor
from savitzky_golay import savitzky_golay
from plot_decorations import minor_ticks, markersize, sci_not_y

## second_derivative fitting functions: all parameters fitted ##
from second_derivative_functions import basic_num_diff,triangular, second_derivative, combined_second_derivative_l5, combined_second_derivative_l4, combined_second_derivative_l3, residuals_l5, residuals_l4, residuals_l3, fitting

## zeroth derivative Lorentzian: constrained position and all peaks fitted ## 
from zeroth_derivative_functions_constrain_position_one_linewidth import Lorentzian3, Lorentzian4, Lorentzian5, residuals_lor3, residuals_lor4, residuals_lor5, Lorentzian, draw_lorentzian, fitting_z_d1

###################################################################
############################### DATA ##############################
time1 = time()

## Generate some data ##
x = numpy.arange(200,300,0.2)
p = [1.0,250,4.0,0.4,256,3.0,0.6,259,5.0]

data = []
for i in range(len(p)/3):
	params = [p[i*3],p[i*3 + 1], p[i*3 + 2]]
	print params
	l = Lorentzian(x,params)
	data.append(l)
	
data = sum(data[0:len(data)])
data = numpy.array((data))

# add a linear background ##
data = (2e-3*x + 2.0) + data

###################################################################
################ FITTING SECOND DERIVATIVE SPECTRA ################
###################################################################

intensity1 = []
position1 = []
linewidth1=[]
fits1 = []
bp = []
positions = []
second_derivative_spectra = []
x_values = []

pylab.interactive(False)

## smoothing algorithms for real data ## 
#data = triangular(data,10)
#data = savitzky_golay(data,11,order=3)

## calculating the second derivative ## 
x_fit1 = []
x_fit2 = []
for j in range(len(x)-1):
    x_fit1.append((x[j]+x[j+1])/2)
for j in range(len(x_fit1)-1):
    x_fit2.append((x_fit1[j]+x_fit1[j+1])/2)
first_der_spec = basic_num_diff(x,data)
second_der_spec = numpy.array(basic_num_diff(x_fit1,first_der_spec))

x_fit2 = numpy.array((x_fit2))
x_fit1 = numpy.array((x_fit1))
first_der_spec = numpy.array((first_der_spec))
second_der_spec = numpy.array((second_der_spec))

x_values.append(x_fit2)
second_derivative_spectra.append(second_der_spec)

###################################################################
####################### INTERACTIVE FITTING #######################
###################################################################

mouse = MouseMonitor()
mouse.set_data(x_values,second_derivative_spectra)
mouse.plot_next()
pylab.show()

xdata = mouse.xdatalist
ydata = mouse.ydatalist
number_lor = mouse.return_length()
number_lor.insert(0,0)

###################################################################
############# SECOND DERIVATIVE INTENSITY CALIBRATION #############
###################################################################
'''
This function was established empirically; comparing second derivative peak intensities with the intensity of the original peak.
'''

def second_derivative_intensity(intensity):
    y = -7.520183858739415*intensity + 2.6656315501400189e-15
    return y

###################################################################
###################### UNCONSTRAINED FITTING ######################
###################################################################
'''
This part of the script fits a linear combination of N  second derivatives to the data.
'''

fits1 = []
peaks = []
peak_intensities = []
peak_max = []
all_second_der_bestparams = []
## starting linewidth ## 
linewidth = 3.0


for i in range(len(number_lor)-1):
    val = number_lor[i]
    valplus = number_lor[i+1]
    peaks.append(xdata[val:valplus])
    peak_intensities.append(ydata[val:valplus])
#    peak_max.append(ymax[val:valplus])

for i in range(len(number_lor)-1):
    time2 = time()
    print 'Fitting Second Derivative Spectrum %d of %d, elapsed time: %5.2f minutes' % (i+1, len(number_lor)-1, (time2 - time1)/60.0)

    x1 = x_values[i]
    y1 = second_derivative_spectra[i]

    peaks_spectra = peaks[i]
    intensities = numpy.array((peak_intensities[i])) 
    datafit, bestparams, trial = fitting(x1, y1, intensities, peaks_spectra, linewidth)
    all_second_der_bestparams.append(bestparams)
    fits1.append(datafit)

 
####################################################################
################### SEPARATE FITTED PARAMETERS #####################
####################################################################

s_d_intensity = []
s_d_position = []
s_d_linewidth = []

for i in range(len(all_second_der_bestparams)):
    bestparams = all_second_der_bestparams[i]
    intensity = numpy.array((bestparams[::3]))/max(numpy.array((bestparams[::3])))
    position = numpy.array((bestparams[1::3]))  
    linewidth_fitting = numpy.array((bestparams[2::3]))  
    s_d_intensity.append(intensity)
    s_d_position.append(position)
    s_d_linewidth.append(linewidth_fitting)

s_d_linewidth = numpy.array((s_d_linewidth))/2.0
s_d_intensity = numpy.array((s_d_intensity))
s_d_position = numpy.array((s_d_position))

############# PLOTTING SECOND DERIVATIVE SPECTRA AND FITS ##########
for i in range(len(all_second_der_bestparams)):
    vspace = 0.1
    pylab.subplot(211)
    pylab.plot(x,data,'b',lw=2)
    pylab.xlim(200, 300)
    minor_ticks()
    markersize(10,5)
    pylab.ylabel(r'y', fontsize = 18)
    draw_lorentzian(x1,s_d_position[i],s_d_intensity[i],s_d_linewidth[i],1,2.5)

    pylab.subplot(212)
    pylab.plot(x_values[i],second_derivative_spectra[i]+(i*vspace),'wo')
    pylab.plot(x1, datafit+(i*vspace),'r-',lw=2)
    pylab.xlim(200, 300)
    pylab.ylim(min(datafit) - 0.1,max(datafit)+(i*vspace) + 0.1)


    minor_ticks()
    markersize(10,5)

    pylab.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
    pylab.ylabel(r'$d^{2}y/dx^{2}$', fontsize = 18)
    pylab.show()

s_d_intensity = numpy.array((s_d_intensity))
s_d_linewidth = numpy.array((s_d_linewidth))
s_d_position = numpy.array((s_d_position))




