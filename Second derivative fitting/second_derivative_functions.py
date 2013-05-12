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

window = 11

###################################################################
######################### DIFFERENTIATING #########################

def basic_num_diff(x,y):
    diff = []
    for i in range(len(x)-1):
        delta_x = x[i+1]-x[i]
        delta_y = y[i+1]-y[i]
        diff.append(delta_y/delta_x)
    return diff

def triangular(x,m):
    smoothed_signal = []
    zeros = ((m-1)/2)
    ## establishing the weighting factors ## 
    weight1 = numpy.arange(1,((m/2)+1))
    weight2 = numpy.arange(1,((m/2)+2))
    w1 = weight2.tolist()
    w1.reverse()
    weight3 = numpy.array(w1)
    weight = numpy.concatenate((weight1,weight3))
   
    for i in range(zeros):
        smoothed_signal.append(0)

    for i in range(m/2,len(x)-(m/2)):
        ind1 = i-(m/2)
        ind2 = i+(m/2)
        x_smooth = x[ind1:ind2]
        y = []
        for j in range(len(x_smooth)):
            y.append(weight[j]*x_smooth[j])
        weighted_mean = numpy.sum(y)/(numpy.sum(weight))
        smoothed_signal.append(weighted_mean)

    for i in range(zeros):
        smoothed_signal.append(0)
  
    return smoothed_signal

###################################################################
######################## SECOND DERIVATIVE ########################
#def second_derivative(a,b,c,x):
#    num1 =  -(1 + ((x-b)/c)**2)
#    num2 = ((2*x - 2*b)**2)*(1/(c**2))
#    den = (1 + ((x-b)/c)**2)**3
#    return ((2*a/(c**2))*(num1+num2))/den

def second_derivative(a,b,c,x):
    lw = c
    pos = b
    intensity = a
    pi = numpy.pi
    constant = 2.0/(pi*lw)
    p1_num = -8*(lw**2 + 4*(pos-x)**2)**2
    p1_den = (lw**2 + 4*(pos-x)**2)**4
    p2_num = 16*((lw**2)*(pos-x) + 4*(pos-x)**3)*(8*(pos-x))
    p2_den = (lw**2 + 4*(pos-x)**2)**4
#    s_d = constant*((p1_num/p1_den) + (p2_num/p2_den))
    s_d = intensity*((p1_num/p1_den) + (p2_num/p2_den))
    return s_d

###################################################################
################### SECOND DERIVATIVE FUNCTIONS ##################

## THIRTY FIVE LORENTZIANS ## 
def combined_second_derivative_l35(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    d2ydx2_30 = second_derivative(p[87],p[88],p[89],x)
    d2ydx2_31 = second_derivative(p[90],p[91],p[92],x)
    d2ydx2_32 = second_derivative(p[93],p[94],p[95],x)
    d2ydx2_33 = second_derivative(p[96],p[97],p[98],x)
    d2ydx2_34 = second_derivative(p[99],p[100],p[101],x)
    d2ydx2_35 = second_derivative(p[102],p[103],p[104],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29 + d2ydx2_30 + d2ydx2_31 + d2ydx2_32 + d2ydx2_33 + d2ydx2_34 + d2ydx2_35
    return y

def residuals_l35(p,y,x):
    err = (y-combined_second_derivative_l35(p,x))
    return err

## THIRTY FOUR LORENTZIANS ## 
def combined_second_derivative_l34(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    d2ydx2_30 = second_derivative(p[87],p[88],p[89],x)
    d2ydx2_31 = second_derivative(p[90],p[91],p[92],x)
    d2ydx2_32 = second_derivative(p[93],p[94],p[95],x)
    d2ydx2_33 = second_derivative(p[96],p[97],p[98],x)
    d2ydx2_34 = second_derivative(p[99],p[100],p[101],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29 + d2ydx2_30 + d2ydx2_31 + d2ydx2_32 + d2ydx2_33 + d2ydx2_34
    return y

def residuals_l34(p,y,x):
    err = (y-combined_second_derivative_l34(p,x))
    return err

## THIRTY THREE LORENTZIANS ## 
def combined_second_derivative_l33(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    d2ydx2_30 = second_derivative(p[87],p[88],p[89],x)
    d2ydx2_31 = second_derivative(p[90],p[91],p[92],x)
    d2ydx2_32 = second_derivative(p[93],p[94],p[95],x)
    d2ydx2_33 = second_derivative(p[96],p[97],p[98],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29 + d2ydx2_30 + d2ydx2_31 + d2ydx2_32 + d2ydx2_33
    return y

def residuals_l33(p,y,x):
    err = (y-combined_second_derivative_l33(p,x))
    return err

## THIRTY TWO LORENTZIANS ## 
def combined_second_derivative_l32(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    d2ydx2_30 = second_derivative(p[87],p[88],p[89],x)
    d2ydx2_31 = second_derivative(p[90],p[91],p[92],x)
    d2ydx2_32 = second_derivative(p[93],p[94],p[95],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29 + d2ydx2_30 + d2ydx2_31 + d2ydx2_32
    return y

def residuals_l32(p,y,x):
    err = (y-combined_second_derivative_l32(p,x))
    return err

## THIRTY ONE LORENTZIANS ## 
def combined_second_derivative_l31(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    d2ydx2_30 = second_derivative(p[87],p[88],p[89],x)
    d2ydx2_31 = second_derivative(p[90],p[91],p[92],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29 + d2ydx2_30 + d2ydx2_31
    return y

def residuals_l31(p,y,x):
    err = (y-combined_second_derivative_l31(p,x))
    return err

## THIRTY LORENTZIANS ## 
def combined_second_derivative_l30(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    d2ydx2_30 = second_derivative(p[87],p[88],p[89],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29 + d2ydx2_30
    return y

def residuals_l30(p,y,x):
    err = (y-combined_second_derivative_l30(p,x))
    return err

## TWENTY NINE LORENTZIANS ## 
def combined_second_derivative_l29(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    d2ydx2_29 = second_derivative(p[84],p[85],p[86],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28 + d2ydx2_29
    return y

def residuals_l29(p,y,x):
    err = (y-combined_second_derivative_l29(p,x))
    return err

## TWENTY EIGHT LORENTZIANS ## 
def combined_second_derivative_l28(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    d2ydx2_28 = second_derivative(p[81],p[82],p[83],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27 + d2ydx2_28
    return y

def residuals_l28(p,y,x):
    err = (y-combined_second_derivative_l28(p,x))
    return err

## TWENTY SEVEN LORENTZIANS ## 
def combined_second_derivative_l27(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    d2ydx2_27 = second_derivative(p[78],p[79],p[80],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26 + d2ydx2_27
    return y

def residuals_l27(p,y,x):
    err = (y-combined_second_derivative_l27(p,x))
    return err

## TWENTY SIX LORENTZIANS ## 
def combined_second_derivative_l26(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    d2ydx2_26 = second_derivative(p[75],p[76],p[77],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25 + d2ydx2_26
    return y

def residuals_l26(p,y,x):
    err = (y-combined_second_derivative_l26(p,x))
    return err

## TWENTY FIVE LORENTZIANS ## 
def combined_second_derivative_l25(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    d2ydx2_25 = second_derivative(p[72],p[73],p[74],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24 + d2ydx2_25
    return y

def residuals_l25(p,y,x):
    err = (y-combined_second_derivative_l25(p,x))
    return err

## TWENTY FOUR LORENTZIANS ## 
def combined_second_derivative_l24(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    d2ydx2_24 = second_derivative(p[69],p[70],p[71],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23 + d2ydx2_24
    return y

def residuals_l24(p,y,x):
    err = (y-combined_second_derivative_l24(p,x))
    return err

## TWENTY THREE LORENTZIANS ## 
def combined_second_derivative_l23(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    d2ydx2_23 = second_derivative(p[66],p[67],p[68],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22 + d2ydx2_23
    return y

def residuals_l23(p,y,x):
    err = (y-combined_second_derivative_l23(p,x))
    return err

## TWENTY TWO LORENTZIANS ## 
def combined_second_derivative_l22(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    d2ydx2_22 = second_derivative(p[63],p[64],p[65],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21 + d2ydx2_22
    return y

def residuals_l22(p,y,x):
    err = (y-combined_second_derivative_l22(p,x))
    return err

## TWENTY ONE LORENTZIANS ## 
def combined_second_derivative_l21(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    d2ydx2_21 = second_derivative(p[60],p[61],p[62],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20 + d2ydx2_21
    return y

def residuals_l21(p,y,x):
    err = (y-combined_second_derivative_l21(p,x))
    return err

## TWENTY LORENTZIANS ## 
def combined_second_derivative_l20(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    d2ydx2_20 = second_derivative(p[57],p[58],p[59],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19 + d2ydx2_20
    return y

def residuals_l20(p,y,x):
    err = (y-combined_second_derivative_l20(p,x))
    return err

## NINETEEN LORENTZIANS ## 
def combined_second_derivative_l19(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    d2ydx2_19 = second_derivative(p[54],p[55],p[56],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18 + d2ydx2_19
    return y

def residuals_l19(p,y,x):
    err = (y-combined_second_derivative_l19(p,x))
    return err

## EIGHTEEN LORENTZIANS ## 
def combined_second_derivative_l18(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    d2ydx2_18 = second_derivative(p[51],p[52],p[53],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17 + d2ydx2_18
    return y

def residuals_l18(p,y,x):
    err = (y-combined_second_derivative_l18(p,x))
    return err

## SEVENTEEN LORENTZIANS ## 
def combined_second_derivative_l17(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    d2ydx2_17 = second_derivative(p[48],p[49],p[50],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16 + d2ydx2_17
    return y

def residuals_l17(p,y,x):
    err = (y-combined_second_derivative_l17(p,x))
    return err

## SIXTEEN LORENTZIANS ## 
def combined_second_derivative_l16(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    d2ydx2_16 = second_derivative(p[45],p[46],p[47],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15 + d2ydx2_16
    return y

def residuals_l16(p,y,x):
    err = (y-combined_second_derivative_l16(p,x))
    return err

## FIFTEEN LORENTZIANS ## 
def combined_second_derivative_l15(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    d2ydx2_15 = second_derivative(p[42],p[43],p[44],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14 + d2ydx2_15
    return y

def residuals_l15(p,y,x):
    err = (y-combined_second_derivative_l15(p,x))
    return err

## FOURTEEN LORENTZIANS ## 
def combined_second_derivative_l14(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    d2ydx2_14 = second_derivative(p[39],p[40],p[41],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13 + d2ydx2_14
    return y

def residuals_l14(p,y,x):
    err = (y-combined_second_derivative_l14(p,x))
    return err

## THIRTEEN LORENTZIANS ## 
def combined_second_derivative_l13(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    d2ydx2_13 = second_derivative(p[36],p[37],p[38],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 + d2ydx2_13
    return y

def residuals_l13(p,y,x):
    err = (y-combined_second_derivative_l13(p,x))
    return err

## TWELVE LORENTZIANS ## 
def combined_second_derivative_l12(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    d2ydx2_12 = second_derivative(p[33],p[34],p[35],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6 + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11 + d2ydx2_12 
    return y

def residuals_l12(p,y,x):
    err = (y-combined_second_derivative_l12(p,x))
    return err


## ELEVEN LORENTZIANS ## 
def combined_second_derivative_l11(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    d2ydx2_11 = second_derivative(p[30],p[31],p[32],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5   + d2ydx2_6  + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10 + d2ydx2_11
    return y

def residuals_l11(p,y,x):
    err = (y-combined_second_derivative_l11(p,x))
    return err

## TEN LORENTZIANS ## 
def combined_second_derivative_l10(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    d2ydx2_10 = second_derivative(p[27],p[28],p[29],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5   + d2ydx2_6  + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 + d2ydx2_10
    return y

def residuals_l10(p,y,x):
    err = (y-combined_second_derivative_l10(p,x))
    return err

## NINE LORENTZIANS ## 
def combined_second_derivative_l9(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    d2ydx2_9 = second_derivative(p[24],p[25],p[26],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5   + d2ydx2_6  + d2ydx2_7 + d2ydx2_8 + d2ydx2_9 
    return y

def residuals_l9(p,y,x):
    err = (y-combined_second_derivative_l9(p,x))
    return err

## EIGHT LORENTZIANS ## 
def combined_second_derivative_l8(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    d2ydx2_8 = second_derivative(p[21],p[22],p[23],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5   + d2ydx2_6  + d2ydx2_7 + d2ydx2_8 
    return y

def residuals_l8(p,y,x):
    err = (y-combined_second_derivative_l8(p,x))
    return err

## SEVEN LORENTZIANS ## 
def combined_second_derivative_l7(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    d2ydx2_7 = second_derivative(p[18],p[19],p[20],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5   + d2ydx2_6  + d2ydx2_7 
    return y

def residuals_l7(p,y,x):
    err = (y-combined_second_derivative_l7(p,x))
    return err

## SIX LORENTZIANS ## 
def combined_second_derivative_l6(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    d2ydx2_6 = second_derivative(p[15],p[16],p[17],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5 + d2ydx2_6  
    return y

def residuals_l6(p,y,x):
    err = (y-combined_second_derivative_l6(p,x))
    return err

## FIVE LORENTZIANS ## 
def combined_second_derivative_l5(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    d2ydx2_5 = second_derivative(p[12],p[13],p[14],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 + d2ydx2_5
    return y

def residuals_l5(p,y,x):
    err = (y-combined_second_derivative_l5(p,x))
    return err

## FOUR LORENTZIANS ## 
def combined_second_derivative_l4(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    d2ydx2_4 = second_derivative(p[9],p[10],p[11],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3 + d2ydx2_4 
    return y

def residuals_l4(p,y,x):
    err = (y-combined_second_derivative_l4(p,x))
    return err

## THREE LORENTZIANS ## 
def combined_second_derivative_l3(p,x):
    d2ydx2_1 = second_derivative(p[0],p[1],p[2],x)
    d2ydx2_2 = second_derivative(p[3],p[4],p[5],x)
    d2ydx2_3 = second_derivative(p[6],p[7],p[8],x)
    y = d2ydx2_1 + d2ydx2_2 + d2ydx2_3
    return y

def residuals_l3(p,y,x):
    err = (y-combined_second_derivative_l3(p,x))
    return err

###################################################################
############# SECOND DERIVATIVE INTENSITY CALIBRATION #############
###################################################################

#def second_derivative_intensity(intensity):
#    y = -13.707904208399375*intensity + 4.5815542268031606e-15
#    return y

#def second_derivative_intensity(intensity):
#    y = -7.520183858739415*intensity + 2.6656315501400189e-15
#    return y

def second_derivative_intensity(intensity):
    y = -9.287*intensity + 0.3078
    return y

###################################################################
############# SECOND DERIVATIVE UNCONSTRAINED FITTING #############
###################################################################

def fitting(x, y, intensities, peaks_spectra, linewidth): 

    if len(peaks_spectra) == 3:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l3(p,numpy.array(x))
        pbest = leastsq(residuals_l3,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l3(bestparams,numpy.array(x))

    if len(peaks_spectra) == 4:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l4(p,numpy.array(x))
        pbest = leastsq(residuals_l4,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l4(bestparams,numpy.array(x))

    if len(peaks_spectra) == 5:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l5(p,numpy.array(x))
        pbest = leastsq(residuals_l5,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l5(bestparams,numpy.array(x))

    if len(peaks_spectra) == 6:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l6(p,numpy.array(x))
        pbest = leastsq(residuals_l6,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l6(bestparams,numpy.array(x))

    if len(peaks_spectra) == 7:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l7(p,numpy.array(x))
        pbest = leastsq(residuals_l7,p,args=(y,x),full_output=1)
        bestparams = pbest[0]     
        df = combined_second_derivative_l7(bestparams,numpy.array(x))

    if len(peaks_spectra) == 8:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l8(p,numpy.array(x))
        pbest = leastsq(residuals_l8,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l8(bestparams,numpy.array(x))

    if len(peaks_spectra) == 9:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l9(p,numpy.array(x))
        pbest = leastsq(residuals_l9,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l9(bestparams,numpy.array(x))

    if len(peaks_spectra) == 10:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l10(p,numpy.array(x))
        pbest = leastsq(residuals_l10,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l10(bestparams,numpy.array(x))

    if len(peaks_spectra) == 11:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l11(p,numpy.array(x))
        pbest = leastsq(residuals_l11,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l11(bestparams,numpy.array(x))

    if len(peaks_spectra) == 12:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l12(p,numpy.array(x))
        pbest = leastsq(residuals_l12,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l12(bestparams,numpy.array(x))

    if len(peaks_spectra) == 13:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l13(p,numpy.array(x))
        pbest = leastsq(residuals_l13,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l13(bestparams,numpy.array(x))

    if len(peaks_spectra) == 14:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l14(p,numpy.array(x))
        pbest = leastsq(residuals_l14,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l14(bestparams,numpy.array(x))    


    if len(peaks_spectra) == 15:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l15(p,numpy.array(x))
        pbest = leastsq(residuals_l15,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l15(bestparams,numpy.array(x))

    if len(peaks_spectra) == 16:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l16(p,numpy.array(x))
        pbest = leastsq(residuals_l16,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l16(bestparams,numpy.array(x))

    if len(peaks_spectra) == 17:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l17(p,numpy.array(x))
        pbest = leastsq(residuals_l17,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l17(bestparams,numpy.array(x))

    if len(peaks_spectra) == 18:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l18(p,numpy.array(x))
        pbest = leastsq(residuals_l18,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l18(bestparams,numpy.array(x))

    if len(peaks_spectra) == 19:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l19(p,numpy.array(x))
        pbest = leastsq(residuals_l19,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l19(bestparams,numpy.array(x))
    
    if len(peaks_spectra) == 20:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l20(p,numpy.array(x))
        pbest = leastsq(residuals_l20,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l20(bestparams,numpy.array(x))
    
    if len(peaks_spectra) == 21:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l21(p,numpy.array(x))
        pbest = leastsq(residuals_l21,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l21(bestparams,numpy.array(x))

    if len(peaks_spectra) == 22:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l22(p,numpy.array(x))
        pbest = leastsq(residuals_l22,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l22(bestparams,numpy.array(x))

    if len(peaks_spectra) == 23:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l23(p,numpy.array(x))
        pbest = leastsq(residuals_l23,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l23(bestparams,numpy.array(x))

    if len(peaks_spectra) == 24:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l24(p,numpy.array(x))
        pbest = leastsq(residuals_l24,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l24(bestparams,numpy.array(x))

    if len(peaks_spectra) == 25:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l25(p,numpy.array(x))
        pbest = leastsq(residuals_l25,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l25(bestparams,numpy.array(x))

    if len(peaks_spectra) == 26:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l26(p,numpy.array(x))
        pbest = leastsq(residuals_l26,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l26(bestparams,numpy.array(x))

    if len(peaks_spectra) == 27:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l27(p,numpy.array(x))
        pbest = leastsq(residuals_l27,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l27(bestparams,numpy.array(x))

    if len(peaks_spectra) == 28:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l28(p,numpy.array(x))
        pbest = leastsq(residuals_l28,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l28(bestparams,numpy.array(x))

    if len(peaks_spectra) == 29:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l29(p,numpy.array(x))
        pbest = leastsq(residuals_l29,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l29(bestparams,numpy.array(x))

    if len(peaks_spectra) == 30:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l30(p,numpy.array(x))
        pbest = leastsq(residuals_l30,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l30(bestparams,numpy.array(x))

    if len(peaks_spectra) == 31:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l31(p,numpy.array(x))
        pbest = leastsq(residuals_l31,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l31(bestparams,numpy.array(x))

    if len(peaks_spectra) == 32:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l32(p,numpy.array(x))
        pbest = leastsq(residuals_l32,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l32(bestparams,numpy.array(x))

    if len(peaks_spectra) == 33:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l33(p,numpy.array(x))
        pbest = leastsq(residuals_l33,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l33(bestparams,numpy.array(x))

    if len(peaks_spectra) == 34:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l34(p,numpy.array(x))
        pbest = leastsq(residuals_l34,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l34(bestparams,numpy.array(x))

    if len(peaks_spectra) == 35:      
        p = []
        for j in range(len(peaks_spectra)):
            intensity = second_derivative_intensity(intensities[j])
            p.append(intensity)
            p.append(peaks_spectra[j])
            p.append(linewidth)
        trial = combined_second_derivative_l35(p,numpy.array(x))
        pbest = leastsq(residuals_l35,p,args=(y,x),full_output=1)
        bestparams = pbest[0]
        df = combined_second_derivative_l35(bestparams,numpy.array(x))

    return df, bestparams, trial

