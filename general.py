# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:54:04 2020

@author: lise
"""

import numpy as np
import os 
from scipy import signal 
# from scipy.optimize import curve_fit
import sys
import matplotlib.pyplot as plt



class ITS90LHe: 
    def __init__(self, domain):
        if domain == 'SF': 
            self.A = [1.392408, 0.527153, 0.166756, 0.050988,  0.026514, 0.001975, -0.017976, 0.005409,  0.013259, 0]
            self.B = 5.6
            self.C = 2.9
        elif domain == 'NF': 
            self.A = [3.146631, 1.357655, 0.413923, 0.091159,  0.016349, 0.001826, -0.004325, -0.004973,  0, 0]
            self.B = 10.3
            self.C = 1.9
        else: 
            sys.exit('Choose \'SF\' or \'NF\' as domains')
            
        self.fit = np.polynomial.polynomial.Polynomial(self.A)
    
    def getTemp(self, pressureIn): # [mbar]
        pressure = np.array(pressureIn) * 100 # [Pa]
        x = (np.log(pressure) - self.B)/self.C
        temp = self.fit(x)
        return temp


class ButterworthFilter: 
    def __init__(self, t, fc = 25, order = 6):
        self.type = 'low'
        
        self.time = t
        self.fc = fc #cut-off frequency
        self.order = order
        
        self.fs = 1 / (t[1] - t[0]) #sampling frequency
        self.fnyq = 0.5 * self.fs #nyquist frequency


        self.fnorm = self.fc / self.fnyq #normalized frequency
        self.b, self.a = signal.butter(order, self.fnorm, btype=self.type, analog=False)
        
    def filterSignal(self, dataArray):
        return signal.filtfilt(self.b, self.a, dataArray)
    
def mainsFilter(data, fs, Q): # Filters 50 Hz mains hum from signal with sampling frequency fs with a notch filter of quality Q
    w0 = 50/(fs/2) #fs/2 is the nyquist frequency?
    b, a = signal.iirnotch(w0, Q)
    out = signal.filtfilt(b, a, data)
    return out

#exponential function used to fit the temperature rise curve
def expFunction(x, a, b, c): 
    return b*(1 - np.exp(-x/a)) + c

#Polynomial function for heat input calibration
def polynomialFunction(x, a, b, c):
    return a + b*x + c*x**3

# #sine function to fit the pickup coil.
# def sineFunction(x, a, b):
#     return a*np.sin(b*x)

# def sinfunc(t, A, w, p, c):
#     return A*np.sin(w*t + p) + c

# def fit_sin(time, ydata): 
#     time = np.array(time)
#     ydata = np.array(ydata)
#     fft_freq = np.fft.rfftfreq(len(time), (time[1]-time[0]))
#     fft_ydata = abs(np.fft.rfft(ydata))
#     guess_freq = abs(fft_freq[np.argmax(fft_ydata[1:])+1])
#     guess_amp = np.std(ydata) * np.sqrt(2)
#     guess_offset = np.mean(ydata)
#     guess = np.array([guess_amp, 2*np.pi*guess_freq, 0., guess_offset])
#     print('Guesses:\nA = {},\tfreq = {}Hz,\tOffset = {}'.format(guess_amp, guess_freq, guess_offset))
#     boundary = ([np.std(ydata), 0.0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
#     popt, pcov = curve_fit(sinfunc, time, ydata, p0=guess, bounds=boundary, method='trf')
#     A, w, p, c = popt
#     f = w / (2.*np.pi)
#     fitfunc = lambda t: sinfunc(t, *popt)
#     print('Amp: {},\tOmega: {},\tPhase: {},\tOffset: {},\tFreq: {},\tPeriod: {},\tFitfunc: {},\tMaxcov: {},\tRawres: {}'.format(A, w, p, c, f, 1/f, fitfunc, np.max(pcov), (guess,popt,pcov)))
#     return fitfunc, A, guess_amp
    
    
def getCSV(folderName):
    os.chdir(folderName)
    # Opens a file dialog and acquires all files ending in '.csv' in the chosen folder
    dataFiles = []
    for file in os.listdir(os.getcwd()):
        if file.endswith('.csv'):
            dataFiles.append(file) 
    return dataFiles

#Used for D11T_GE02
def removeChannelName(df): 
    columnNames = df.columns
    measured = np.array(columnNames.get_level_values(1))
    df.columns = measured
    return df

#Used for MQXFS1R2-3
def renameColumns(df):
    if 'cDAQ1Mod3/ai0' in df.columns: 
        df.columns = [r'',r'hallPot', r'cxPot', r'hallMag', r'cxBO', r'pickUpCoil', r'cxTC', r'MKS', r'MKSpos', r'pt100', r'heater', r'pickUpBAD', r'time']
    else: 
        df.columns = [r'','hallPot', r'cxPot', r'hallMag', r'cxBO', r'cxTC', r'MKS', r'MKSpos', r'pt100', r'heater', r'pickUpBAD', r'time']
    return df

#Values obtaind by reading out mbar at MKS and voltage read out with Python.
class MKSfit: 
    def __init__(self): 
        self.p_mbar = np.array([0, 2.535, 2.895, 3.68, 5.16, 6.49, 8.14, 10.45, 12.81, 13.49, 16.31, 20.11, 22.945, 24.5, 41.55, 48.47, 63.78, 92.93])
        self.p_accuracy = self.p_mbar*0.00025
        
        self.voltage = np.array([-0.00033, 0.2533, 0.2878, 0.3663, 0.514, 0.64725, 0.8126, 1.0435, 1.2789, 1.3469, 1.62857, 2.00922, 2.29295, 2.44798, 4.1532, 4.84424, 6.3759, 9.29115])
        self.voltage_uncertainty = self.voltage*0.06/100 + 10.4*0.003/100
        # print(self.voltage_uncertainty)
    
        self.fit = np.polynomial.polynomial.Polynomial.fit(self.voltage, self.p_mbar, 1)
        
        
    def getPressure(self, voltage): 
        return self.fit(voltage)
    
    def plot(self):
        voltage = np.arange(0,10,0.1)
        pressure = self.getPressure(voltage)
        
        print('Coefficients: ', self.fit.convert().coef)
        
        
        
        plt.plot(voltage, pressure, label ='fit: p [mbar] = {:.4f} + {:.4f} V [V]'.format(self.fit.convert().coef[0],self.fit.convert().coef[1]))
        plt.errorbar(self.voltage, self.p_mbar, xerr=self.voltage_uncertainty, yerr=self.p_accuracy, fmt='o', label = 'MKS readout through NI 9209 card')
        plt.xlabel('Voltage [V]')
        plt.ylabel('Pressure [mbar]')
        plt.grid()
        plt.legend()
        plt.show()
    
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]



# fit = MKSfit()
# fit.plot()






