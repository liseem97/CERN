# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:14:24 2020

@author: lmurberg
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pynverse import inversefunc
from scipy.signal import hilbert
 
import general as gen
import sensorCalibration as sC

import matplotlib.backends.backend_pdf

ITS90 = gen.ITS90LHe('SF')
MKSfit = gen.MKSfit()
cxCurrent = 3E-6

sensorCalib = sC.SensorCalibration()

# pdf = matplotlib.backends.backend_pdf.PdfPages("CoilHeating.pdf")

def f_inv(T): 
    temp = np.array([1.8,1.9,2.0,2.1])
    fSVP = np.array([9749.90, 12862.33, 10519.24, 2400.26])
    index = np.where(temp == T)
    return fSVP[index[0][0]]    

class heatInputCalib():
    def __init__(self): 
        self.folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
        self.file = r'HeatInputCalibration.csv'
        self.df = pd.read_csv('{}/{}'.format(self.folder, self.file))
        
        self.df_18K = self.df.groupby('filePressure').get_group(16.38)
        self.df_19K = self.df.groupby('filePressure').get_group(22.99)
        self.df_20K = self.df.groupby('filePressure').get_group(31.29)
        self.df_21K = self.df.groupby('filePressure').get_group(41.41)
        
        self.x_18K = self.df_18K['heatInput']/np.power((self.df_18K['MKStemp_stable']+self.df_18K['Tpot_ss'])/2,3)
        self.x_19K = self.df_19K['heatInput']/np.power((self.df_19K['MKStemp_stable']+self.df_19K['Tpot_ss'])/2,3)
        self.x_20K = self.df_20K['heatInput']/np.power((self.df_20K['MKStemp_stable']+self.df_20K['Tpot_ss'])/2,3)
        self.x_21K = self.df_21K['heatInput']/np.power((self.df_21K['MKStemp_stable']+self.df_21K['Tpot_ss'])/2,3)
        
        self.popt_18K, self.pcov_18K = curve_fit(gen.polynomialFunction, self.x_18K, self.df_18K['dTPot'])
        self.popt_19K, self.pcov_19K = curve_fit(gen.polynomialFunction, self.x_19K, self.df_19K['dTPot'])
        self.popt_20K, self.pcov_20K = curve_fit(gen.polynomialFunction, self.x_20K, self.df_20K['dTPot'])
        self.popt_21K, self.pcov_21K = curve_fit(gen.polynomialFunction, self.x_21K, self.df_21K['dTPot'])
        
        
        self.fitFunc_18K = (lambda x: gen.polynomialFunction(x,*self.popt_18K))
        self.invFunc_18K = inversefunc(self.fitFunc_18K)
        
        self.fitFunc_19K = (lambda x: gen.polynomialFunction(x,*self.popt_19K))
        self.invFunc_19K = inversefunc(self.fitFunc_19K)
        
        self.fitFunc_20K = (lambda x: gen.polynomialFunction(x,*self.popt_20K))
        self.invFunc_20K = inversefunc(self.fitFunc_20K)
        
        self.fitFunc_21K = (lambda x: gen.polynomialFunction(x,*self.popt_21K))
        self.invFunc_21K = inversefunc(self.fitFunc_21K)
        
        
        #Information about fit
        self.RMSE_18K = np.sqrt(np.mean(np.power(self.df_18K['dTPot'] - gen.polynomialFunction(self.x_18K, *self.popt_18K), 2)))
        self.RMSEinv_18K = np.sqrt(np.mean(np.power(self.x_18K - self.invFunc_18K(self.df_18K['dTPot']),2)))
        self.RMSEinv_18K_Q = np.sqrt(np.mean(np.power(self.df_18K['heatInput'] - self.invFunc_18K(self.df_18K['dTPot'])* np.power((self.df_18K['MKStemp_stable']+self.df_18K['Tpot_ss'])/2,3),2)))
        
        self.residual_18K = self.df_18K['heatInput'] - self.invFunc_18K(self.df_18K['dTPot'])* np.power((self.df_18K['MKStemp_stable']+self.df_18K['Tpot_ss'])/2,3)


        self.RMSE_19K = np.sqrt(np.mean(np.power(self.df_19K['dTPot'] - gen.polynomialFunction(self.x_19K, *self.popt_19K), 2)))
        self.RMSEinv_19K = np.sqrt(np.mean(np.power(self.x_19K - self.invFunc_19K(self.df_19K['dTPot']),2)))
        self.RMSEinv_19K_Q = np.sqrt(np.mean(np.power(self.df_19K['heatInput'] - self.invFunc_19K(self.df_19K['dTPot'])* np.power((self.df_19K['MKStemp_stable']+self.df_19K['Tpot_ss'])/2,3),2)))
        
        self.residual_19K = self.df_19K['heatInput'] - self.invFunc_19K(self.df_19K['dTPot'])* np.power((self.df_19K['MKStemp_stable']+self.df_19K['Tpot_ss'])/2,3)

        
        self.RMSE_20K = np.sqrt(np.mean(np.power(self.df_20K['dTPot'] - gen.polynomialFunction(self.x_20K, *self.popt_20K), 2)))
        self.RMSEinv_20K = np.sqrt(np.mean(np.power(self.x_20K - self.invFunc_20K(self.df_20K['dTPot']),2)))
        self.RMSEinv_20K_Q = np.sqrt(np.mean(np.power(self.df_20K['heatInput'] - self.invFunc_20K(self.df_20K['dTPot'])* np.power((self.df_20K['MKStemp_stable']+self.df_20K['Tpot_ss'])/2,3),2)))
        
        self.residual_20K = self.df_20K['heatInput'] - self.invFunc_20K(self.df_20K['dTPot'])* np.power((self.df_20K['MKStemp_stable']+self.df_20K['Tpot_ss'])/2,3)

        
        self.RMSE_21K = np.sqrt(np.mean(np.power(self.df_21K['dTPot'] - gen.polynomialFunction(self.x_21K, *self.popt_21K), 2)))
        self.RMSEinv_21K = np.sqrt(np.mean(np.power(self.x_21K - self.invFunc_21K(self.df_21K['dTPot']),2)))
        self.RMSEinv_21K_Q = np.sqrt(np.mean(np.power(self.df_21K['heatInput'] - self.invFunc_21K(self.df_21K['dTPot'])* np.power((self.df_21K['MKStemp_stable']+self.df_21K['Tpot_ss'])/2,3),2)))
        
        self.residual_21K = self.df_21K['heatInput'] - self.invFunc_21K(self.df_21K['dTPot'])* np.power((self.df_21K['MKStemp_stable']+self.df_21K['Tpot_ss'])/2,3)

        
    def getHeatInput(self, filePressure, MKStemp, dTpot): 
        if filePressure == 16.38: 
            x = self.invFunc_18K(dTpot)
            heat = x * np.power(MKStemp,3)
            return heat
        elif filePressure == 22.99: 
            x = self.invFunc_19K(dTpot)
            heat = x * np.power(MKStemp,3)
            return heat
        elif filePressure == 31.29: 
            x = self.invFunc_20K(dTpot)
            heat = x * np.power(MKStemp,3)
            return heat
        elif filePressure == 41.41: 
            x = self.invFunc_21K(dTpot)
            heat = x * np.power(MKStemp,3)
            return heat
        else: 
            print('Error: Insert correct filePressure')
            exit
        
        
    def plot(self):
        plt.scatter(self.df_18K['dTPot'], self.df_18K['heatInput'], label = '1.8 K', c = 'y', marker ='x')#, c = df_18K['Unnamed: 0'])
        plt.scatter(self.df_19K['dTPot'], self.df_19K['heatInput'], label = '1.9 K', c = 'r', marker ='x')#, c = df_19K['Unnamed: 0'])
        plt.scatter(self.df_20K['dTPot'], self.df_20K['heatInput'], label = '2.0 K', c = 'g', marker ='x')#, c = df_19K['Unnamed: 0'])
        plt.scatter(self.df_21K['dTPot'], self.df_21K['heatInput'], label = '2.1 K', c = 'b', marker ='x')#, c = df_21K['Unnamed: 0'])
        
        dT = np.arange(0,16,0.1)
        
        plt.plot(dT, self.invFunc_18K(dT)* np.power(1.8,3), label = 'fit 1.8 K', c = 'y')#, c = df_18K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_19K(dT)* np.power(1.9,3), label = 'fit 1.9 K', c = 'r')#, c = df_19K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_20K(dT)* np.power(2.0,3), label = 'fit 2.0 K', c = 'g')#, c = df_19K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_21K(dT)* np.power(2.1,3), label = 'fit 2.1 K', c = 'b')#, c = df_21K['Unnamed: 0'])
        
        plt.title('Heat Input Calibration')
        plt.xlabel('$dT_{pot}$ [mK]')
        plt.ylabel('Q [mW]')
        # plt.colorbar()
        # plt.ylim([-50,400])
        # plt.xlim([-0.5,5])
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.scatter(self.df_18K.loc[self.df_18K['dTPot']<3,'dTPot'], self.df_18K.loc[self.df_18K['dTPot']<3,'heatInput'], label = '1.8 K', c = 'y', marker ='x')#, c = df_18K['Unnamed: 0'])
        plt.scatter(self.df_19K.loc[self.df_19K['dTPot']<3,'dTPot'], self.df_19K.loc[self.df_19K['dTPot']<3,'heatInput'], label = '1.9 K', c = 'r', marker ='x')#, c = df_19K['Unnamed: 0'])
        plt.scatter(self.df_20K.loc[self.df_20K['dTPot']<3,'dTPot'], self.df_20K.loc[self.df_20K['dTPot']<3,'heatInput'], label = '2.0 K', c = 'g', marker ='x')#, c = df_19K['Unnamed: 0'])
        plt.scatter(self.df_21K.loc[self.df_21K['dTPot']<3,'dTPot'], self.df_21K.loc[self.df_21K['dTPot']<3,'heatInput'], label = '2.1 K', c = 'b', marker ='x')#, c = df_21K['Unnamed: 0'])
        
        dT = np.arange(0,3,0.1)
        
        plt.plot(dT, self.invFunc_18K(dT)* np.power(1.8,3), label = 'fit 1.8 K', c = 'y')#, c = df_18K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_19K(dT)* np.power(1.9,3), label = 'fit 1.9 K', c = 'r')#, c = df_19K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_20K(dT)* np.power(2.0,3), label = 'fit 2.0 K', c = 'g')#, c = df_19K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_21K(dT)* np.power(2.1,3), label = 'fit 2.1 K', c = 'b')#, c = df_21K['Unnamed: 0'])
        
        plt.title('Heat Input Calibration')
        plt.xlabel('$dT_{pot}$ [mK]')
        plt.ylabel('Q [mW]')
        # plt.colorbar()
        # plt.ylim([-50,400])
        # plt.xlim([-0.5,5])
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.scatter(self.df_18K['dTPot'], self.x_18K, label = '1.8 K', c = 'y', marker ='x')#, c = df_18K['Unnamed: 0'])
        plt.scatter(self.df_19K['dTPot'], self.x_19K, label = '1.9 K', c = 'r', marker ='x')#, c = df_19K['Unnamed: 0'])
        plt.scatter(self.df_20K['dTPot'], self.x_20K, label = '2.0 K', c = 'g', marker ='x')#, c = df_19K['Unnamed: 0'])
        plt.scatter(self.df_21K['dTPot'], self.x_21K, label = '2.1 K', c = 'b', marker ='x')#, c = df_21K['Unnamed: 0'])
        
        dT = np.arange(0,16,0.1)
        
        plt.plot(dT, self.invFunc_18K(dT), label = 'fit 1.8 K', c = 'y')#, c = df_18K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_19K(dT), label = 'fit 1.9 K', c = 'r')#, c = df_19K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_20K(dT), label = 'fit 2.0 K', c = 'g')#, c = df_19K['Unnamed: 0'])
        plt.plot(dT, self.invFunc_21K(dT), label = 'fit 2.1 K', c = 'b')#, c = df_21K['Unnamed: 0'])
        
        plt.title('Heat Input Calibration')
        plt.xlabel('$dT_{pot}$ [mK]')
        plt.ylabel('Q/T^3 [mW/K^3]')
        # plt.colorbar()
        # plt.ylim([-50,400])
        # plt.xlim([-0.5,5])
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.title('Residual 1.8 K')
        plt.scatter(self.df_18K['heatInput'], self.residual_18K)
        plt.ylabel('Residual [mW]')
        plt.xlabel('Heat Input [mW]')
        plt.grid()
        plt.show()
        
        
        print('Coeff 1.8 K: ',*self.popt_18K)        
        print('Kapitza  for 1.8 K (b = 1 / 4o): {} \n'.format(self.popt_18K[1]))
        print('Wide channel for 1.8 K (dx = c /T^9 * f-1) : {} \n'.format(f_inv(1.8) * self.popt_18K[2] / 1.8**9))
        print('RMSE error 1.8 K: ', self.RMSE_18K, ' mK')
        print('RMSE error after inversed function 1.8 K: ', self.RMSEinv_18K, 'mW/T^3')
        print('RMSE error after inversed function 1.8 K: ', self.RMSEinv_18K_Q, 'mW\n\n')
        
        plt.title('Residual 1.9 K')
        plt.scatter(self.df_19K['heatInput'], self.residual_19K)
        plt.ylabel('Residual [mW]')
        plt.xlabel('Heat Input [mW]')
        plt.grid()
        plt.show()
        
        print('Coeff 1.9 K: ',*self.popt_19K)     
        print('Kapitza  for 1.9 K (b = 1 / 4o): {} \n'.format(self.popt_19K[1]))
        print('Wide channel for 1.9 K (dx = c /T^9 * f-1) : {} \n'.format(f_inv(1.9) * self.popt_19K[2] / 1.9**9))
        print('RMSE error 1.9 K: ', self.RMSE_19K, ' mK')
        print('RMSE error after inversed function 1.9 K: ', self.RMSEinv_19K, 'mW/T^3')
        print('RMSE error after inversed function 1.9 K: ', self.RMSEinv_19K_Q, 'mW\n\n')
        
        plt.title('Residual 2.0 K')
        plt.scatter(self.df_20K['heatInput'], self.residual_20K)
        plt.ylabel('Residual [mW]')
        plt.xlabel('Heat Input [mW]')
        plt.grid()
        plt.show()
        
        print('Coeff 2.0 K: ',*self.popt_20K)        
        print('Kapitza  for 2.0 K (b = 1 / 4o): {} \n'.format(self.popt_20K[1]))
        print('Wide channel for 2.0 K (dx = c /T^9 * f-1) : {} \n'.format(f_inv(2.0) * self.popt_20K[2] / 2.0**9))
        print('RMSE error 2.0 K: ', self.RMSE_20K, ' mK')
        print('RMSE error after inversed function 2.0 K: ', self.RMSEinv_20K, 'mW/T^3')
        print('RMSE error after inversed function 2.0 K: ', self.RMSEinv_20K_Q, 'mW\n\n')
        
        plt.title('Residual 2.1 K')
        plt.scatter(self.df_21K['heatInput'], self.residual_21K)
        plt.ylabel('Residual [mW]')
        plt.xlabel('Heat Input [mW]')
        plt.grid()
        plt.show()
        
        print('Coeff 2.1 K: ',*self.popt_21K)       
        print('Kapitza  for 2.1 K (b = 1 / 4o): {} \n'.format(self.popt_21K[1]))
        print('Wide channel for 2.1 K (dx = c /T^9 * f-1) : {} \n'.format(f_inv(2.1) * self.popt_21K[2] / 2.1**9))
        print('RMSE error 2.1 K: ', self.RMSE_21K, ' mK')
        print('RMSE error after inversed function 2.1 K: ', self.RMSEinv_21K, 'mW/T^3')
        print('RMSE error after inversed function 2.1 K: ', self.RMSEinv_21K_Q, 'mW\n\n')

hic = heatInputCalib()
# hic.plot()
    

class CoilHeatingData(): 
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        self.info = file.split('_')[0]
        
        print(self.file)
        
        #Extract information from the file name:
        self.filePressure = float(file.split('_')[1]+'.'+file.split('_')[2].strip('mbar')) 
        self.fileTemp = ITS90.getTemp(self.filePressure)

        self.fileVoltage = float(file.split('_')[7].strip('Coil')+'.'+file.split('_')[8].strip('V'))
        self.fileFreq = float(file.split('_')[9]+'.'+file.split('_')[10].strip('Hz'))
        self.fileResistance = float(file.split('_')[11]+'.'+file.split('_')[12].strip('Ohm'))
        
        df = gen.removeChannelName(pd.read_csv('{}/{}'.format(folder, file), header = [0,1]))   
        self.rawData = df
        self.timeFull = df['Time']
        
        #100:-100 to remove spikes in temperature measurements. 
        spikeStart = 100
        spikeEnd = -100
        
        #Time without spikes for plotting purposes
        self.time = self.timeFull[spikeStart:spikeEnd].to_numpy()
        
        #PickUp coil
        self.pickUpCoil = df['pickUpCoil']
        
        #Choosing of trigger area/ Area where the coil is powered.
        noise = 5E-4 #Noise in pickup coil is less than this, but when we have a signal in the pickup coil its always more than this. 
        
        #for mesurements where i use the full range.
        self.trgStartFull = np.where(self.pickUpCoil > noise)[0][0]
        self.trgEndFull = np.where(self.pickUpCoil > noise)[0][-1]
        
        #for measurements where i have taken out spikes
        self.trgStart = self.trgStartFull - spikeStart
        self.trgEnd = self.trgEndFull - spikeStart
        

        #One second of the pickup coil (for plotting only)
        pucRange =  [np.where(np.array(self.timeFull,dtype=int) == 20)[0][0], np.where(np.array(self.timeFull,dtype=int) == 21)[0][0]]
        pickUpCoil_zoom =  self.pickUpCoil[pucRange[0]:pucRange[1]]
        time_zoom = self.time[pucRange[0]:pucRange[1]]
        
        #To get the amplitude of the signa
        #Hilbert transform to get the amplitude of the pickupcoil. more info at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        analytic_signal = hilbert(self.pickUpCoil[self.trgStartFull:self.trgEndFull])  
        self.amplitude = np.mean(np.abs(analytic_signal))
        self.amplitudeSTD = np.std(np.abs(analytic_signal))
        self.amplitudeSEM = self.amplitudeSTD / np.sqrt(len(analytic_signal))
        
        #get the exact frequency from the fourier transform 
        fft = np.fft.fft(self.pickUpCoil)
        fft_freq = np.fft.fftfreq(len(self.timeFull), self.timeFull[1]-self.timeFull[0])
        i = np.where(fft == np.max(fft))[0][0]
        self.frequency = np.abs(fft_freq[i])
        
        #Analysing the Absolute He bath temperature from the MKS signal. 
        self.MKSpos = self.rawData['MKSpos']*10 #in prosentage open
        self.MKSpressureArray = MKSfit.getPressure(self.rawData['MKS'])
        self.MKStempArray = ITS90.getTemp(self.MKSpressureArray)
        
        #Mean values for the whole range: 
        self.MKSpressure = np.mean(self.MKSpressureArray)
        self.MKSpressureSTD = np.std(self.MKSpressureArray)
        self.MKSpressureSEM = self.MKSpressureSTD / np.sqrt(len(self.MKSpressureArray))
        
        self.MKStemp = np.mean(self.MKStempArray)
        self.MKStempSTD = np.std(self.MKStempArray)
        self.MKStempSEM = self.MKStempSTD / np.sqrt(len(self.MKStempArray))
        
        
        butter = gen.ButterworthFilter(self.timeFull, self.fileFreq/2 , order = 6) 

        self.lengthMeasurement = np.array(self.timeFull)[-1]
        
        
        #Voltage values
        self.cxPot_V = butter.filterSignal(self.rawData['cxPot'].to_numpy())[spikeStart:spikeEnd]
        self.cxBR_V = butter.filterSignal(self.rawData['cxBR'].to_numpy())[spikeStart:spikeEnd]
        self.cxTL_V = butter.filterSignal(self.rawData['cxTL'].to_numpy())[spikeStart:spikeEnd]
        self.cxBL_V = butter.filterSignal(self.rawData['cxBL'].to_numpy())[spikeStart:spikeEnd]
        self.cxTR_V = butter.filterSignal(self.rawData['cxTR'].to_numpy())[spikeStart:spikeEnd]
        
        #Resistance values
        self.cxPot_R = self.cxPot_V / cxCurrent
        self.cxBR_R = self.cxBR_V / cxCurrent
        self.cxTL_R = self.cxTL_V / cxCurrent
        self.cxBL_R = self.cxBL_V / cxCurrent
        self.cxTR_R = self.cxTR_V / cxCurrent
        
        #Temperature values full range
        self.cxPot_T = sensorCalib.Pot(self.cxPot_R)
        self.cxBR_T = sensorCalib.BR(self.cxBR_R)
        self.cxTL_T = sensorCalib.TL(self.cxTL_R)
        self.cxBL_T = sensorCalib.BL(self.cxBL_R)
        self.cxTR_T = sensorCalib.TR(self.cxTR_R)
        
        
        fig = plt.figure()
        plt.title('All temperature measurements \n{}'.format(self.file))
        plt.plot(self.time, self.cxPot_T, label = 'Pot')
        plt.plot(self.time, self.cxBR_T, label = 'BR')
        plt.plot(self.time, self.cxBL_T, label = 'BL')
        plt.plot(self.time, self.cxTR_T, label = 'TR')
        plt.plot(self.time, self.cxTL_T, label = 'TL')
        plt.plot(self.time, self.MKStempArray[spikeStart:spikeEnd], label = 'MKS temp')
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [K]')
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
    
        
        fig, ax = plt.subplots(2)
        fig.suptitle('Pick-Up Coil, f = {:.2f} Hz\n{}'.format(self.frequency, self.file))
        ax[0].plot(self.timeFull, self.pickUpCoil, label = 'measurement')
        ax[0].plot(self.time,np.full(len(self.time),self.amplitude), label = 'A = {:.2f}mV'.format(self.amplitude*1000))
        ax[0].scatter(self.timeFull[self.trgStartFull], self.pickUpCoil[self.trgStartFull], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        ax[0].scatter(self.timeFull[self.trgEndFull], self.pickUpCoil[self.trgEndFull], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Voltage [V]')
        ax[0].grid()
        
        ax[1].plot(time_zoom, pickUpCoil_zoom,)
        ax[1].plot(time_zoom,np.full(len(time_zoom),self.amplitude))
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Voltage [V]')
        ax[1].grid()
        
        fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        # fig.show()
        
        fig, ax = plt.subplots(2)
        fig.suptitle('MKS pressure and position \n{}'.format(self.file))
        ax[0].plot(self.timeFull, self.MKSpressureArray, label = 'MKS pressure')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Pressure [mbar]')
        ax[0].grid()
        
        ax[1].plot(self.timeFull, self.MKSpos, label = 'MKS position', c= 'tab:orange')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Position open [%]')
        ax[1].grid()
        
        fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        # fig.show()
        
        
        self.time_trgRange = self.timeFull[self.trgStartFull:self.trgEndFull] - self.timeFull[self.trgStartFull]
        self.len_trgRange = len(self.time_trgRange)
        
        
        self.MKS_init = np.mean(self.MKStempArray[:self.trgStart])
        self.MKS_init_STD = np.std(self.MKStempArray[:self.trgStart])
        self.MKS_init_SEM = self.MKS_init_STD / np.sqrt(len(self.MKStempArray[:self.trgStart]))
        
        self.MKS_trgRange = self.MKStempArray[self.trgStart:self.trgEnd]
        
        self.MKS_stabilized = np.mean(self.MKS_trgRange[-int(self.len_trgRange/2):]) 
        self.MKS_stabilized_STD = np.std(self.MKS_trgRange[-int(self.len_trgRange/2):])
        self.MKS_stabilized_SEM = self.MKS_stabilized_STD / np.sqrt(len(self.MKS_trgRange[-int(self.len_trgRange/2):]))
        
        
        self.MKS_dT = float(self.MKS_stabilized - self.MKS_init)*1000
        
        
        fig = plt.figure()
        plt.title('MKS temperature, dT = {} mK \n{}'.format(self.MKS_dT, self.file))
        plt.plot(self.time, self.MKStempArray[spikeStart:spikeEnd], label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.MKS_stabilized), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange/2):]+19, np.full(len(self.time_trgRange[-int(self.len_trgRange/2):]), self.MKS_stabilized), c = 'tab:orange', linewidth = 4, label = 'MKS_stabilized = {:.4f}'.format(self.MKS_stabilized))
    
        plt.plot(self.time, np.full(len(self.time), self.MKS_init), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:self.trgStart], np.full(len(self.time[:self.trgStart]), self.MKS_init), c = 'tab:green', linewidth = 4, label = r'MKS_init = {:.4f}'.format(self.MKS_init))
        
        plt.scatter(self.time[self.trgStart], self.MKStempArray[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[self.trgEnd], self.MKStempArray[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        #Pot sensor
        self.cxPot_Tinit = np.mean(sensorCalib.Pot(self.cxPot_R[:self.trgStart]))
        self.cxPot_Tinit_STD = np.std(sensorCalib.Pot(self.cxPot_R[:self.trgStart]))
        self.cxPot_Tinit_SEM = self.cxPot_Tinit_STD / np.sqrt(len(self.cxPot_R[:self.trgStart]))
        
        
        self.cxPot_T_trgRange = sensorCalib.Pot(self.cxPot_R[self.trgStart:self.trgEnd])

        self.cxPot_Tss_trg = np.mean(self.cxPot_T_trgRange[-int(self.len_trgRange/10):])
        self.cxPot_Tss_trg_STD = np.std(self.cxPot_T_trgRange[-int(self.len_trgRange/10):])
        self.cxPot_Tss_trg_SEM = self.cxPot_Tss_trg_STD / np.sqrt(len(self.cxPot_T_trgRange[-int(self.len_trgRange/10):]))
        
        self.cxPot_dT_trg = float(self.cxPot_Tss_trg-self.cxPot_Tinit)
        
        self.cxPot_fitcoeff, self.cxPot_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxPot_T_trgRange, p0 = [1, self.cxPot_dT_trg, self.cxPot_Tinit])
       
        #Steady state values for the temperature when the coil is powered
        self.cxPot_Tss_fit = self.cxPot_fitcoeff[1] + self.cxPot_fitcoeff[2]
       
        # Errors in fit
        self.cxPot_fit_rmse = np.sqrt(np.mean(np.power(gen.expFunction(self.time_trgRange,*self.cxPot_fitcoeff)-self.cxPot_T_trgRange,2)))
        
        #Tau values (i.e. time to reach 90?% of ss)
        self.cxPot_tau = self.cxPot_fitcoeff[0]
        self.cxPot_dT_fit = float(self.cxPot_Tss_fit - self.cxPot_Tinit)*1000    
        
        self.cxPot_dT_trg *=1000
        
        tol = np.sqrt(np.power(self.cxPot_Tinit_STD*1000,2) + np.power(self.cxPot_Tss_trg_STD*1000,2)) # mK
        if np.abs(self.cxPot_dT_trg - self.cxPot_dT_trg) < tol: 
            self.cxPot_SS = True
        else: 
            self.cxPot_SS = False
            
        tol = np.sqrt(np.power(self.MKS_init_STD*1000,2) + np.power(self.MKS_stabilized_STD*1000,2)) # mK
        if np.abs(self.cxPot_dT_trg - self.MKS_dT) < tol: 
            self.cxPot_uncertain = True
        else: 
            self.cxPot_uncertain = False
            
        self.cxPot_dT_trg_MKS = self.cxPot_dT_trg - self.MKS_dT
        self.cxPot_dT_fit_MKS = self.cxPot_dT_fit - self.MKS_dT
        
        #Heat from heat input calibration
        self.heat_trg = hic.getHeatInput(self.filePressure, (self.MKS_stabilized+self.cxPot_Tss_trg)/2, self.cxPot_dT_trg)
        self.heat_fit = hic.getHeatInput(self.filePressure, (self.MKS_stabilized+self.cxPot_Tss_fit)/2, self.cxPot_dT_fit)
        
        
        self.heat_MKS_trg = hic.getHeatInput(self.filePressure, (self.MKS_stabilized+self.cxPot_Tss_trg)/2, self.cxPot_dT_trg-self.MKS_dT)
        self.heat_MKS_fit = hic.getHeatInput(self.filePressure, (self.MKS_stabilized+self.cxPot_Tss_fit)/2, self.cxPot_dT_fit-self.MKS_dT)
        
        fig = plt.figure()
        
        plt.title('Cernox Pot Sensor, dT_trg = {} mK \n{}'.format(self.cxPot_dT_trg, self.file))
        plt.plot(self.time, self.cxPot_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange/10):]+19, np.full(len(self.time_trgRange[-int(self.len_trgRange/10):]), self.cxPot_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss_trg = {:.4f}'.format(self.cxPot_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:self.trgStart], np.full(len(self.time[:self.trgStart]), self.cxPot_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxPot_Tinit))
        
        plt.scatter(self.time[self.trgStart], self.cxPot_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[self.trgEnd], self.cxPot_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.title('Cernox Pot Sensor, dT_fit = {} mK \n{}'.format(self.cxPot_dT_fit, self.file))
        plt.plot(self.time, self.cxPot_T, label = 'measurement', c = 'tab:blue')
        plt.plot(self.time_trgRange + 19, gen.expFunction(self.time_trgRange, *self.cxPot_fitcoeff), c = 'tab:cyan', label = 'exponential fit')

        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tss_trg), linestyle='dashed', c = 'tab:orange',label = 'T_ss_trg = {:.4f}'.format(self.cxPot_Tss_trg))
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tss_fit), linestyle='dashed', c = 'tab:pink', label = 'T_ss_fit = {:.4f}'.format(self.cxPot_Tss_fit))
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tinit), linestyle='dashed', c = 'tab:green', label = r'T_init = {:.4f}'.format(self.cxPot_Tinit))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
    
        
        #outerlayer (top)
        #initial values
        self.cxTR_Tinit =  np.mean(sensorCalib.TR(self.cxTR_R[:self.trgStart]))
        self.cxTR_Tinit_STD =  np.std(sensorCalib.TR(self.cxTR_R[:self.trgStart]))
        self.cxTR_Tinit_SEM = self.cxTR_Tinit_STD / np.sqrt(len(self.cxTR_R[:self.trgStart]))
        
        self.cxTL_Tinit =  np.mean(sensorCalib.TL(self.cxTL_R[:self.trgStart]))
        self.cxTL_Tinit_STD =  np.std(sensorCalib.TL(self.cxTL_R[:self.trgStart]))
        self.cxTL_Tinit_SEM = self.cxTL_Tinit_STD / np.sqrt(len(self.cxTL_R[:self.trgStart]))
        

        #Area to be used when fitting exponential curve: (from trigger to about middle of measurement)
        self.cxTR_T_trgRange = sensorCalib.TR(self.cxTR_R[self.trgStart:self.trgEnd])
        self.cxTL_T_trgRange = sensorCalib.TL(self.cxTL_R[self.trgStart:self.trgEnd])
        
        ssArea = 0.9
        self.cxTR_Tss_trg = np.mean(self.cxTR_T_trgRange[-int(self.len_trgRange*ssArea):-1]) 
        self.cxTR_Tss_trg_STD = np.std(self.cxTR_T_trgRange[-int(self.len_trgRange*ssArea):-1])
        self.cxTR_Tss_trg_SEM = self.cxTR_Tss_trg_STD / np.sqrt(len(self.cxTR_T_trgRange[-int(self.len_trgRange*ssArea):-1]))
        
        self.cxTL_Tss_trg = np.mean(self.cxTL_T_trgRange[-int(self.len_trgRange*ssArea):-1])  
        self.cxTL_Tss_trg_STD = np.std(self.cxTL_T_trgRange[-int(self.len_trgRange*ssArea):-1])
        self.cxTL_Tss_trg_SEM = self.cxTL_Tss_trg_STD / np.sqrt(len(self.cxTL_T_trgRange[-int(self.len_trgRange*ssArea):-1]))
        
        #A initial guess of the dT for each sensor when triggering
        self.cxTR_dT_trg = float(self.cxTR_Tss_trg-self.cxTR_Tinit) #in K to use it in curve_fit
        self.cxTL_dT_trg = float(self.cxTL_Tss_trg-self.cxTL_Tinit)
        
         #Find time constant tau: 
        self.cxTR_tempTau = self.cxTR_dT_trg * (1 - 1/np.e) + self.cxTR_Tinit
        self.cxTR_tauIndex = np.where(self.cxTR_T == gen.find_nearest(self.cxTR_T[:int(self.len_trgRange/2)], self.cxTR_tempTau))[0][0]
        self.TR_tau = float(self.time[self.cxTR_tauIndex] - self.time[self.trgStart])
        # print('tau TR new: ', self.TR_tau)
        
        self.cxTL_tempTau = self.cxTL_dT_trg * (1 - 1/np.e) + self.cxTL_Tinit
        self.cxTL_tauIndex = np.where(self.cxTL_T == gen.find_nearest(self.cxTL_T[:int(self.len_trgRange/2)], self.cxTL_tempTau))[0][0]
        self.TL_tau = float(self.time[self.cxTL_tauIndex] - self.time[self.trgStart])
        # print('tau TL new: ', self.TL_tau)
        
  
        # #Exponentional curve fit of the rise in temperature
        # self.cxTR_fitcoeff, self.cxTR_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxTR_T_trgRange, p0 = [1, self.cxTR_dT_trg, self.cxTR_Tinit])
        # # print(self.cxTL_T_trgRange)
        # self.cxTL_fitcoeff, self.cxTL_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxTL_T_trgRange, p0 = [1, self.cxTL_dT_trg, self.cxTL_Tinit])
        
        # #Steady state values for the temperature when the coil is powered
        # self.cxTR_Tss_fit = self.cxTR_fitcoeff[1] + self.cxTR_fitcoeff[2]
        # self.cxTL_Tss_fit = self.cxTL_fitcoeff[1] + self.cxTL_fitcoeff[2]
        
        # #Tau values (i.e. time to reach 90?% of ss)
        # self.cxTR_tau = self.cxTR_fitcoeff[0]
        # self.cxTL_tau = self.cxTL_fitcoeff[0]
        
        # #Calculated dT values
        # self.cxTR_dT_fit =  float(self.cxTR_Tss_fit - self.cxTR_Tinit)*1000        
        # self.cxTL_dT_fit =  float(self.cxTL_Tss_fit - self.cxTL_Tinit)*1000
        
        self.cxTR_dT_trg *= 1000  #to get mK instead of K
        self.cxTL_dT_trg *= 1000
        
        
        # # Errors in fit
        # self.cxTR_fit_rmse = np.sqrt(np.mean(np.power(gen.expFunction(self.time_trgRange,*self.cxTR_fitcoeff)-self.cxTR_T_trgRange,2)))
        # self.cxTL_fit_rmse = np.sqrt(np.mean(np.power(gen.expFunction(self.time_trgRange,*self.cxTL_fitcoeff)-self.cxTL_T_trgRange,2)))
        
        
        print('dT_OL = ', (self.cxTR_dT_trg+self.cxTL_dT_trg)/2, 'mK')
        
        fig = plt.figure()
        plt.title('Top Right, dT = {} mK, tau = {} s \n{}'.format(self.cxTR_dT_trg, self.TR_tau, self.file))
        plt.plot(self.time, self.cxTR_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxTR_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[self.trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxTR_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxTR_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxTR_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:self.trgStart], np.full(len(self.time[:self.trgStart]), self.cxTR_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxTR_Tinit))
        
        plt.scatter(self.time[self.trgStart], self.cxTR_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[self.trgEnd], self.cxTR_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxTR_tauIndex], self.cxTR_T[self.cxTR_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxTR_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.title('Top Left, dT = {} mK, tau = {} s \n{}'.format(self.cxTL_dT_trg, self.TL_tau, self.file))
        plt.plot(self.time, self.cxTL_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxTL_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[self.trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxTL_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxTL_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxTL_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:self.trgStart], np.full(len(self.time[:self.trgStart]), self.cxTL_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxTL_Tinit))
        
        plt.scatter(self.time[self.trgStart], self.cxTL_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[self.trgEnd], self.cxTL_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxTL_tauIndex], self.cxTL_T[self.cxTL_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxTL_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        
        #innerlayer (bottom)
        
        #initial values
        self.cxBR_Tinit =  np.mean(sensorCalib.BR(self.cxBR_R[:self.trgStart]))
        self.cxBR_Tinit_STD =  np.std(sensorCalib.BR(self.cxBR_R[:self.trgStart]))
        self.cxBR_Tinit_SEM = self.cxBR_Tinit_STD / np.sqrt(len(self.cxBR_R[:self.trgStart]))
        
        self.cxBL_Tinit =  np.mean(sensorCalib.BL(self.cxBL_R[:self.trgStart]))
        self.cxBL_Tinit_STD =  np.std(sensorCalib.BL(self.cxBL_R[:self.trgStart]))
        self.cxBL_Tinit_SEM = self.cxBL_Tinit_STD / np.sqrt(len(self.cxBL_R[:self.trgStart]))
        
        
        #Area to be used when fitting exponential curve: 
        self.cxBR_T_trgRange = sensorCalib.BR(self.cxBR_R[self.trgStart:self.trgEnd])
        self.cxBL_T_trgRange = sensorCalib.BL(self.cxBL_R[self.trgStart:self.trgEnd])
        
        #Steady state calculated from trigger range
        self.cxBR_Tss_trg = np.mean(self.cxBR_T_trgRange[-int(self.len_trgRange*ssArea):-1]) 
        self.cxBR_Tss_trg_STD = np.std(self.cxBR_T_trgRange[-int(self.len_trgRange*ssArea):-1])
        self.cxBR_Tss_trg_SEM = self.cxBR_Tss_trg_STD / np.sqrt(len(self.cxBR_T_trgRange[-int(self.len_trgRange*ssArea):-1]))
        
        self.cxBL_Tss_trg = np.mean(self.cxBL_T_trgRange[-int(self.len_trgRange*ssArea):-1]) 
        self.cxBL_Tss_trg_STD = np.std(self.cxBL_T_trgRange[-int(self.len_trgRange*ssArea):-1])
        self.cxBL_Tss_trg_SEM = self.cxBL_Tss_trg_STD / np.sqrt(len(self.cxBL_T_trgRange[-int(self.len_trgRange*ssArea):-1]))
        
        #A initial guess of the dT for each sensor when triggering
        self.cxBR_dT_trg = float(self.cxBR_Tss_trg - self.cxBR_Tinit)
        self.cxBL_dT_trg = float(self.cxBL_Tss_trg - self.cxBL_Tinit)
        
         #Find time constant tau: 
        self.cxBR_tempTau = self.cxBR_dT_trg * (1 - 1/np.e) + self.cxBR_Tinit
        self.cxBR_tauIndex = np.where(self.cxBR_T == gen.find_nearest(self.cxBR_T[:int(self.len_trgRange/2)], self.cxBR_tempTau))[0][0]
        self.BR_tau = float(self.time[self.cxBR_tauIndex] - self.time[self.trgStart])
        
        self.cxBL_tempTau = self.cxBL_dT_trg * (1 - 1/np.e) + self.cxBL_Tinit
        self.cxBL_tauIndex = np.where(self.cxBL_T == gen.find_nearest(self.cxBL_T[:int(self.len_trgRange/2)], self.cxBL_tempTau))[0][0]
        self.BL_tau = float(self.time[self.cxBL_tauIndex] - self.time[self.trgStart])

  
        # #Exponentional curve fit of the rise in temperature
        # self.cxBR_fitcoeff, self.cxBR_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxBR_T_trgRange, p0 = [1, self.cxBR_dT_trg, self.cxBR_Tinit])
        # self.cxBL_fitcoeff, self.cxBL_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxBL_T_trgRange, p0 = [1, self.cxBL_dT_trg, self.cxBL_Tinit])
        
        # #Steady state values for the temperature when the coil is powered
        # self.cxBR_Tss_fit = self.cxBR_fitcoeff[1] + self.cxBR_fitcoeff[2]
        # self.cxBL_Tss_fit = self.cxBL_fitcoeff[1] + self.cxBL_fitcoeff[2]
        
        # #Tau values (i.e. time to reach 90?% of ss)
        # self.cxBR_tau = self.cxBR_fitcoeff[0]
        # self.cxBL_tau = self.cxBL_fitcoeff[0]
        
        # #Calculated dT values
        # self.cxBR_dT_fit =  float(self.cxBR_Tss_fit - self.cxBR_Tinit)*1000        
        # self.cxBL_dT_fit =  float(self.cxBL_Tss_fit - self.cxBL_Tinit)*1000
        
        self.cxBR_dT_trg *= 1000  #to get mK instead of K
        self.cxBL_dT_trg *= 1000
        
        print('dT_IL = ', (self.cxBR_dT_trg+self.cxBL_dT_trg)/2, 'mK')
        
        
        fig = plt.figure()
        plt.title('Bottom Right, dT = {} mK, tau = {} s \n{}'.format(self.cxBR_dT_trg, self.BR_tau, self.file))
        plt.plot(self.time, self.cxBR_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxBR_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[self.trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxBR_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxBR_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxBR_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:self.trgStart], np.full(len(self.time[:self.trgStart]), self.cxBR_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxBR_Tinit))
        
        plt.scatter(self.time[self.trgStart], self.cxBR_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[self.trgEnd], self.cxBR_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxBR_tauIndex], self.cxBR_T[self.cxBR_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxBR_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.title('Bottom Left, dT = {} mK, tau = {} s \n{}'.format(self.cxBL_dT_trg, self.BL_tau, self.file))
        plt.plot(self.time, self.cxBL_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxBL_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[self.trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxBL_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxBL_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxBL_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:self.trgStart], np.full(len(self.time[:self.trgStart]), self.cxBL_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxBL_Tinit))
        
        plt.scatter(self.time[self.trgStart], self.cxBL_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[self.trgEnd], self.cxBL_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxBL_tauIndex], self.cxBL_T[self.cxBL_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxBL_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        #Plot of exponentials
        
        # fig = plt.figure()
        # plt.title('Inner layer \n{}'.format(self.file))
        # plt.plot(self.time, self.cxBL_T, zorder=1, label = 'full measurement BL')
        # plt.plot(self.time, self.cxBR_T, zorder=1, label = 'full measurement BR')
        
        # plt.plot(self.time_trgRange + 19, gen.expFunction(self.time_trgRange,*self.cxBL_fitcoeff), label = 'fit BL')
        # plt.plot(self.time_trgRange + 19, gen.expFunction(self.time_trgRange,*self.cxBR_fitcoeff), label = 'fit BR')
        
        # plt.scatter(self.time[self.trgStart], self.cxBL_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        # plt.scatter(self.time[self.trgEnd], self.cxBL_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        # plt.plot(self.time, np.full(len(self.time), self.cxBL_Tinit), label = r'Tinit BL = {:.4f}'.format(self.cxBL_Tinit))
        # plt.plot(self.time, np.full(len(self.time), self.cxBR_Tinit), label = r'Tinit BR = {:.4f}'.format(self.cxBR_Tinit))
        
        # plt.grid()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # # pdf.savefig(fig, bbox_inches='tight')
        # plt.show()
        
        # fig = plt.figure()
        # plt.title('Outer layer \n{}'.format(self.file))
        # plt.plot(self.time, self.cxTL_T, zorder=1, label = 'full measurement TL')
        # plt.plot(self.time, self.cxTR_T, zorder=1, label = 'full measurement TR')
        
        # plt.plot(self.time_trgRange + 19, gen.expFunction(self.time_trgRange,*self.cxTL_fitcoeff), label = 'fit TL')
        # plt.plot(self.time_trgRange + 19, gen.expFunction(self.time_trgRange,*self.cxTR_fitcoeff), label = 'fit TR')
        
        # plt.scatter(self.time[self.trgStart], self.cxTL_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        # plt.scatter(self.time[self.trgEnd], self.cxTL_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        
        # plt.plot(self.time, np.full(len(self.time), self.cxTL_Tinit), label = r'Tinit TL = {:.4f}'.format(self.cxTL_Tinit))
        # plt.plot(self.time, np.full(len(self.time), self.cxTR_Tinit), label = r'Tinit TR = {:.4f}'.format(self.cxTR_Tinit))
        
        # plt.grid()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # # pdf.savefig(fig, bbox_inches='tight')
        # plt.show()
    
        
        
    def getDict(self): 
        auxDict = {}
        auxDict.update({'file': self.file}) 
        auxDict.update({'info': self.info})
        auxDict.update({'filePressure': self.filePressure})
        auxDict.update({'fileVoltage': self.fileVoltage})
        auxDict.update({'fileFreq': self.fileFreq})
        
        auxDict.update({'PuC_Amplitude': self.amplitude})
        auxDict.update({'PuC_Amplitude_STD': self.amplitudeSTD})
        auxDict.update({'PuC_Amplitude_SEM': self.amplitudeSEM})
        auxDict.update({'PuC_Frequency': self.frequency})
        
        auxDict.update({'MKSpressure': self.MKSpressure})
        
        auxDict.update({'MKSinit': self.MKS_init})
        auxDict.update({'MKSinitSTD': self.MKS_init_STD})
        auxDict.update({'MKSinitSEM': self.MKS_init_SEM})
        
        auxDict.update({'MKSstable': self.MKS_stabilized})
        auxDict.update({'MKSstableSTD': self.MKS_stabilized_STD})
        auxDict.update({'MKSstableSEM': self.MKS_stabilized_SEM})
        
        auxDict.update({'MKS_dT': self.MKS_dT}) #K
        
        
        auxDict.update({'cxPot_Tinit': self.cxPot_Tinit}) #K
        auxDict.update({'cxPot_Tinit_STD': self.cxPot_Tinit_STD}) #K
        auxDict.update({'cxPot_Tinit_SEM': self.cxPot_Tinit_SEM}) #K
         
        auxDict.update({'cxPot_Tss_trg': self.cxPot_Tss_trg}) #K
        auxDict.update({'cxPot_Tss_trg_STD': self.cxPot_Tss_trg_STD}) #K
        auxDict.update({'cxPot_Tss_trg_SEM': self.cxPot_Tss_trg_SEM}) #K
        
        auxDict.update({'cxPot_Tss_fit': self.cxPot_Tss_fit}) #K
        auxDict.update({'cxPot_fit_RMSE': self.cxPot_fit_rmse}) #K
        
        auxDict.update({'cxPot_dT_trg': self.cxPot_dT_trg}) #mK
        auxDict.update({'cxPot_dT_fit': self.cxPot_dT_fit}) #mK
        
        auxDict.update({'cxPot_MKS_dT_trg': self.cxPot_dT_trg-self.MKS_dT}) #mK
        auxDict.update({'cxPot_MKS_dT_fit': self.cxPot_dT_fit-self.MKS_dT}) #mK
        
        
        auxDict.update({'cxPot_trg_SS': self.cxPot_SS}) #bool If trg and fit are close... 
        auxDict.update({'cxPot_trg_uncertain': self.cxPot_uncertain}) #bool If pot_trg is in the standard deviation of the MKS... 
        
        auxDict.update({'HeatInput_trg': self.heat_trg}) #mW
        auxDict.update({'HeatInput_fit': self.heat_fit}) #mW
        
        auxDict.update({'HeatInput_MKS_trg': self.heat_MKS_trg}) #mW
        auxDict.update({'HeatInput_MKS_fit': self.heat_MKS_fit}) #mW
    
        
        auxDict.update({'dT_il': (self.cxBR_dT_trg+self.cxBL_dT_trg)/2})#mK
        auxDict.update({'dT_ol': (self.cxTR_dT_trg+self.cxTL_dT_trg)/2}) #mK
        
        auxDict.update({'dT_BR': self.cxBR_dT_trg}) #mK
        auxDict.update({'dT_BL': self.cxBL_dT_trg}) #mK
        auxDict.update({'dT_TR': self.cxTR_dT_trg}) #mK
        auxDict.update({'dT_TL': self.cxTL_dT_trg}) #mK
        
        auxDict.update({'tau_il': (self.BR_tau+self.BL_tau)/2}) 
        auxDict.update({'tau_ol': (self.TR_tau+self.TL_tau)/2}) 
        
        auxDict.update({'tau_BR': self.BR_tau}) 
        auxDict.update({'tau_TL': self.TL_tau}) 
        auxDict.update({'tau_BL': self.BL_tau}) 
        auxDict.update({'tau_TR': self.TR_tau}) 
        
    
        return auxDict


def loopCoilHeatingData():
    # analyse the raw data to a new file
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements'
    allFilesInFolder = gen.getCSV(folder)
    unsortedFiles = filter(lambda x: x.startswith('CoilPVT_'), allFilesInFolder)
    
    CoilPVTFiles = sorted(unsortedFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
    rowList = []
    
    for fileName in CoilPVTFiles: 
        file = CoilHeatingData(folder, fileName)
        rowList.append(file.getDict())
    
    
    unsortedCalibFiles = filter(lambda x: x.startswith('CoilPVTInv'), allFilesInFolder)
    calibFiles = sorted(unsortedCalibFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
    
    for fileName in calibFiles: 
        file = CoilHeatingData(folder, fileName)
        rowList.append(file.getDict())
        
    dataCalib = pd.DataFrame(rowList)    
    dataCalib.to_csv('CoilHeating.csv') 
    
    
    return

loopCoilHeatingData()




# folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements'
# File1 = CoilHeatingData(folder, r'CoilPVT_16_38mbar_RH0_0mA_SH0_0mA_Coil100_5V_50_0Hz_0_0Ohm__20200810_192828.csv')



# File2 = CoilHeatingData(folder, r'CoilPVT_22_99mbar_RH0_0mA_SH0_0mA_Coil50_8V_50_0Hz_0_0Ohm__20200805_223210.csv')

# pdf.close()