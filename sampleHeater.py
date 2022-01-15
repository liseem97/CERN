# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:45:52 2020

@author: lise
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# import matplotlib.cm as cm
import matplotlib.backends.backend_pdf

import general as gen
import sensorCalibration as sC

ITS90 = gen.ITS90LHe('SF')
MKSfit = gen.MKSfit()
cxCurrent = 3E-6

sensorCalib = sC.SensorCalibration()


pdf = matplotlib.backends.backend_pdf.PdfPages("SampleHeater.pdf")

class HeatInputAdjustments():
    def __init__(self): 
        ## Heater distribution calculations: 
        #Thermal conductivity of G10/epoxy. x corresponds to temperature. Works in temperature range 0-4K. 
        self.k_epoxy = lambda x: 0.0179*x - 0.0129 
        self.k_ULTEM = 0.002663 # Thermal conductivity at 4 K for ULTEM
        #D11T magnet sample information
        self.cableLength = 0.0147 # m 
        self.k_cables = 300 #Estimate from Kirtana (from where again?) high conductive oxygen free copper OFHC . 
        #Insulation in magnet
        #Drawing dimentions:
        self.ins_innerLayer = 250E-6 #m
        self.ins_interLayer = 1000E-6 #m
        self.ins_outerLayer = 250E-6 #m (180 in drawing, but drawing without quench heater...)
        #Insulation above sample heater
        self.top_G10_length = 10E-3 #m 
        self.top_ULTEM_length = 10E-3 #m 
        
        self.length_sample = self.ins_innerLayer + self.ins_interLayer + self.ins_outerLayer + 2* self.cableLength
        self.length_topInsulation = self.top_G10_length + self.top_ULTEM_length
        
        self.thermalResistance_sample = (self.ins_innerLayer + self.ins_interLayer + self.ins_outerLayer)/self.k_epoxy(4) + (2*self.cableLength)/self.k_cables
        self.thermalResistance_topIns = self.top_G10_length/self.k_epoxy(4) + self.top_ULTEM_length/self.k_ULTEM 
        
        self.thermalConductivity_sample = self.length_sample/self.thermalResistance_sample
        self.thermalConductivity_topIns = self.length_topInsulation/self.thermalResistance_topIns
        
        self.ratio_sample_topIns = (self.thermalConductivity_sample * self.length_topInsulation) / (self.thermalConductivity_topIns * self.length_sample)
        
        self.heat_in_sample = 1/(1 + 1/self.ratio_sample_topIns)
        self.heat_in_topIns = 1/(1 + self.ratio_sample_topIns)
        
        # print(self.heat_in_sample)
        
        ## Heat in wires after readout point to insulation calculation: 
        #wire lenght after readout point, before insulation
        self.length_readOut_to_ins = 10E-3 #m 
        
        #from wire information/lakeshore
        self.maganin_RperL = 4.2 #Ohm/m 
        self.dualTwist_RperL = 8.56 #Ohm/m 
        
        #Resistance from readoutpoint to liquid helium bath: 
        self.R_readOut_to_LHe = 2*(self.maganin_RperL * self.length_readOut_to_ins ) + 2*(self.dualTwist_RperL * self.length_readOut_to_ins)
        
        ## Heat loss around cable feedthrough calculation: 
        self.side_G10_length = 5E-3 #m
        self.side_ULTEM_length = 5E-3 #m 
        
        self.dx = self.side_G10_length + self.side_ULTEM_length
        
        self.thermalConductivity_dualTwist = 1.6 #W/(m*K) at 4K 
        self.thermalConductivity_maganin = 0.5 #W/(M*K) at 4K 
        
        self.diameter_dualTwist = 0.127E-3
        self.diameter_maganin = 0.4E-3
        
        self.crossSection_dualTwist =  np.pi*np.power((self.diameter_dualTwist/2), 2)
        self.crossSection_maganin =  np.pi*np.power((self.diameter_maganin/2), 2)
        
        self.thermalResistance_dualTwist = self.dx / (self.thermalConductivity_dualTwist * self.crossSection_dualTwist) 
        self.thermalResistance_maganin = self.dx / (self.thermalConductivity_maganin * self.crossSection_maganin)
        
        ## Voltage card limit calculations: 
        #Readout resistance i.e. resistance where voltage wires are connected. 
        folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
        df = pd.read_csv('{}\{}'.format(folder, r'CircuitAnalysis.csv'))
        self.df = df
        self.R_readout = np.mean(df.loc[(df['cardLimitReached'] == False) & (df['errR_readout']<10), 'R_readout'])
        R_readout_std = np.std(df.loc[(df['cardLimitReached'] == False) & (df['errR_readout']<10), 'R_readout'])
        R_readout_SEM = R_readout_std/np.sqrt(len(df.loc[(df['cardLimitReached'] == False) & (df['errR_readout']<10), 'R_readout']))
        R_readoutError = np.mean(df.loc[(df['cardLimitReached'] == False) & (df['errR_readout']<10), 'errR_readout'])
        
        self.R_readout_error = R_readoutError + R_readout_SEM
        
        # V_circuit = df.loc[(df['cardLimitReached'] == False) & (df['HAMEG_Recorded'] == True), 'V_HAMEG'] -  df.loc[(df['cardLimitReached'] == False) & (df['HAMEG_Recorded'] == True), 'V_sampleHeater']      
       
        # self.R_circuit = np.mean(V_circuit/(df.loc[(df['cardLimitReached'] == False) & (df['HAMEG_Recorded'] == True), 'fileCurrent']/1000))
    
        # self.R_circuit_std = np.std(V_circuit/(df.loc[(df['cardLimitReached'] == False) & (df['HAMEG_Recorded'] == True), 'fileCurrent']/1000))
        
        
        
        # print('Readout resistance: ', self.R_readout)
        # print('Circuit resistance: ', self.R_circuit)
        pass
    def insulationHeatLoss(self, heat):         
        return heat * self.heat_in_sample
        
    def wireHeatAdjustment(self, heat, fileCurrent, dT_ol): 
        heat += self.R_readOut_to_LHe*np.power(fileCurrent/1000,2)*1000
        heat -= 2*((dT_ol/self.thermalResistance_dualTwist)+(dT_ol/self.thermalResistance_maganin))
        return heat
    
    def voltageCardLimit(self, fileCurrent, fileName):
        P = (np.power(fileCurrent/1000, 2)*self.R_readout)*1000 #mW
        errorI = fileCurrent/1000 * 0.2/100 + 3E-3
        errorP = P * (errorI/fileCurrent/1000 + self.R_readout/self.R_readout_error)
        # # Vr = P / fileCurrent
        # # print(Vr)
        # V_hameg = self.df.loc[self.df['file'] == fileName, 'V_HAMEG'].values[0]
        # V_hameg_error = V_hameg*0.2/100 + 0.01
        
        # V_readout = V_hameg - self.R_circuit * (fileCurrent/1000)
        # # print(V_readout)
        # P2 = V_readout * fileCurrent #mW
        # # print(P, P2)
        return P, errorP 

    
hia = HeatInputAdjustments()


class SampleHeaterData(): 
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        self.info = file.split('_')[0]
        
        print(self.file)
        
        #Extract information from the file name:
        self.filePressure = float(file.split('_')[1]+'.'+file.split('_')[2].strip('mbar')) 
        # self.fileTemp = ITS90.getTemp(self.filePressure)

        self.fileCurrent = float(file.split('_')[5].strip('SH')+'.'+file.split('_')[6].strip('mA'))
        
        df = gen.removeChannelName(pd.read_csv('{}/{}'.format(folder, file), header = [0,1]))   
        self.rawData = df
        self.timeFull = df['Time']
        
        #100:-100 to remove spikes in temperature measurements. 
        spikeStart = 100
        spikeEnd = -100
        
        #Time without spikes for plotting purposes
        self.time = self.timeFull[spikeStart:spikeEnd].to_numpy()
        
        #PickUp coil
        self.sampleHeater = df['sampleHeater'].to_numpy()
        
        #Choosing of trigger area/ Area where the coil is powered.
        noise = 1E-4 #Noise in pickup coil is less than this, but when we have a signal in the pickup coil its always more than this. 
        
        #for mesurements where i use the full range.
        self.trgStartFull = np.where(self.sampleHeater > noise)[0][0] - 1 # -1 to make up for that this is recorded with a slow card. 
        self.trgEndFull = np.where(self.sampleHeater > noise)[0][-1] 

        #for measurements where i have taken out spikes
        self.trgStart = self.trgStartFull - spikeStart
        self.trgEnd = self.trgEndFull - spikeStart
        
        #Analysing sampleheater and getting the heat from this. 
        self.SH_V = np.mean(self.sampleHeater[self.trgStartFull + 2*spikeStart:self.trgEndFull + 2*spikeEnd])
        SH_STD = np.std(self.sampleHeater[self.trgStartFull + 2*spikeStart:self.trgEndFull + 2*spikeEnd])
        self.SH_SEM = SH_STD/np.sqrt(len(self.sampleHeater[self.trgStartFull+spikeStart:self.trgEndFull+spikeEnd]))
        
        
        self.SH_P = self.SH_V * self.fileCurrent #Power in mW ([V] * [mA]) 
        
        #Error calculation of heat: 
        error_I = self.fileCurrent/1000 * 0.2/100 + 3E-3 #A  (I * 0.2 % + 3digit = accuracy from HAMEG manufacturer)
        error_V = self.SH_V*0.2/100 + 0.01 + self.SH_V*0.06/100 + 10.4*0.003/100 + self.SH_SEM#V (V*0.2 % + 1digit from HAMEG, V*0.06% + range*0.003% from NIDAQ card)
        
        relError_P = error_I/self.fileCurrent/1000 + error_V/self.SH_V
        error_P = relError_P * self.SH_P #mW since P is in mW
        
        # self.relErrorV = error_V /self.SH_V 
      
        self.heat = self.SH_P
        self.heat_error = error_P
        
        # self.R_readout = self.SH_V / (self.fileCurrent/1000)
        # self.error_R_readout = (error_I/(self.fileCurrent/1000) + self.error_V/self.SH_V)*self.R_readout
        
        #If the card is reading more than 10V the card reading capability is reached and the voltage is calculated from the hameg voltage.
        self.cardLimitReached = False
        if 10 < self.SH_V: 
            self.cardLimitReached = True
            self.heat, self.heat_error = hia.voltageCardLimit(self.fileCurrent, self.file)
    
        #Analysing the Absolute He bath temperature from the MKS signal. 
        self.MKSpos = self.rawData['MKSpos']*10 #in prosentage open
        MKSpressureArray = MKSfit.getPressure(self.rawData['MKS'])
        self.MKStempArray = ITS90.getTemp(MKSpressureArray)
            
        #Mean values for the whole range: 
        # self.MKSpressure = np.mean(MKSpressureArray)
        # self.MKSpressureSTD = np.std(MKSpressureArray)
        # self.MKSpressureSEM = self.MKSpressureSTD / np.sqrt(len(MKSpressureArray))
        
        self.MKStemp = np.mean(self.MKStempArray)
        self.MKStempSTD = np.std(self.MKStempArray)
        # self.MKStempSEM = self.MKStempSTD / np.sqrt(len(self.MKStempArray))
        
        butter = gen.ButterworthFilter(self.timeFull, order = 6)
        self.lengthMeasurement = np.array(self.timeFull)[-1]
        
        #Voltage values
        cxPot_V = butter.filterSignal(self.rawData['cxPot'].to_numpy())[spikeStart:spikeEnd]
        cxBR_V = butter.filterSignal(self.rawData['cxBR'].to_numpy())[spikeStart:spikeEnd]
        cxTL_V = butter.filterSignal(self.rawData['cxTL'].to_numpy())[spikeStart:spikeEnd]
        cxBL_V = butter.filterSignal(self.rawData['cxBL'].to_numpy())[spikeStart:spikeEnd]
        cxTR_V = butter.filterSignal(self.rawData['cxTR'].to_numpy())[spikeStart:spikeEnd]
        
        #Resistance values
        self.cxPot_R = cxPot_V / cxCurrent
        self.cxBR_R = cxBR_V / cxCurrent
        self.cxTL_R = cxTL_V / cxCurrent
        self.cxBL_R = cxBL_V / cxCurrent
        self.cxTR_R = cxTR_V / cxCurrent
        
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
        
        fig = plt.figure()
        plt.title('Sample heater, card_limit = {} \n{}'.format(self.cardLimitReached, self.file))
        plt.scatter(self.timeFull, self.sampleHeater)
        plt.plot(self.timeFull, np.full(len(self.timeFull), self.SH_V), label = 'V_SH = {:.2f} V'.format(self.SH_V))
        
        plt.scatter(self.timeFull[self.trgStartFull], self.sampleHeater[self.trgStartFull], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.timeFull[self.trgEndFull], self.sampleHeater[self.trgEndFull], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.grid()
        fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(2)
        fig.suptitle('MKS pressure and position \n{}'.format(self.file))
        ax[0].plot(self.timeFull, MKSpressureArray, label = 'MKS pressure')
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
        
        
        self.time_trgRange = self.time[self.trgStart:self.trgEnd] - self.time[self.trgStart]
        
        self.len_trgRange = len(self.time_trgRange)
        
        #Check of change in MKS
        self.MKS_init = np.mean(self.MKStempArray[:self.trgStartFull])
        self.MKS_init_STD = np.std(self.MKStempArray[:self.trgStartFull])
        # self.MKS_init_SEM = self.MKS_init_STD / np.sqrt(len(self.MKStempArray[:self.trgStartFull]))
        
        self.MKS_trgRange = self.MKStempArray[self.trgStartFull:self.trgEndFull]
        
        self.MKS_stabilized = np.mean(self.MKS_trgRange[-int(self.len_trgRange/2):]) 
        self.MKS_stabilized_STD = np.std(self.MKS_trgRange[-int(self.len_trgRange/2):])
        # self.MKS_stabilized_SEM = self.MKS_stabilized_STD / np.sqrt(len(self.MKS_trgRange[-int(self.len_trgRange/2):]))
        
        self.MKS_dT = float(self.MKS_stabilized - self.MKS_init)*1000
        
        
        fig = plt.figure()
        plt.title('MKS temperature, dT = {} mK \n{}'.format(self.MKS_dT, self.file))
        plt.plot(self.time, self.MKStempArray[spikeStart:spikeEnd], label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.MKS_stabilized), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange/2):]+self.time[self.trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange/2):]), self.MKS_stabilized), c = 'tab:orange', linewidth = 4, label = 'MKS_stabilized = {:.4f}'.format(self.MKS_stabilized))
    
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
        # self.cxPot_Tinit_SEM = self.cxPot_Tinit_STD / np.sqrt(len(self.cxPot_R[:self.trgStart]))
        
        
        self.cxPot_T_trgRange = sensorCalib.Pot(self.cxPot_R[self.trgStart:self.trgEnd])

        self.cxPot_Tss_trg = np.mean(self.cxPot_T_trgRange[-int(self.len_trgRange/10):]) #-1000 to choose area at the end of cxPot_T_trgRange
        self.cxPot_Tss_trg_STD = np.std(self.cxPot_T_trgRange[-int(self.len_trgRange/10):])
        # self.cxPot_Tss_trg_SEM = self.cxPot_Tss_trg_STD / np.sqrt(len(self.cxPot_T_trgRange[-int(self.len_trgRange/10):]))
        
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
        
        tol = np.sqrt(np.power(self.cxPot_Tinit_STD,2) + np.power(self.cxPot_Tss_trg_STD,2))*1000 # mK
        if np.abs(self.cxPot_dT_fit - self.cxPot_dT_trg) < tol: 
            self.cxPot_SS = True
        else: 
            self.cxPot_SS = False
            
        tol = np.sqrt(np.power(self.MKS_init_STD,2) + np.power(self.MKS_stabilized_STD,2))*1000 # mK
        if np.abs(self.cxPot_dT_trg - self.MKS_dT) < tol: 
            self.cxPot_uncertain = True
        else: 
            self.cxPot_uncertain = False
            
        self.cxPot_dT_trg_MKS = self.cxPot_dT_trg - self.MKS_dT
     
        print('dT_pot_trg = ', self.cxPot_dT_trg, 'mK')
        print('dT_pot_fit = ', self.cxPot_dT_fit, 'mK')
        
        fig = plt.figure()
        
        plt.title('Cernox Pot Sensor, dT_trg = {} mK \n{}'.format(self.cxPot_dT_trg, self.file))
        plt.plot(self.time, self.cxPot_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange/10):]+self.time[self.trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange/10):]), self.cxPot_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss_trg = {:.4f}'.format(self.cxPot_Tss_trg))
    
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
        plt.plot(self.time_trgRange+self.time[self.trgStart], gen.expFunction(self.time_trgRange, *self.cxPot_fitcoeff), c = 'tab:cyan', label = 'exponential fit')

        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tss_trg), linestyle='dashed', c = 'tab:orange',label = 'T_ss_trg = {:.4f}'.format(self.cxPot_Tss_trg))
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tss_fit), linestyle='dashed', c = 'tab:pink', label = 'T_ss_fit = {:.4f}'.format(self.cxPot_Tss_fit))
        plt.plot(self.time, np.full(len(self.time), self.cxPot_Tinit), linestyle='dashed', c = 'tab:green', label = r'T_init = {:.4f}'.format(self.cxPot_Tinit))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        #outerlayer (top)
        
        #Adjust the triggers to they are as correct as possible for each measurement. (bit of a long code check if it can be changed to something faster)
        TR_trgStart = self.trgStart
        TR_trgEnd = self.trgEnd
        
        TL_trgStart = self.trgStart
        TL_trgEnd = self.trgEnd
        
        tol = 0.5 #mK
        while float(self.cxTR_T[TR_trgStart] - np.mean(self.cxTR_T[:TR_trgStart]))*1000 > tol: 
            TR_trgStart -= 1
            TR_trgEnd -= 1
        while float(self.cxTL_T[TL_trgStart] - np.mean(self.cxTL_T[:TL_trgStart]))*1000 > tol: 
            TL_trgStart -= 1
            TL_trgEnd -= 1
            
        
        #initial values
        self.cxTR_Tinit =  np.mean(sensorCalib.TR(self.cxTR_R[:TR_trgStart]))
        self.cxTR_Tinit_STD =  np.std(sensorCalib.TR(self.cxTR_R[:TR_trgStart]))
        self.cxTR_Tinit_SEM = self.cxTR_Tinit_STD / np.sqrt(len(self.cxTR_R[:TR_trgStart]))
        
        self.cxTL_Tinit =  np.mean(sensorCalib.TL(self.cxTL_R[:TL_trgStart]))
        self.cxTL_Tinit_STD =  np.std(sensorCalib.TL(self.cxTL_R[:TL_trgStart]))
        self.cxTL_Tinit_SEM = self.cxTL_Tinit_STD / np.sqrt(len(self.cxTL_R[:TL_trgStart]))
        
        #Area to be used when fitting exponential curve: 
        self.cxTR_T_trgRange = sensorCalib.TR(self.cxTR_R[TR_trgStart:TR_trgEnd])
        self.cxTL_T_trgRange = sensorCalib.TL(self.cxTL_R[TL_trgStart:TL_trgEnd])
        
        #Steady state calculated from trigger range
        #90% of the range is the steady state value (90% chosen by looking at the graphs)
        ssArea = 0.9
        self.cxTR_Tss_trg = np.mean(self.cxTR_T_trgRange[-int(self.len_trgRange*ssArea):-1]) 
        self.cxTR_Tss_trg_STD = np.std(self.cxTR_T_trgRange[-int(self.len_trgRange*ssArea):-1])
        self.cxTR_Tss_trg_SEM = self.cxTR_Tss_trg_STD / np.sqrt(len(self.cxTR_T_trgRange[-int(self.len_trgRange*ssArea):-1]))
        
        self.cxTL_Tss_trg = np.mean(self.cxTL_T_trgRange[-int(self.len_trgRange*ssArea):-1]) 
        self.cxTL_Tss_trg_STD = np.std(self.cxTL_T_trgRange[-int(self.len_trgRange*ssArea):-1])
        self.cxTL_Tss_trg_SEM = self.cxTL_Tss_trg_STD / np.sqrt(len(self.cxTL_T_trgRange[-int(self.len_trgRange*ssArea):-1]))
        
        #A initial guess of the dT for each sensor when triggering
        self.cxTR_dT_trg = float(self.cxTR_Tss_trg - self.cxTR_Tinit)
        self.cxTL_dT_trg = float(self.cxTL_Tss_trg - self.cxTL_Tinit)
        
        #Find time constant tau: 
        self.cxTR_tempTau = self.cxTR_dT_trg * (1 - 1/np.e) + self.cxTR_Tinit
        self.cxTR_tauIndex = np.where(self.cxTR_T == gen.find_nearest(self.cxTR_T[:int(self.len_trgRange/2)], self.cxTR_tempTau))[0][0]
        self.TR_tau = float(self.time[self.cxTR_tauIndex] - self.time[TR_trgStart])
        # print('tau TR new: ', self.TR_tau)
        
        self.cxTL_tempTau = self.cxTL_dT_trg * (1 - 1/np.e) + self.cxTL_Tinit
        self.cxTL_tauIndex = np.where(self.cxTL_T == gen.find_nearest(self.cxTL_T[:int(self.len_trgRange/2)], self.cxTL_tempTau))[0][0]
        self.TL_tau = float(self.time[self.cxTL_tauIndex] - self.time[TL_trgStart])
        # print('tau TL new: ', self.TL_tau)
        
  
        # #Exponentional curve fit of the rise in temperature
        # self.cxTR_fitcoeff, self.cxTR_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxTR_T_trgRange, p0 = [1, self.cxTR_dT_trg, self.cxTR_Tinit])
        # self.cxTL_fitcoeff, self.cxTL_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxTL_T_trgRange, p0 = [1, self.cxTL_dT_trg, self.cxTL_Tinit])
        
        # #Steady state values for the temperature when the coil is powered
        # self.cxTR_Tss_fit = self.cxTR_fitcoeff[1] + self.cxTR_fitcoeff[2]
        # self.cxTL_Tss_fit = self.cxTL_fitcoeff[1] + self.cxTL_fitcoeff[2]
        
        
        # #Tau values 
        # self.cxTR_tau = self.cxTR_fitcoeff[0]
        # self.cxTL_tau = self.cxTL_fitcoeff[0]
        
        # #Calculated dT values
        # self.cxTR_dT_fit =  float(self.cxTR_Tss_fit - self.cxTR_Tinit)*1000        
        # self.cxTL_dT_fit =  float(self.cxTL_Tss_fit - self.cxTL_Tinit)*1000

        self.cxTR_dT_trg *= 1000
        self.cxTL_dT_trg *= 1000
        
        print('dT_OL = ', (self.cxTR_dT_trg+self.cxTL_dT_trg)/2, 'mK')
        print('tau_OL = ', (self.TR_tau+self.TL_tau)/2, 'mK')

        
        fig = plt.figure()
        plt.title('Top Right, dT = {} mK, tau = {} s \n{}'.format(self.cxTR_dT_trg, self.TR_tau, self.file))
        plt.plot(self.time, self.cxTR_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxTR_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[TR_trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxTR_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxTR_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxTR_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:TR_trgStart], np.full(len(self.time[:TR_trgStart]), self.cxTR_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxTR_Tinit))
        
        plt.scatter(self.time[TR_trgStart], self.cxTR_T[TR_trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[TR_trgEnd], self.cxTR_T[TR_trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxTR_tauIndex], self.cxTR_T[self.cxTR_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxTR_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.title('Top Left, dT = {} mK, tau = {} s \n{}'.format(self.cxTL_dT_trg, self.TL_tau, self.file))
        plt.plot(self.time, self.cxTL_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxTL_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[TL_trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxTL_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxTL_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxTL_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:TL_trgStart], np.full(len(self.time[:TL_trgStart]), self.cxTL_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxTL_Tinit))
        
        plt.scatter(self.time[TL_trgStart], self.cxTL_T[TL_trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[TL_trgEnd], self.cxTL_T[TL_trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxTL_tauIndex], self.cxTL_T[self.cxTL_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxTL_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        
        #innerlayer (bottom)
        
        BR_trgStart = self.trgStart
        BR_trgEnd = self.trgEnd
        
        BL_trgStart = self.trgStart
        BL_trgEnd = self.trgEnd
        
        tol = 0.5 #mK
        while float(self.cxBR_T[BR_trgStart] - np.mean(self.cxBR_T[:BR_trgStart]))*1000 > tol: 
            BR_trgStart -= 1
            BR_trgEnd -= 1
        while float(self.cxBL_T[BL_trgStart] - np.mean(self.cxBL_T[:BL_trgStart]))*1000 > tol: 
            BL_trgStart -= 1
            BL_trgEnd -= 1
        
        #initial values
        self.cxBR_Tinit =  np.mean(sensorCalib.BR(self.cxBR_R[:BR_trgStart]))
        self.cxBR_Tinit_STD =  np.std(sensorCalib.BR(self.cxBR_R[:BR_trgStart]))
        self.cxBR_Tinit_SEM = self.cxBR_Tinit_STD / np.sqrt(len(self.cxBR_R[:BR_trgStart]))
        
        self.cxBL_Tinit =  np.mean(sensorCalib.BL(self.cxBL_R[:BL_trgStart]))
        self.cxBL_Tinit_STD =  np.std(sensorCalib.BL(self.cxBL_R[:BL_trgStart]))
        self.cxBL_Tinit_SEM = self.cxBL_Tinit_STD / np.sqrt(len(self.cxBL_R[:BL_trgStart]))
        
        
        #Area to be used when fitting exponential curve: 
        self.cxBR_T_trgRange = sensorCalib.BR(self.cxBR_R[BR_trgStart:BR_trgEnd])
        self.cxBL_T_trgRange = sensorCalib.BL(self.cxBL_R[BL_trgStart:BL_trgEnd])
        
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
        self.BR_tau = float(self.time[self.cxBR_tauIndex] - self.time[BR_trgStart])
        # print('tau BR new: ', self.BR_tau)
        
        self.cxBL_tempTau = self.cxBL_dT_trg * (1 - 1/np.e) + self.cxBL_Tinit
        self.cxBL_tauIndex = np.where(self.cxBL_T == gen.find_nearest(self.cxBL_T[:int(self.len_trgRange/2)], self.cxBL_tempTau))[0][0]
        self.BL_tau = float(self.time[self.cxBL_tauIndex] - self.time[BL_trgStart])
        # print('tau BL new: ', self.BL_tau)
        
  
        # #Exponentional curve fit of the rise in temperature
        # self.cxBR_fitcoeff, self.cxBR_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxBR_T_trgRange, p0 = [1, self.cxBR_dT_trg, self.cxBR_Tinit])
        # self.cxBL_fitcoeff, self.cxBL_fitcov = curve_fit(gen.expFunction, self.time_trgRange, self.cxBL_T_trgRange, p0 = [1, self.cxBL_dT_trg, self.cxBL_Tinit])
        
        # #Steady state values for the temperature when the coil is powered
        # self.cxBR_Tss_fit = self.cxBR_fitcoeff[1] + self.cxBR_fitcoeff[2]
        # self.cxBL_Tss_fit = self.cxBL_fitcoeff[1] + self.cxBL_fitcoeff[2]
        
        # #Tau values (i.e. time to reach 90?% of ss)
        # self.cxBR_tau = self.cxBR_fitcoeff[0]
        # self.cxBL_tau = self.cxBL_fitcoeff[0]
        
        # # print('tau BR old', self.cxBR_tau)
        # # print('tau BL old', self.cxBL_tau)
        
        # #Calculated dT values
        # self.cxBR_dT_fit =  float(self.cxBR_Tss_fit - self.cxBR_Tinit)*1000        
        # self.cxBL_dT_fit =  float(self.cxBL_Tss_fit - self.cxBL_Tinit)*1000
        
        self.cxBR_dT_trg *= 1000
        self.cxBL_dT_trg *= 1000
        
        print('dT_IL = ', (self.cxBR_dT_trg+self.cxBL_dT_trg)/2, 'mK')
        print('tau_IL = ', (self.BR_tau+self.BL_tau)/2, 'mK')
        
        fig = plt.figure()
        plt.title('Bottom Right, dT = {} mK, tau = {} s \n{}'.format(self.cxBR_dT_trg, self.BR_tau, self.file))
        plt.plot(self.time, self.cxBR_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxBR_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[BR_trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxBR_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxBR_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxBR_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:BR_trgStart], np.full(len(self.time[:BR_trgStart]), self.cxBR_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxBR_Tinit))
        
        plt.scatter(self.time[BR_trgStart], self.cxBR_T[BR_trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[BR_trgEnd], self.cxBR_T[BR_trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxBR_tauIndex], self.cxBR_T[self.cxBR_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxBR_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.title('Bottom Left, dT = {} mK, tau = {} s \n{}'.format(self.cxBL_dT_trg, self.BL_tau, self.file))
        plt.plot(self.time, self.cxBL_T, label = 'measurement', c = 'tab:blue')
        
        plt.plot(self.time, np.full(len(self.time), self.cxBL_Tss_trg), linestyle='dashed', c = 'tab:orange')
        plt.plot(self.time_trgRange[-int(self.len_trgRange*ssArea):]+self.time[BL_trgStart], np.full(len(self.time_trgRange[-int(self.len_trgRange*ssArea):]), self.cxBL_Tss_trg), c = 'tab:orange', linewidth = 4, label = 'T_ss= {:.4f}'.format(self.cxBL_Tss_trg))
    
        plt.plot(self.time, np.full(len(self.time), self.cxBL_Tinit), linestyle='dashed', c = 'tab:green')
        plt.plot(self.time[:BL_trgStart], np.full(len(self.time[:BL_trgStart]), self.cxBL_Tinit), c = 'tab:green', linewidth = 4, label = r'T_init = {:.4f}'.format(self.cxBL_Tinit))
        
        plt.scatter(self.time[BL_trgStart], self.cxBL_T[BL_trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        plt.scatter(self.time[BL_trgEnd], self.cxBL_T[BL_trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        plt.scatter(self.time[self.cxBL_tauIndex], self.cxBL_T[self.cxBL_tauIndex], c = 'tab:purple', marker = "x", zorder=10, label = 'T_tau = {:.4f}'.format(self.cxBL_tempTau))
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        
        #Plots with exponential fits 
        # fig = plt.figure()
        # plt.title('Inner layer \n{}'.format(self.file))
        # plt.plot(self.time, self.cxBL_T, zorder=1, label = 'full measurement BL')
        # plt.plot(self.time, self.cxBR_T, zorder=1, label = 'full measurement BR')
        
        # plt.plot(self.time_trgRange + self.time[self.trgStart], gen.expFunction(self.time_trgRange,*self.cxBL_fitcoeff), label = 'fit BL')
        # plt.plot(self.time_trgRange + self.time[self.trgStart], gen.expFunction(self.time_trgRange,*self.cxBR_fitcoeff), label = 'fit BR')
        
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
        
        # plt.plot(self.time_trgRange + self.time[self.trgStart], gen.expFunction(self.time_trgRange,*self.cxTL_fitcoeff), label = 'fit TL')
        # plt.plot(self.time_trgRange + self.time[self.trgStart], gen.expFunction(self.time_trgRange,*self.cxTR_fitcoeff), label = 'fit TR')
        
        # plt.scatter(self.time[self.trgStart], self.cxTL_T[self.trgStart], c = 'tab:red', marker = "x", zorder=10, label = 'trgStart')
        # plt.scatter(self.time[self.trgEnd], self.cxTL_T[self.trgEnd], c = 'tab:red', marker = "x", zorder=10, label = 'trgEnd')
        
        
        # plt.plot(self.time, np.full(len(self.time), self.cxTL_Tinit), label = r'Tinit TL = {:.4f}'.format(self.cxTL_Tinit))
        # plt.plot(self.time, np.full(len(self.time), self.cxTR_Tinit), label = r'Tinit TR = {:.4f}'.format(self.cxTR_Tinit))
        
        # plt.grid()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # # pdf.savefig(fig, bbox_inches='tight')
        # plt.show()
        
        
        self.heat = hia.wireHeatAdjustment(self.heat, self.fileCurrent, ((self.cxTR_dT_trg+self.cxTL_dT_trg)/2)/1000)
        self.heat = hia.insulationHeatLoss(self.heat)
        
        
        print(self.heat, '+/-', self.heat_error, 'mW')
        

        
        
    def getDict(self): 
        auxDict = {}
        auxDict.update({'file': self.file}) 
        auxDict.update({'info': self.info})
        auxDict.update({'filePressure': self.filePressure})
        auxDict.update({'fileCurrent': self.fileCurrent})
        
        auxDict.update({'heatInput': self.heat})
        auxDict.update({'heatInput_error': self.heat})
        
        # auxDict.update({'MKSpressure': self.MKSpressure})
        # auxDict.update({'MKSpressureSTD': self.MKSpressureSTD}) 
        auxDict.update({'avgMKStemp': self.MKStemp})
        # auxDict.update({'MKStempSTD': self.stdMKStemp})
        auxDict.update({'MKStemp_stable': self.MKS_stabilized})
        auxDict.update({'MKS_dT': self.MKS_dT})
        
        # auxDict.update({'V_sampleHeater': self.SH_V})
        # auxDict.update({'errV_sampleHater': self.error_V})
        # auxDict.update({'cardLimitReached': self.cardLimitReached})
        # auxDict.update({'R_readout': self.R_readout})
        # auxDict.update({'errR_readout': self.error_R_readout})
   
        
        # auxDict.update({'SH_V_STD': self.SH_std}) #V
        
        auxDict.update({'Tpot_init': self.cxPot_Tinit}) #mK
        auxDict.update({'Tpot_ss': self.cxPot_Tss_trg}) #mK
        
        auxDict.update({'dTPot': self.cxPot_dT_trg}) #mK
        auxDict.update({'dTPot_fit': self.cxPot_dT_fit}) #mK
        auxDict.update({'cxPot_SS': self.cxPot_SS}) #bool
        
        auxDict.update({'dTPot_MKS': self.cxPot_dT_trg_MKS}) #mK
        auxDict.update({'cxPot_uncertain': self.cxPot_uncertain}) #bool
        
        # auxDict.update({'R_readout': self.R_readOut})
        # auxDict.update({'lengthMeasurement': self.lengthMeasurement}) #s
        # auxDict.update({'cxPot_tau': self.cxPot_tau}) #s 
        # auxDict.update({'cxPot_SteadyState': self.cxPot_SS}) #bool
    
        # auxDict.update({'cxPot_Tinit': self.cxPot_Tinit}) #K
        # auxDict.update({'cxPot_Tinit_STD': self.cxPot_Tinit_STD*1000}) #mK
        # auxDict.update({'cxPot_Tss_trg': self.cxPot_Tss_trg}) #K
        # auxDict.update({'cxPot_Tss_trg_STD': self.cxPot_Tss_trg_STD*1000}) #mK
        # auxDict.update({'cxPot_Tss_fit': self.cxPot_Tss_fit}) #K
        auxDict.update({'cxPot_fit_RMSE': self.cxPot_fit_rmse*1000}) #mK
        
        
        
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
        
        # auxDict.update({'tau_BR_old': self.cxBR_tau}) 
        # auxDict.update({'tau_TL_old': self.cxTL_tau}) 
        # auxDict.update({'tau_BL_old': self.cxBL_tau}) 
        # auxDict.update({'tau_TR_old': self.cxTR_tau})
    
        return auxDict




class RefHeaterData(): 
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        self.info = file.split('_')[0]
        
        self.filePressure = float(file.split('_')[1]+'.'+file.split('_')[2].strip('mbar')) 
        self.fileTemp = ITS90.getTemp(self.filePressure)
        
        self.fileCurrent = float(file.split('_')[3].strip('RH')+'.'+file.split('_')[4].strip('mA'))
        
        print('current', self.fileCurrent,'mA')
        
        df = gen.removeChannelName(pd.read_csv('{}/{}'.format(folder, file), header = [0,1]))   
        self.rawData = df
        self.time = df['Time']
        
        self.MKSpressureArray = MKSfit.getPressure(self.rawData['MKS'])
        self.MKStempArray = ITS90.getTempArray(self.MKSpressureArray)
            
        self.MKSpressure = np.mean(self.MKSpressureArray)
        self.stdMKSpressure = np.std(self.MKSpressureArray)
            
        self.MKStemp = np.mean(self.MKStempArray)
        self.stdMKStemp = np.std(self.MKSpressureArray)
        
        
        butter = gen.ButterworthFilter(self.time, 25, order=6) 
        
        self.refHeater = df['refHeater']
        

        
        self.trgStart = np.where(np.array(self.time, dtype=int) == 18)[0][0]
        self.trgEnd = np.where(np.array(self.refHeater)[int(len(self.refHeater)/2)::]  == np.min(self.refHeater))[0][0] + int(len(self.refHeater)/2)
 
        plt.title('Reference heater')
        plt.plot(self.time, self.refHeater)
        plt.scatter(self.time[self.trgStart], self.refHeater[self.trgStart], c ='r')
        plt.scatter(self.time[self.trgEnd], self.refHeater[self.trgEnd], c = 'r')

        plt.grid()
        plt.show()
        
        self.RH_V = np.mean(np.array(self.refHeater)[self.trgStart:self.trgEnd])
        self.RH_std = np.std(np.array(self.refHeater)[self.trgStart:self.trgEnd])
        print(self.RH_V, 'V')
        
        self.RH_P = self.RH_V * self.fileCurrent  #Power in mW ([V] * [mA])
        print(self.RH_P, 'mW')
        
        #Voltage values
        self.cxPot_V = butter.filterSignal(self.rawData['cxPot'].to_numpy())
        self.cxBR_V = butter.filterSignal(self.rawData['cxBR'].to_numpy())
        self.cxTL_V = butter.filterSignal(self.rawData['cxTL'].to_numpy())
        self.cxBL_V = butter.filterSignal(self.rawData['cxBL'].to_numpy())
        self.cxTR_V = butter.filterSignal(self.rawData['cxTR'].to_numpy())
        
        #Resistance values
        self.cxPot_R = self.cxPot_V / cxCurrent
        self.cxBR_R = self.cxBR_V / cxCurrent
        self.cxTL_R = self.cxTL_V / cxCurrent
        self.cxBL_R = self.cxBL_V / cxCurrent
        self.cxTR_R = self.cxTR_V / cxCurrent
        
        #Temperature values
        self.cxPot_T = sensorCalib.Pot(self.cxPot_R)
        
        self.cxBR_T = sensorCalib.BR(self.cxBR_R)
        self.cxTL_T = sensorCalib.TL(self.cxTL_R)
        self.cxBL_T = sensorCalib.BL(self.cxBL_R)
        self.cxTR_T = sensorCalib.TR(self.cxTR_R)
        
        plt.title('cxPot')
        plt.plot(self.time, self.cxPot_T, zorder=1)
        plt.scatter(self.time[self.trgStart], self.cxPot_T[self.trgStart], c = 'r', zorder=2)
        plt.scatter(self.time[self.trgEnd], self.cxPot_T[self.trgEnd], c = 'r', zorder=2)
        
        plt.grid()
        plt.show()
        
        #Pot sensor
        
        self.cxPot_Tinit = np.mean(np.array(self.cxPot_T)[500:self.trgStart]) 
        
        self.cxPot_fittime = self.time[self.trgStart:self.trgEnd]
        self.cxPot_fitT = self.cxPot_T[self.trgStart:self.trgEnd] 
        
        self.cxPot_dTinit = float(np.mean(self.cxPot_fitT[-1000:-1])-self.cxPot_Tinit) *1000
        
        self.cxPot_dTinit_test = float(np.mean(self.cxPot_fitT[-1000:-1])-self.fileTemp) *1000
        
        print('dT_pot = ', self.cxPot_dTinit, 'mK')
        print('dT_potTest = ', self.cxPot_dTinit_test, 'mK')
        
        
        
    def getDict(self): 
        auxDict = {}
        auxDict.update({'file': self.file}) 
        auxDict.update({'info': self.info})
        auxDict.update({'filePressure': self.filePressure})
        auxDict.update({'fileCurrent': self.fileCurrent})
        
        auxDict.update({'MKSpressure': self.MKSpressure})
        # auxDict.update({'MKSpressureSTD': self.MKSpressureSTD}) 
        auxDict.update({'MKStemp': self.MKStemp})
        # auxDict.update({'MKStempSTD': self.MKStempSTD})
        
        
        auxDict.update({'refHeater': self.RH_V})
        auxDict.update({'P': self.RH_P})
        # auxDict.update({'HeaterSTD': self.heaterVoltageSTD})
        
        
        auxDict.update({'dT_Pot': self.cxPot_dTinit}) #mK
        
        auxDict.update({'dT_test': self.cxPot_dTinit_test})
        return auxDict
       
        


def loopHeaterData():
    # analyse the raw data to a new file
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements'
    allFilesInFolder = gen.getCSV(folder)
    unsortedSHFiles = filter(lambda x: x.startswith('SampleHeater_'), allFilesInFolder)
    
    SHFiles = sorted(unsortedSHFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
    rowList = []
    
    for fileName in SHFiles: 
        file = SampleHeaterData(folder, fileName)
        # file.plots()
        
        rowList.append(file.getDict())
    
    
    dataCalib = pd.DataFrame(rowList)    
    dataCalib.to_csv('SampleHeater.csv')
    # dataCalib.to_csv('CircuitAnalysisNew.csv')
    
    # folder = r'C:\Users\lise\Documents\CERN\11T\Sample2\Measurements'
    # allFilesInFolder = f.getCSV(folder)
    # unsortedRHFiles = filter(lambda x: x.startswith('refHeater_'), allFilesInFolder)
    
    # RHFiles = sorted(unsortedRHFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
    # rowList = []
    
    # for fileName in RHFiles: 
    #     file = RefHeaterData(folder, fileName)
    #     # file.plots()
        
    #     rowList.append(file.getDict())
    
    
    # dataCalib = pd.DataFrame(rowList)    
    # dataCalib.to_csv('ReferenceHeater.csv')
    
    return

loopHeaterData()

# folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements'
# r = RefHeaterData(folder, r'refHeater_22_99mbar_RH100_0mA_SH0_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200805_181856.csv')

# s = SampleHeaterData(folder, r'SampleHeater_16_38mbar_RH0_0mA_SH90_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200810_164704.csv')
# s = SampleHeaterData(folder, r'SampleHeater_22_99mbar_RH0_0mA_SH15_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200804_220328.csv')

pdf.close()
# df =  pd.read_csv('{}\{}'.format(folder, r'SampleHeater.csv'))


# print(np.mean(df.loc[df['cardVoltageLimit'] == False, 'readout_R'])*1000)
# print(np.std(df.loc[df['cardVoltageLimit'] == False, 'readout_R'])*1000)