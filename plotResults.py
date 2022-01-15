# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:19:20 2020

@author: lmurberg
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sampleHeaterResults():
    # folderSH = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Results & Plots\Early results'
    # os.chdir(folderSH)
    # df_SH =  pd.read_csv('{}'.format(r'SampleHeater.csv'))
    
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
    os.chdir(folder)
    df_SH = pd.read_csv('{}'.format(r'SampleHeater.csv'))
    

    df_SH_19K = df_SH.groupby('filePressure').get_group(22.99)
    
    # folderRH = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Results & Plots\New folder'
    # os.chdir(folderRH)
    # df_RH =  pd.read_csv('{}'.format(r'ReferenceHeater_early.csv'))
    
    df_SH_21K = df_SH.groupby('filePressure').get_group(41.41)
    
    df_SH_20K = df_SH.groupby('filePressure').get_group(31.29)
    
    df_SH_18K = df_SH.groupby('filePressure').get_group(16.38)
    
    # plt.scatter(df_SH_19K['dT_test'], df_SH_19K['P'], label = 'Sample heater 1.9K')
    # plt.scatter(df_RH['dT_test'], df_RH['P'], label = 'Reference heater', zorder = 5)
    
    # # plt.scatter(df_SH_21K['dT_test'][:-1], df_SH_21K['P'][:-1], label = 'Sample heater 2.1K')
    
    
    
    # plt.xlabel('dT[mK]')
    # plt.ylabel('Q [mW]')
    # # plt.xlim([0,15])
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    
    # plt.scatter(df_SH_19K['dT_Pot'], df_SH_19K['P'], label = 'Sample heater 1.9K', c = df_SH_19K['fileCurrent'])
    # plt.scatter(df_RH['dT_test'], df_RH['P'], label = 'Reference heater', zorder = 5)
    
    # plt.scatter(df_SH_21K['dT_test'][:-1], df_SH_21K['P'][:-1], label = 'Sample heater 2.1K', c = df_SH_21K['fileCurrent'][:-1])
    
    # plt.scatter(df_SH_20K['dT_Pot'], df_SH_20K['P'], label = 'Sample heater 2.0K', c = df_SH_20K['fileCurrent'])
    plt.scatter(df_SH_18K.loc[df_SH_18K['cxPot_SS'] == True, 'dTPot'], df_SH_18K.loc[df_SH_18K['cxPot_SS'] == True, 'heatInput'], label = 'Sample heater 1.8K')#, c = df_SH_18K.loc[df_SH_18K['cxPot_SS'] == True, 'Unnamed: 0'])
    
    plt.scatter(df_SH_19K.loc[df_SH_19K['cxPot_SS'] == True, 'dTPot'], df_SH_19K.loc[df_SH_19K['cxPot_SS'] == True, 'heatInput'], label = 'Sample heater 1.9K')#, c = df_SH_19K.loc[df_SH_19K['cxPot_SS'] == True, 'fileCurrent'])

    plt.scatter(df_SH_20K.loc[df_SH_20K['cxPot_SS'] == True, 'dTPot'], df_SH_20K.loc[df_SH_20K['cxPot_SS'] == True, 'heatInput'], label = 'Sample heater 2.0K')#, c = df_SH_20K.loc[df_SH_20K['cxPot_SS'] == True, 'fileCurrent'])
    
    plt.scatter(df_SH_21K.loc[df_SH_21K['cxPot_SS'] == True, 'dTPot'], df_SH_21K.loc[df_SH_21K['cxPot_SS'] == True, 'heatInput'], label = 'Sample heater 2.1K')#, c = df_SH_21K.loc[df_SH_21K['cxPot_SS'] == True, 'fileCurrent'])
    
    # df_SH_18K.loc[df_SH_18K['cxPot_SS'] == False, 'dT_Pot']
    
    
    plt.xlabel('dT[mK]')
    plt.ylabel('Q [mW]')
    plt.xlim([-0.1,8])
    plt.ylim([-0.1,500])
    # plt.colorbar()
    plt.legend()
    plt.grid()
    plt.show()
                    
    
    plt.scatter(df_SH_18K['heatInput']/41, df_SH_18K['dT_ol']/1000 + 1.8, label = '1.8K, outer layer', marker = '^', c = 'y', zorder = 10)
    plt.scatter(df_SH_18K['heatInput']/41, df_SH_18K['dT_il']/1000 + 1.8, label = '1.8K, inner layer', marker = 'v', c = 'y', zorder = 10)
    
    
    plt.scatter(df_SH_19K['heatInput']/41, df_SH_19K['dT_ol']/1000 + 1.9, label = '1.9K, outer layer', marker = '^', c = 'r')
    plt.scatter(df_SH_19K['heatInput']/41, df_SH_19K['dT_il']/1000 + 1.9, label = '1.9K, inner layer', marker = 'v', c = 'r')
    
    # plt.scatter(df_SH_20K['heatInput']/41, df_SH_20K['dT_ol']/1000 + 2.0, label = '2.0K, outer layer', marker = '^', c = 'g')
    # plt.scatter(df_SH_20K['heatInput']/41, df_SH_20K['dT_il']/1000 + 2.0, label = '2.0K, inner layer', marker = 'v', c = 'g')

    
    # plt.scatter(df_SH_21K['heatInput']/41, df_SH_21K['dT_ol']/1000 + 2.1, label = '2.1K, outer layer', marker = '^', c = 'b')
    # plt.scatter(df_SH_21K['heatInput']/41, df_SH_21K['dT_il']/1000 + 2.1, label = '2.1K, inner layer', marker = 'v', c = 'b')
    # plt.scatter(df_SH_21K['dT_test'][:-1], df_SH_21K['P'][:-1], label = 'Sample heater 2.1K')
    
    
    plt.ylabel('Steady state temperatures [K]')
    plt.xlabel('Volumetric heat input [mW/ccm]')
    # plt.ylim([1.6, 8.6])
    # plt.xlim([-1, 40])
    
    plt.ylim([1.7, 3.2])
    plt.xlim([-0.1, 3.7])
    
    # plt.colorbar()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()     
    
    plt.scatter(df_SH_18K['heatInput']/41, df_SH_18K['dT_ol']/1000 + 1.8, label = '1.8K, outer layer', marker = '^', c = 'y', zorder = 10)
    plt.scatter(df_SH_19K['heatInput']/41, df_SH_19K['dT_ol']/1000 + 1.9, label = '1.9K, outer layer', marker = '^', c = 'r')
    plt.scatter(df_SH_20K['heatInput']/41, df_SH_20K['dT_ol']/1000 + 2.0, label = '2.0K, outer layer', marker = '^', c = 'g')
    plt.scatter(df_SH_21K['heatInput']/41, df_SH_21K['dT_ol']/1000 + 2.1, label = '2.1K, outer layer', marker = '^', c = 'b')
    
    
    plt.ylabel('Steady state temperatures [K]')
    plt.xlabel('Volumetric heat input [mW/ccm]')
    plt.ylim([1.7, 5])
    plt.xlim([-0.1, 6])
    
    # plt.colorbar()
    plt.legend()
    plt.grid()
    plt.show()    
    
    
    plt.scatter(df_SH_18K['heatInput']/41, df_SH_18K['dT_il']/1000 + 1.8, label = '1.8K, inner layer', marker = 'v', c = 'y', zorder = 10)
    plt.scatter(df_SH_19K['heatInput']/41, df_SH_19K['dT_il']/1000 + 1.9, label = '1.9K, inner layer', marker = 'v', c = 'r')
    plt.scatter(df_SH_20K['heatInput']/41, df_SH_20K['dT_il']/1000 + 2.0, label = '2.0K, inner layer', marker = 'v', c = 'g')
    plt.scatter(df_SH_21K['heatInput']/41, df_SH_21K['dT_il']/1000 + 2.1, label = '2.1K, inner layer', marker = 'v', c = 'b')
    
    
    plt.ylabel('Steady state temperatures [K]')
    plt.xlabel('Volumetric heat input [mW/ccm]')
    plt.ylim([1.7, 3.5])
    # plt.ylim([0, 2000])
    plt.xlim([-0.1, 6])
    
    # plt.colorbar()
    plt.legend()
    plt.grid()
    plt.show()    
    
    tau_il_18K = (df_SH_18K['tau_BR'] + df_SH_18K['tau_BL']) /2
    tau_ol_18K = (df_SH_18K['tau_TR'] + df_SH_18K['tau_TL']) /2
    
    tau_il_19K = (df_SH_19K['tau_BR'] + df_SH_19K['tau_BL']) /2
    tau_ol_19K = (df_SH_19K['tau_TR'] + df_SH_19K['tau_TL']) /2
    
    tau_il_21K = (df_SH_21K['tau_BR'] + df_SH_21K['tau_BL']) /2
    tau_ol_21K = (df_SH_21K['tau_TR'] + df_SH_21K['tau_TL']) /2
    
    tau_il_20K = (df_SH_20K['tau_BR'] + df_SH_20K['tau_BL']) /2
    tau_ol_20K = (df_SH_20K['tau_TR'] + df_SH_20K['tau_TL']) /2
    
    
    
    plt.scatter(df_SH_18K['dT_ol']/1000*(1 - 1/np.e) + 1.8,tau_ol_18K ,label = '1.8 K, outer layer', marker = '^', c = 'y')
    plt.scatter(df_SH_18K['dT_il']/1000*(1 - 1/np.e) + 1.8,tau_il_18K ,label = '1.8 K, inner layer', marker = 'v', c = 'y')

    
    plt.scatter(df_SH_19K['dT_ol']/1000*(1 - 1/np.e) + 1.9,tau_ol_19K ,label = '1.9K, outer layer', marker = '^', c = 'r')
    plt.scatter(df_SH_19K['dT_il']/1000*(1 - 1/np.e) + 1.9,tau_il_19K ,label = '1.9K, inner layer', marker = 'v', c = 'r')

    plt.scatter(df_SH_20K['dT_ol']/1000*(1 - 1/np.e) + 2.0,tau_ol_20K ,label = '2.0K, outer layer', marker = '^', c = 'g')
    plt.scatter(df_SH_20K['dT_il']/1000*(1 - 1/np.e) + 2.0,tau_il_20K ,label = '2.0K, inner layer', marker = 'v', c = 'g')
    
    
    plt.scatter(df_SH_21K['dT_ol']/1000*(1 - 1/np.e) + 2.1, tau_ol_21K, label = '2.1K, outer layer', marker = '^', c = 'b') #df_SH_21K['fileCurrent'][:-1]
    plt.scatter(df_SH_21K['dT_il']/1000*(1 - 1/np.e) + 2.1, tau_il_21K, label = '2.1K, inner layer', marker = 'v', c = 'b')

    
    # plt.xlim()
    plt.title('Sample heater / Joule heating')
    plt.xlabel('Cable temperatures at tau [K]')
    plt.ylabel('Transient behaviour/ tau [s]')
    plt.xlim([1.79,2.9])
    # plt.ylim([-0.1, 3.5])
    # plt.colorbar()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()       
    
    
    diff_18K = (df_SH_18K['dT_ol'] - df_SH_18K['dT_il'])
    diff_19K = (df_SH_19K['dT_ol'] - df_SH_19K['dT_il'])
    diff_20K = (df_SH_20K['dT_ol'] - df_SH_20K['dT_il'])
    diff_21K = (df_SH_21K['dT_ol'] - df_SH_21K['dT_il'])
    
    A = 11.67E-3 * 114.87E-3 #m^2 
    dx = 500E-6 #m
    
    # k_18K = ( df_SH_18K['P']/1000 * dx ) / ( A * diff_18K)
    # k_19K = ( df_SH_19K['P']/1000 * dx ) / ( A * diff_19K)
    # k_20K = ( df_SH_20K['P']/1000 * dx ) / ( A * diff_20K)
    # k_21K = ( df_SH_21K['P']/1000 * dx ) / ( A * diff_21K)
    
    
    
    # interLayer_temp_18K = ( (1.8 + df_SH_18K['dT_ol']/1000) + (1.8 + df_SH_18K['dT_il']/1000) )/2
    # interLayer_temp_19K = ( (1.9 + df_SH_19K['dT_ol']/1000) + (1.9 + df_SH_19K['dT_il']/1000) )/2
    # interLayer_temp_20K = ( (2.0 + df_SH_20K['dT_ol']/1000) + (2.0 + df_SH_20K['dT_il']/1000) )/2
    # interLayer_temp_21K = ( (2.1 + df_SH_21K['dT_ol']/1000) + (2.1 + df_SH_21K['dT_il']/1000) )/2
    
    
    # plt.scatter(interLayer_temp_18K, k_18K, label = '1.8K', c = 'y', zorder = 10)
    # plt.scatter(interLayer_temp_19K, k_19K, label = '1.9K', c = 'r')
    # plt.scatter(interLayer_temp_20K, k_20K, label = '2.0K', c = 'g')
    # plt.scatter(interLayer_temp_21K[:-1], k_21K[:-1], label = '2.1K', c = 'b')
    
    # plt.ylabel('k [W/ (m * K)]')
    # plt.xlabel('Inter layer temperature [K]')
    # # plt.ylim([0.015,0.035])
    # # plt.xlim([1.8,3])
    # plt.legend()
    # plt.grid()
    # plt.show()
    

def coilHeatingResults():
    
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
    os.chdir(folder)
    df = pd.read_csv('{}'.format(r'CoilHeating.csv'))
    
    df_18K = df.groupby('filePressure').get_group(16.38)
     
    df_19K = df.groupby('filePressure').get_group(22.99)
   
    df_20K = df.groupby('filePressure').get_group(31.29)
    
    df_21K = df.groupby('filePressure').get_group(41.41)

    
    # plt.title('PickUp Coil amplitude => dB/dt')
    # plt.scatter(df_18K.loc[(df_18K['SteadyState'] == True), 'HeatInput'], np.power(df_18K.loc[(df_18K['SteadyState'] == True), 'PuC_Amplitude'],2)/df_18K.loc[(df_18K['SteadyState'] == True), 'fileFreq'], label = '1.8 K, clear trend', marker = 'o', c = 'y')
    # # plt.scatter(df_18K.loc[(df_18K['Uncertain'] == True), 'HeatInput'], df_18K.loc[(df_18K['Uncertain'] == True), 'PuC_Amplitude'], label = '1.8 K, uncertain', marker = 'x', c = 'r')
    # # plt.scatter(df_18K.loc[(df_18K['Rising'] == True), 'HeatInput'], df_18K.loc[(df_18K['Rising'] == True), 'PuC_Amplitude'], label = '1.8 K, rising', marker = '^', c = 'r')
    # plt.scatter(df_18K['HeatImput'], np.power(df_18K['PuC_Amplitude'],3)/np.power(df_18K['PuC_Frequency'],2))
    
    # plt.scatter(df_19K.loc[(df_19K['SteadyState'] == True), 'HeatInput'], np.power(df_19K.loc[(df_19K['SteadyState'] == True), 'PuC_Amplitude'],2)/df_19K.loc[(df_19K['SteadyState'] == True), 'fileFreq'], label = '1.9 K, clear trend', marker = 'o', c = 'r')
    # # plt.scatter(df_19K.loc[(df_19K['Uncertain'] == True), 'HeatInput'], df_19K.loc[(df_19K['Uncertain'] == True), 'PuC_Amplitude']*1000/df_19K.loc[(df_19K['Uncertain'] == True), 'fileFreq'], label = '1.9 K, uncertain', marker = 'x', c = 'g')
    # # plt.scatter(df_19K.loc[(df_19K['Rising'] == True), 'HeatInput'], df_19K.loc[(df_19K['Rising'] == True), 'PuC_Amplitude']*1000/df_19K.loc[(df_19K['Rising'] == True), 'fileFreq'], label = '1.9 K, rising', marker = '^', c = 'g')
    
    
    # plt.scatter(df_20K.loc[(df_20K['SteadyState'] == True), 'HeatInput'], np.power(df_20K.loc[(df_20K['SteadyState'] == True), 'PuC_Amplitude'],2)/df_20K.loc[(df_20K['SteadyState'] == True), 'fileFreq'], label = '2.0 K, clear trend', marker = 'o', c = 'g')
    # # plt.scatter(df_20K.loc[(df_20K['Uncertain'] == True), 'HeatInput'], df_20K.loc[(df_20K['Uncertain'] == True), 'PuC_Amplitude'], label = '2.0 K, uncertain', marker = 'x', c = 'b')
    # # plt.scatter(df_20K.loc[(df_20K['Rising'] == True), 'HeatInput'], df_20K.loc[(df_20K['Rising'] == True), 'PuC_Amplitude'], label = '2.0 K, rising', marker = '^', c = 'b')
    
    
    # plt.scatter(df_21K.loc[(df_21K['SteadyState'] == True), 'HeatInput'], np.power(df_21K.loc[(df_21K['SteadyState'] == True), 'PuC_Amplitude'],2)/df_21K.loc[(df_21K['SteadyState'] == True), 'fileFreq'], label = '2.1 K, clear trend', marker = 'o', c = 'b')
    # # plt.scatter(df_21K.loc[(df_21K['Uncertain'] == True), 'HeatInput'], df_21K.loc[(df_21K['Uncertain'] == True), 'PuC_Amplitude'], label = '2.1 K, uncertain', marker = 'x', c = 'y')
    # # plt.scatter(df_21K.loc[(df_21K['Rising'] == True), 'HeatInput'], df_21K.loc[(df_21K['Rising'] == True), 'PuC_Amplitude'], label = '2.1 K, rising', marker = '^', c = 'y')
    
    # plt.ylabel('$ε_{peak}^2$ / f [$V^2$ / Hz]')
    # plt.xlabel('Heat input [mW]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.ylim([-0.0001,0.0101])
    # plt.show()  
    
    plt.scatter(df_18K['HeatInput'], np.power(df_18K['PuC_Amplitude'],3)/np.power(df_18K['PuC_Frequency'],2),label = '1.8 K')
    plt.scatter(df_19K['HeatInput'], np.power(df_19K['PuC_Amplitude'],3)/np.power(df_19K['PuC_Frequency'],2),label = '1.9 K')
    # plt.scatter(df_20K['HeatInput'], np.power(df_20K['PuC_Amplitude'],3)/np.power(df_20K['PuC_Frequency'],2),label = '2.0 K')
    plt.scatter(df_21K['HeatInput'], np.power(df_21K['PuC_Amplitude'],3)/np.power(df_21K['PuC_Frequency'],2),label = '2.1 K')
    plt.ylabel('$ε_{peak}^3$ / $f^2$')
    plt.xlabel('Heat input [mW]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.ylim([-0.00001,0.00015])
    plt.show()  
    
    # fit = np.polynomial.polynomial.Polynomial.fit(self.voltage, self.p_mbar, 1)
    
    
    
    
    plt.scatter(df_18K['HeatInput'], df_18K['PuC_Amplitude'],label = '1.8 K')
    plt.scatter(df_19K['HeatInput'], df_19K['PuC_Amplitude'],label = '1.9 K')
    # plt.scatter(df_20K['HeatInput'], df_20K['PuC_Amplitude'],label = '2.0 K')
    plt.scatter(df_21K['HeatInput'], df_21K['PuC_Amplitude'],label = '2.1 K')
    plt.ylabel('$ε_{peak}$')
    plt.xlabel('Heat input [mW]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    # plt.ylim([-0.0001,0.0101])
    plt.show()  
    
    
    
    
    
    # plt.scatter(df_18K['Q_trg'], np.power(df_18K['PuC_Amplitude']*1000,2)/df_18K['fileFreq'], label = '1.8 K', marker = 'o', c = 'y')
    # plt.scatter(df_19K['Q_trg'], np.power(df_19K['PuC_Amplitude']*1000,2)/df_19K['fileFreq'], label = '1.9 K', marker = 'o', c = 'r')
    # plt.scatter(df_20K['Q_trg'], np.power(df_20K['PuC_Amplitude']*1000,2)/df_20K['fileFreq'], label = '2.0 K', marker = 'o', c = 'g')
    # plt.scatter(df_21K['Q_trg'], np.power(df_21K['PuC_Amplitude']*1000,2)/df_21K['fileFreq'], label = '2.1 K', marker = 'o', c = 'b')
    # plt.ylabel('$ε_{peak}^2 / f$ [$mV^2 / Hz$]')
    # plt.xlabel('Heat input [mW]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show()  
    
    # plt.title('Volumetric heat input')
    # plt.scatter(df_18K['Q_trg']/41, df_18K['dT_ol']/1000 + 1.8, label = '1.8K, outer layer', marker = '^', c = 'y', zorder = 10)
    # plt.scatter(df_18K['Q_trg']/41, df_18K['dT_il']/1000 + 1.8, label = '1.8K, inner layer', marker = 'v', c = 'y', zorder = 10)
    
    
    # plt.scatter(df_19K['Q_trg']/41, df_19K['dT_ol']/1000 + 1.9, label = '1.9K, outer layer', marker = '^', c = 'r')
    # plt.scatter(df_19K['Q_trg']/41, df_19K['dT_il']/1000 + 1.9, label = '1.9K, inner layer', marker = 'v', c = 'r')
    
    # plt.scatter(df_20K['Q_trg']/41, df_20K['dT_ol']/1000 + 2.0, label = '2.0K, outer layer', marker = '^', c = 'g')
    # plt.scatter(df_20K['Q_trg']/41, df_20K['dT_il']/1000 + 2.0, label = '2.0K, inner layer', marker = 'v', c = 'g')
    
    
    # plt.scatter(df_21K['Q_trg']/41, df_21K['dT_ol']/1000 + 2.1, label = '2.1K, outer layer', marker = '^', c = 'b')
    # plt.scatter(df_21K['Q_trg']/41, df_21K['dT_il']/1000 + 2.1, label = '2.1K, inner layer', marker = 'v', c = 'b')
    
    
    # plt.ylabel('Steady state temperatures [K]')
    # plt.xlabel('Volumetric heat input [mW/ccm]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show()   
    
    
                    
    plt.title('Volumetric heat input')
    plt.scatter(df_18K['HeatInput']/41, df_18K['dT_ol']/1000 + 1.8, label = '1.8K, outer layer', marker = '^', c = 'y', zorder = 10)
    plt.scatter(df_18K['HeatInput']/41, df_18K['dT_il']/1000 + 1.8, label = '1.8K, inner layer', marker = 'v', c = 'y', zorder = 10)
    
    
    plt.scatter(df_19K['HeatInput']/41, df_19K['dT_ol']/1000 + 1.9, label = '1.9K, outer layer', marker = '^', c = 'r')
    plt.scatter(df_19K['HeatInput']/41, df_19K['dT_il']/1000 + 1.9, label = '1.9K, inner layer', marker = 'v', c = 'r')
    
    # plt.scatter(df_20K['HeatInput']/41, df_20K['dT_ol']/1000 + 2.0, label = '2.0K, outer layer', marker = '^', c = 'g')
    # plt.scatter(df_20K['HeatInput']/41, df_20K['dT_il']/1000 + 2.0, label = '2.0K, inner layer', marker = 'v', c = 'g')
    
    
    # plt.scatter(df_21K['HeatInput']/41, df_21K['dT_ol']/1000 + 2.1, label = '2.1K, outer layer', marker = '^', c = 'b')
    # plt.scatter(df_21K['HeatInput']/41, df_21K['dT_il']/1000 + 2.1, label = '2.1K, inner layer', marker = 'v', c = 'b')
    
    
    plt.ylabel('Steady state temperatures [K]')
    plt.xlabel('Volumetric heat input [mW/ccm]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()     
    
    plt.title('SteadyState Temperature vs PickUpCoil')
    plt.scatter(df_18K['PuC_Amplitude'], df_18K['dT_ol']/1000 + 1.8, label = '1.8K, outer layer', marker = '^', c = 'y')
    plt.scatter(df_18K['PuC_Amplitude'], df_18K['dT_il']/1000 + 1.8, label = '1.8K, inner layer', marker = 'v', c = 'y')
    
    
    plt.scatter(df_19K['PuC_Amplitude'], df_19K['dT_ol']/1000 + 1.9, label = '1.9K, outer layer', marker = '^', c = 'r')
    plt.scatter(df_19K['PuC_Amplitude'], df_19K['dT_il']/1000 + 1.9, label = '1.9K, inner layer', marker = 'v', c = 'r')
    
    plt.scatter(df_20K['PuC_Amplitude'], df_20K['dT_ol']/1000 + 2.0, label = '2.0K, outer layer', marker = '^', c = 'g')
    plt.scatter(df_20K['PuC_Amplitude'], df_20K['dT_il']/1000 + 2.0, label = '2.0K, inner layer', marker = 'v', c = 'g')
    
    
    plt.scatter(df_21K['PuC_Amplitude'], df_21K['dT_ol']/1000 + 2.1, label = '2.1K, outer layer', marker = '^', c = 'b')
    plt.scatter(df_21K['PuC_Amplitude'], df_21K['dT_il']/1000 + 2.1, label = '2.1K, inner layer', marker = 'v', c = 'b')
    
    
    plt.ylabel('Steady state temperatures [K]')
    plt.xlabel('$ε_{peak}$ [V]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()   
    
    plt.title('SteadyState Temperature vs PickUpCoil')
    # plt.scatter(np.power(df_18K['PuC_Amplitude'],2)*2*np.pi*df_18K['fileFreq'], df_18K['dT_ol']/1000 + 1.8, label = '1.8K, outer layer', marker = '^', c = 'y')
    # plt.scatter(np.power(df_18K['PuC_Amplitude'],2)*2*np.pi*df_18K['fileFreq'], df_18K['dT_il']/1000 + 1.8, label = '1.8K, inner layer', marker = 'v', c = 'y')
    
    #(2*np.pi*df_19K['fileFreq'])
    plt.scatter(np.power(df_19K['PuC_Amplitude'],3)/np.power(df_19K['fileFreq'],2), df_19K['dT_ol']/1000 + 1.9, label = '1.9K, outer layer', marker = '^', c = df_19K['fileVoltage'])
    plt.scatter(np.power(df_19K['PuC_Amplitude'],3)/np.power(df_19K['fileFreq'],2), df_19K['dT_il']/1000 + 1.9, label = '1.9K, inner layer', marker = 'v', c = df_19K['fileVoltage'])
    
    # plt.scatter(np.power(df_20K['PuC_Amplitude'],2)*2*np.pi*df_20K['fileFreq'], df_20K['dT_ol']/1000 + 2.0, label = '2.0K, outer layer', marker = '^', c = 'g')
    # plt.scatter(np.power(df_20K['PuC_Amplitude'],2)*2*np.pi*df_20K['fileFreq'], df_20K['dT_il']/1000 + 2.0, label = '2.0K, inner layer', marker = 'v', c = 'g')
    
    
    # plt.scatter(np.power(df_21K['PuC_Amplitude'],2)*2*np.pi*df_21K['fileFreq'], df_21K['dT_ol']/1000 + 2.1, label = '2.1K, outer layer', marker = '^', c = 'b')
    # plt.scatter(np.power(df_21K['PuC_Amplitude'],2)*2*np.pi*df_21K['fileFreq'], df_21K['dT_il']/1000 + 2.1, label = '2.1K, inner layer', marker = 'v', c = 'b')
    
    
    plt.ylabel('Steady state temperatures [K]')
    plt.xlabel('$ε_{peak}^3 / f^2$   [$V^3$ / $Hz^2$]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    cbar = plt.colorbar()
    # cbar.set_label('file frequency [Hz]')
    plt.xlim([-0.000001,0.00014])
    # plt.xlim([0.0,0.1])
    # plt.ylim([1.89,2.21])
    plt.show()   
    
    # plt.title('Heat Input')
    # plt.scatter(df_18K['dT_Pot_trg'], df_18K['HeatInput'], label = '1.8 K trigger', marker = 'o', c = 'y')
    # plt.scatter(df_19K['dT_Pot_trg'], df_19K['HeatInput'], label = '1.9 K trigger', marker = 'o', c = 'r')
    # plt.scatter(df_20K['dT_Pot_trg'], df_20K['HeatInput'], label = '2.0 K trigger', marker = 'o', c = 'g')
    # plt.scatter(df_21K['dT_Pot_trg'], df_21K['HeatInput'], label = '2.1 K trigger', marker = 'o', c = 'b')
    
    # plt.scatter(df_18K.loc[df_18K['dT_Pot_fit'] < 20,'dT_Pot_fit'], df_18K.loc[df_18K['dT_Pot_fit'] < 20,'HeatInput'], label = '1.8 K fit', marker = 'x', c = 'y')
    # plt.scatter(df_19K.loc[df_19K['dT_Pot_fit'] < 20,'dT_Pot_fit'], df_19K.loc[df_19K['dT_Pot_fit'] < 20,'HeatInput'], label = '1.9 K fit', marker = 'x', c = 'r')
    # plt.scatter(df_20K.loc[df_20K['dT_Pot_fit'] < 20,'dT_Pot_fit'], df_20K.loc[df_20K['dT_Pot_fit'] < 20,'HeatInput'], label = '2.0 K fit', marker = 'x', c = 'g')
    # plt.scatter(df_21K.loc[df_21K['dT_Pot_fit'] < 20,'dT_Pot_fit'], df_21K.loc[df_21K['dT_Pot_fit'] < 20,'HeatInput'], label = '2.1 K fit', marker = 'x', c = 'b')
    
    # plt.xlabel('dT_pot [mK]')
    # plt.ylabel('Volumetric heat input [mW]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show() 
    
    
    
    
    # plt.title('Heat Input for 1.8 K')
    # plt.scatter(df_18K.loc[(df_18K['SteadyState'] == True), 'HeatInput']/41, df_18K.loc[(df_18K['SteadyState'] == True), 'dT_ol']/1000 + 1.8, label = 'Clear trend', marker = 'o', c = 'y')
    # plt.scatter(df_18K.loc[(df_18K['SteadyState'] == True), 'HeatInput']/41, df_18K.loc[(df_18K['SteadyState'] == True), 'dT_il']/1000 + 1.8, marker = 'o', c = 'y')
    
    # # plt.scatter(df_18K.loc[(df_18K['Rising'] == True), 'HeatInput']/41, df_18K.loc[(df_18K['Rising'] == True), 'dT_ol']/1000 + 1.8, label = 'Rising', marker = '^', c = 'red')
    # # plt.scatter(df_18K.loc[(df_18K['Rising'] == True), 'HeatInput']/41, df_18K.loc[(df_18K['Rising'] == True), 'dT_il']/1000 + 1.8, marker = '^', c = 'red')
    
    # # plt.scatter(df_18K.loc[(df_18K['Uncertain'] == True), 'HeatInput']/41, df_18K.loc[(df_18K['Uncertain'] == True), 'dT_ol']/1000 + 1.8, label = 'Uncertain', marker = 'x', c = 'red')
    # # plt.scatter(df_18K.loc[(df_18K['Uncertain'] == True), 'HeatInput']/41, df_18K.loc[(df_18K['Uncertain'] == True), 'dT_il']/1000 + 1.8, marker = 'x', c = 'red')
    
    # plt.ylabel('Steady state temperatures [K]')
    # plt.xlabel('Volumetric heat input [mW/ccm]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show() 
    
    
    # plt.title('Heat Input for 1.9 K')
    
    # plt.scatter(df_19K.loc[(df_19K['SteadyState'] == True), 'HeatInput']/41, df_19K.loc[(df_19K['SteadyState'] == True), 'dT_ol']/1000 + 1.9, label = 'Clear treng', marker = 'o', c = 'r')
    # plt.scatter(df_19K.loc[(df_19K['SteadyState'] == True), 'HeatInput']/41, df_19K.loc[(df_19K['SteadyState'] == True), 'dT_il']/1000 + 1.9, marker = 'o', c = 'r')
   
    # plt.scatter(df_19K.loc[(df_19K['Rising'] == True), 'HeatInput']/41, df_19K.loc[(df_19K['Rising'] == True), 'dT_ol']/1000 + 1.9, label = 'Rising', marker = '^', c = 'g')
    # plt.scatter(df_19K.loc[(df_19K['Rising'] == True), 'HeatInput']/41, df_19K.loc[(df_19K['Rising'] == True), 'dT_il']/1000 + 1.9, marker = '^', c = 'g')
    
    # plt.scatter(df_19K.loc[(df_19K['Uncertain'] == True), 'HeatInput']/41, df_19K.loc[(df_19K['Uncertain'] == True), 'dT_ol']/1000 + 1.9, label = 'Uncertain', marker = 'x', c = 'g')
    # plt.scatter(df_19K.loc[(df_19K['Uncertain'] == True), 'HeatInput']/41, df_19K.loc[(df_19K['Uncertain'] == True), 'dT_il']/1000 + 1.9, marker = 'x', c = 'g')
    
   
    # plt.ylabel('Steady state temperatures [K]')
    # plt.xlabel('Volumetric heat input [mW/ccm]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show() 
    
    # plt.title('Heat Input for 2.0 K')
    
    # plt.scatter(df_20K.loc[(df_20K['SteadyState'] == True), 'HeatInput']/41, df_20K.loc[(df_20K['SteadyState'] == True), 'dT_ol']/1000 + 2.0, label = 'Clear trend', marker = 'o', c = 'g')
    # plt.scatter(df_20K.loc[(df_20K['SteadyState'] == True), 'HeatInput']/41, df_20K.loc[(df_20K['SteadyState'] == True), 'dT_il']/1000 + 2.0, marker = 'o', c = 'g')
    
   
    # plt.scatter(df_20K.loc[(df_20K['Rising'] == True), 'HeatInput']/41, df_20K.loc[(df_20K['Rising'] == True), 'dT_ol']/1000 + 2.0, label = 'Rising', marker = '^', c = 'b')
    # plt.scatter(df_20K.loc[(df_20K['Rising'] == True), 'HeatInput']/41, df_20K.loc[(df_20K['Rising'] == True), 'dT_il']/1000 + 2.0,  marker = '^', c = 'b')
    
    # plt.scatter(df_20K.loc[(df_20K['Uncertain'] == True), 'HeatInput']/41, df_20K.loc[(df_20K['Uncertain'] == True), 'dT_ol']/1000 + 2.0, label = 'Uncertain', marker = 'x', c = 'b')
    # plt.scatter(df_20K.loc[(df_20K['Uncertain'] == True), 'HeatInput']/41, df_20K.loc[(df_20K['Uncertain'] == True), 'dT_il']/1000 + 2.0, marker = 'x', c = 'b')
    

    # plt.ylabel('Steady state temperatures [K]')
    # plt.xlabel('Volumetric heat input [mW/ccm]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show() 
    
    # plt.title('Heat Input for 2.1 K')
    
    # plt.scatter(df_21K.loc[(df_21K['SteadyState'] == True), 'HeatInput']/41, df_21K.loc[(df_21K['SteadyState'] == True), 'dT_ol']/1000 + 2.1, label = 'Clear trend', marker = 'o', c = 'b')
    # plt.scatter(df_21K.loc[(df_21K['SteadyState'] == True), 'HeatInput']/41, df_21K.loc[(df_21K['SteadyState'] == True), 'dT_il']/1000 + 2.1,  marker = 'o', c = 'b')
    
   
    # plt.scatter(df_21K.loc[(df_21K['Rising'] == True), 'HeatInput']/41, df_21K.loc[(df_21K['Rising'] == True), 'dT_ol']/1000 + 2.1, label = 'Rising', marker = '^', c = 'y')
    # plt.scatter(df_21K.loc[(df_21K['Rising'] == True), 'HeatInput']/41, df_21K.loc[(df_21K['Rising'] == True), 'dT_il']/1000 + 2.1, marker = '^', c = 'y')
    
    # plt.scatter(df_21K.loc[(df_21K['Uncertain'] == True), 'HeatInput']/41, df_21K.loc[(df_21K['Uncertain'] == True), 'dT_ol']/1000 + 2.1, label = 'Uncertain', marker = 'x', c = 'y')
    # plt.scatter(df_21K.loc[(df_21K['Uncertain'] == True), 'HeatInput']/41, df_21K.loc[(df_21K['Uncertain'] == True), 'dT_il']/1000 + 2.1, marker = 'x', c = 'y')
    

    # plt.ylabel('Steady state temperatures [K]')
    # plt.xlabel('Volumetric heat input [mW/ccm]')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid()
    # plt.show() 
 
    
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
    os.chdir(folder)
    df = pd.read_csv('{}'.format(r'CoilHeating.csv'))
    
    df_18K = df.groupby('filePressure').get_group(16.38)
     
    df_19K = df.groupby('filePressure').get_group(22.99)
   
    df_20K = df.groupby('filePressure').get_group(31.29)
    
    df_21K = df.groupby('filePressure').get_group(41.41)
    
    
    tau_il_18K = (df_18K['tau_BR'] + df_18K['tau_BL']) /2
    tau_ol_18K = (df_18K['tau_TR'] + df_18K['tau_TL']) /2
    
    tau_il_19K = (df_19K['tau_BR'] + df_19K['tau_BL']) /2
    tau_ol_19K = (df_19K['tau_TR'] + df_19K['tau_TL']) /2
    
    tau_il_21K = (df_21K['tau_BR'] + df_21K['tau_BL']) /2
    tau_ol_21K = (df_21K['tau_TR'] + df_21K['tau_TL']) /2
    
    tau_il_20K = (df_20K['tau_BR'] + df_20K['tau_BL']) /2
    tau_ol_20K = (df_20K['tau_TR'] + df_20K['tau_TL']) /2
    
    
    
    plt.scatter(df_18K['dT_ol']/1000*0.63 + 1.8, tau_ol_18K ,label = '1.8K, outer layer', marker = '^', c = 'y')
    plt.scatter(df_18K['dT_il']/1000*0.63 + 1.8, tau_il_18K ,label = '1.8K, inner layer', marker = 'v', c = 'y')
    
    
    plt.scatter(df_19K['dT_ol']/1000*0.63 + 1.9,tau_ol_19K ,label = '1.9K, outer layer', marker = '^', c = 'r')
    plt.scatter(df_19K['dT_il']/1000*0.63 + 1.9,tau_il_19K ,label = '1.9K, inner layer', marker = 'v', c = 'r')
    
    plt.scatter(df_20K['dT_ol']/1000*0.63 + 2.0,tau_ol_20K ,label = '2.0K, outer layer', marker = '^', c = 'g')
    plt.scatter(df_20K['dT_il']/1000*0.63 + 2.0,tau_il_20K ,label = '2.0K, inner layer', marker = 'v', c = 'g')
    
    
    plt.scatter(df_21K['dT_ol']/1000*0.63 + 2.1, tau_ol_21K, label = '2.1K, outer layer', marker = '^', c = 'b') 
    plt.scatter(df_21K['dT_il']/1000*0.63 + 2.1, tau_il_21K, label = '2.1K, inner layer', marker = 'v', c = 'b')
    # plt.scatter(df_SH_21K['dT_test'][:-1], df_SH_21K['P'][:-1], label = 'Sample heater 2.1K')
    
    plt.title('Coil heating')
    plt.xlabel('Cable temperatures at tau [K]')
    plt.ylabel('Transient behaviour/ tau [s]')
    # plt.xlim([1.8,3])
    # plt.ylim([-0.1, 1.2])
    # plt.colorbar()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()       
    
    
    # diff_18K = (df_SH_18K['dT_ol'] - df_SH_18K['dT_il'])/1000
    # diff_19K = (df_SH_19K['dT_ol'] - df_SH_19K['dT_il'])/1000
    # diff_20K = (df_SH_20K['dT_ol'] - df_SH_20K['dT_il'])/1000
    # diff_21K = (df_SH_21K['dT_ol'] - df_SH_21K['dT_il'])/1000
    
    # A = 11.67E-3 * 114.87E-3 #m^2 
    # dx = 500E-6 #m
    
    # k_18K = ( df_SH_18K['P']/1000 * dx ) / ( A * diff_18K)
    # k_19K = ( df_SH_19K['P']/1000 * dx ) / ( A * diff_19K)
    # k_20K = ( df_SH_20K['P']/1000 * dx ) / ( A * diff_20K)
    # k_21K = ( df_SH_21K['P']/1000 * dx ) / ( A * diff_21K)
    
    
    
    # interLayer_temp_18K = ( (1.8 + df_SH_18K['dT_ol']/1000) + (1.8 + df_SH_18K['dT_il']/1000) )/2
    # interLayer_temp_19K = ( (1.9 + df_SH_19K['dT_ol']/1000) + (1.9 + df_SH_19K['dT_il']/1000) )/2
    # interLayer_temp_20K = ( (2.0 + df_SH_20K['dT_ol']/1000) + (2.0 + df_SH_20K['dT_il']/1000) )/2
    # interLayer_temp_21K = ( (2.1 + df_SH_21K['dT_ol']/1000) + (2.1 + df_SH_21K['dT_il']/1000) )/2
    
    
    # plt.scatter(interLayer_temp_18K, k_18K, label = '1.8K', c = 'y', zorder = 10)
    # plt.scatter(interLayer_temp_19K, k_19K, label = '1.9K', c = 'r')
    # plt.scatter(interLayer_temp_20K, k_20K, label = '2.0K', c = 'g')
    # plt.scatter(interLayer_temp_21K[:-1], k_21K[:-1], label = '2.1K', c = 'b')
    
    # plt.ylabel('k [W/ (m * K)]')
    # plt.xlabel('Inter layer temperature [K]')
    # # plt.ylim([0.015,0.035])
    # # plt.xlim([1.8,3])
    # plt.legend()
    # plt.grid()
    # plt.show()

sampleHeaterResults()
# coilHeatingResults()