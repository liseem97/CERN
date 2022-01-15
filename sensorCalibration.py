# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:45:04 2020

@author: lise
"""

import pandas as pd
import numpy as np
import general as gen
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

ITS90 = gen.ITS90LHe('SF')
ITS90_NF = gen.ITS90LHe('NF')

cxCurrent = 2.998E-6
# cxCurrent = 10E-6
# cxCurrentPot = 3E-6
MKSfit = gen.MKSfit()

folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
df_KellerAdjustment = pd.read_csv('{}/{}'.format(folder, r'KellerPressureAdjustment.csv'))

# print(ITS90_NF.getTemp(100)-ITS90_NF.getTemp(100.31))
# print(ITS90.getTemp(41.42))
# print(ITS90.getTemp(41.42)-ITS90.getTemp(41.42+0.13))
# print(ITS90.getTemp(31.29))
# print(ITS90.getTemp(31.29)-ITS90.getTemp(31.29+0.1))
# print(ITS90.getTemp(22.99))
# print(ITS90.getTemp(22.99)-ITS90.getTemp(22.99+0.07))
# print(ITS90.getTemp(16.38))
# print(ITS90.getTemp(16.38)-ITS90.getTemp(16.38+0.05))

class RawSensorData():
    #Filename form: info_0_0mbar_RH0_0mA_SH0_0mA_Coil0_0V_0_0Hz_0Ohm_date_time.csv
    def __init__(self, folder, file):
        # get raw data into in a class. 
        self.cxCurrent = cxCurrent
        self.folder = folder
        self.file = file
        self.info = file.split('_')[0]
        self.filePressure = float(file.split('_')[1]+'.'+file.split('_')[2].strip('mbar')) 
        
        
        df = gen.removeChannelName(pd.read_csv('{}/{}'.format(folder, file), header = [0,1]))   
        self.rawData = df
        self.time = df['Time']
        
        self.SF = False
        self.MKS = False
        self.adjustedKeller = False
        
        if self.info == 'RT': 
            self.fileTemp = 295
            self.temp = 295
        elif self.info == 'LN2': 
            self.fileTemp = 77
            self.temp = 77
            self.cxCurrent = 10E-6
        else: 
            if self.filePressure < 95: 
                self.MKS = True
                
                if self.filePressure < 50.42: 
                    self.SF = True
                
                self.MKSvalve = self.rawData['MKSpos']*10 #times 10 to change it from voltage to prosentage open.
                
                self.MKSpressureArray = MKSfit.getPressure(self.rawData['MKS'])
                
                if self.SF == True: 
                    self.MKStempArray = ITS90.getTemp(self.MKSpressureArray)
                else: 
                    self.MKStempArray = ITS90_NF.getTemp(self.MKSpressureArray)
                    
                    
                self.MKSpressure = np.mean(self.MKSpressureArray)
                self.stdMKSpressure = np.std(self.MKSpressureArray)
                
                self.MKStemp = np.mean(self.MKStempArray)
                self.stdMKStemp = np.std(self.MKSpressureArray)
                
            else: 
                self.kellerPressure = self.filePressure #For 4.2 values
                
                if file in df_KellerAdjustment.file.values: 
                    self.adjustedKeller = True
                    df = df_KellerAdjustment[df_KellerAdjustment['file'].astype(str).str.contains(self.file)]
                    self.kellerPressure = float(df['Pressure'].values)
                    self.startPressure = float(df['startPressure'].values)
                    self.endPressure = float(df['endPressure'].values)
                    self.kellerError = float(df['Error'].values)
        
        
            if self.SF == True: 
                self.fileTemp = ITS90.getTemp(self.filePressure)
            else: 
                self.fileTemp = ITS90_NF.getTemp(self.filePressure)
        
            #Temperature used for calibration:
            if self.MKS == True:
                self.temp = self.MKStemp
            else: 
                self.temp = ITS90_NF.getTemp(self.kellerPressure)
        
        
        butter = gen.ButterworthFilter(self.time)
        
        #Voltage values
        self.cxPot_V = butter.filterSignal(self.rawData['cxPot'].to_numpy())
        self.cxBR_V = butter.filterSignal(self.rawData['cxBR'].to_numpy())
        self.cxTL_V = butter.filterSignal(self.rawData['cxTL'].to_numpy())
        self.cxBL_V = butter.filterSignal(self.rawData['cxBL'].to_numpy())
        self.cxTR_V = butter.filterSignal(self.rawData['cxTR'].to_numpy())
        
        #Resistance values
        self.cxPot_R = self.cxPot_V / self.cxCurrent
        self.cxBR_R = self.cxBR_V / self.cxCurrent
        self.cxTL_R = self.cxTL_V / self.cxCurrent
        self.cxBL_R = self.cxBL_V / self.cxCurrent
        self.cxTR_R = self.cxTR_V / self.cxCurrent
        
        
        #To remove spikes in start and end of measurement, 200 points at each side are taken out. 
        spikeCut = 200
        
        self.meanResPot = np.mean(self.cxPot_R[spikeCut:-spikeCut])
        self.meanResBR = np.mean(self.cxBR_R[spikeCut:-spikeCut])
        self.meanResTL = np.mean(self.cxTL_R[spikeCut:-spikeCut])
        self.meanResBL = np.mean(self.cxBL_R[spikeCut:-spikeCut])
        self.meanResTR = np.mean(self.cxTR_R[spikeCut:-spikeCut])
        
        self.stdResPot = np.std(self.cxPot_R[spikeCut:-spikeCut])
        self.stdResBR = np.std(self.cxBR_R[spikeCut:-spikeCut])
        self.stdResTL = np.std(self.cxTL_R[spikeCut:-spikeCut])
        self.stdResBL = np.std(self.cxBL_R[spikeCut:-spikeCut])
        self.stdResTR = np.std(self.cxTR_R[spikeCut:-spikeCut])
        
    def plots(self): 
        
        ###
        # Plot of filtered voltage signals over time
        ###
        
        plt.plot(self.time, self.cxPot_V, label = 'cxPot')
        plt.plot(self.time, self.cxBR_V, label = 'cxBR')
        plt.plot(self.time, self.cxTL_V, label = 'cxTL')
        plt.plot(self.time, self.cxBL_V, label = 'cxBL')
        plt.plot(self.time, self.cxTR_V, label = 'cxTR')
        
        plt.title('Filtered voltage signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.show()
        
        ###
        # Plots of resistance and mean values of resistance
        ###
    
        plt.plot(self.time, self.cxPot_R, label = 'Resistance, cxPot')
        plt.plot(self.time, np.full(len(self.time), self.meanResPot), label = 'Mean R =  {:.1f},cxPot'.format(self.meanResPot))
        
        plt.plot(self.time, self.cxBR_R, label = 'Resistance, cxBR')
        plt.plot(self.time, np.full(len(self.time), self.meanResBR), label = 'Mean R = {:.1f}, cxBR'.format(self.meanResBR))
        
        plt.plot(self.time, self.cxTL_R, label = 'Resistance, cxTL')
        plt.plot(self.time, np.full(len(self.time), self.meanResTL), label = 'Mean R = {:.1f}, cxTL'.format(self.meanResTL))
        
        plt.plot(self.time, self.cxBL_R, label = 'Resistance, cxBL')
        plt.plot(self.time, np.full(len(self.time), self.meanResBL), label = 'Mean R = {:.1f}, cxBL'.format(self.meanResBL))
        
        plt.plot(self.time, self.cxTR_R, label = 'Resistance, cxTR')
        plt.plot(self.time, np.full(len(self.time), self.meanResTR), label = 'Mean R = {:.1f}, cxTR'.format(self.meanResTR))
        
        plt.title('Resistance signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.show()
        
        ###
        # Plots of resistance of for each sensor
        ###
    
        plt.plot(self.time[200:-200], self.cxPot_R[200:-200])
        plt.title('Cernox Pot resistance signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.grid()
        plt.show()
    
        plt.plot(self.time[200:-200], self.cxTL_R[200:-200])
        plt.title('Cernox Top Left resistance signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.grid()
        plt.show()

        plt.plot(self.time[200:-200], self.cxTR_R[200:-200])
        plt.title('Cernox Top Right resistance signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.grid()
        plt.show()

        plt.plot(self.time[200:-200], self.cxBL_R[200:-200])
        plt.title('Cernox Bottom Left resistance signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.grid()
        plt.show()

        plt.plot(self.time[200:-200], self.cxBR_R[200:-200])
        plt.title('Cernox Bottom Right resistance signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.grid()
        plt.show()
        
        
        if self.MKS == True: 
            ###
            # MKS temperature signal 
            ###
            
            plt.plot(self.time, self.MKStempArray, label = 'MKS temperature')
            plt.plot(self.time, np.full(len(self.time), self.MKStemp), label = 'Mean T =  {:.2f} K'.format(self.MKStemp))
     
        
            plt.title('MKS signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
            plt.xlabel('Time [s]')
            plt.ylabel('Temperature [K]')
            plt.legend()
            plt.grid()
            plt.show()
            
            
            ###
            # MKS valve signal 
            ###
            
            plt.plot(self.time, self.MKSvalve)
           
            plt.title('MKS signal for {}, {}mbar, {:.2f}K'.format(self.info, self.filePressure, self.fileTemp))
            plt.xlabel('Time [s]')
            plt.ylabel('Prosentage open [%]')
            # plt.legend()
            plt.grid()
            plt.show()
            
        pass
        
    def getDict(self):
        auxDict = {}
        
        auxDict.update({'file': self.file}) 
        
        auxDict.update({'info': self.info}) 
        auxDict.update({'filePressure': self.filePressure})
        auxDict.update({'fileTemp': self.fileTemp})
        auxDict.update({'MKS regime': self.MKS})
        auxDict.update({'SF regime': self.SF})
        auxDict.update({'adjustedKeller': self.adjustedKeller})
        
        
        auxDict.update({'Temperature': self.temp})  
        
        #Mean values in ss conditions
        auxDict.update({'meanResPot': self.meanResPot})
        auxDict.update({'meanResBR': self.meanResBR})
        auxDict.update({'meanResTL': self.meanResTL})
        auxDict.update({'meanResBL': self.meanResBL})
        auxDict.update({'meanResTR': self.meanResTR})
        
        #Standard deviations for mean values. 
        auxDict.update({'stdPot': self.stdResPot})
        auxDict.update({'stdBR': self.stdResBR})
        auxDict.update({'stdTL': self.stdResTL})
        auxDict.update({'stdBL': self.stdResBL})
        auxDict.update({'stdTR': self.stdResTR})
        
        if self.MKS == True: 
            auxDict.update({'MKSpressure': self.MKSpressure})
            auxDict.update({'MKStemp': self.MKStemp})
            auxDict.update({'stdMKStemp': self.stdMKStemp})
        if self.adjustedKeller == True: 
            auxDict.update({'KellerPressure': self.kellerPressure})
            auxDict.update({'startPressure': self.startPressure})
            auxDict.update({'endPressure': self.endPressure})
            auxDict.update({'kellerError': self.kellerError})
        
        return auxDict


def CalibrationLoop():
    # analyse the raw data to a new file
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements'
    allFilesInFolder = gen.getCSV(folder)
    rowList = []
    
    ### SUPERFLUID CALIBRATION FILES
    
    unsortedCalibFiles = filter(lambda x: x.startswith('CalibSF'), allFilesInFolder)
    calibFiles = sorted(unsortedCalibFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
    
    for fileName in calibFiles: 
        file = RawSensorData(folder, fileName)
        file.plots()
        
        rowList.append(file.getDict())
    
    ### STEADY STATE CHECK FILES
    unsortedSSCheckFiles = filter(lambda x: x.startswith('SSCheck'), allFilesInFolder)
    SSCheckFiles = sorted(unsortedSSCheckFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
    
    for fileName in SSCheckFiles: 
        file = RawSensorData(folder, fileName)
        file.plots()
        
        rowList.append(file.getDict())
    
    ### COOLDOWN2 FILES
    unsortedCooldownFiles = filter(lambda x: x.startswith('Cool'), allFilesInFolder) #Gets all 'cooldown files'
    cooldownFiles = sorted(unsortedCooldownFiles, key = lambda x: int(x.split('_')[1]), reverse = False)
  
    for fileName in cooldownFiles: 
        file = RawSensorData(folder, fileName)
        file.plots()
        
        rowList.append(file.getDict())
    
    ### NORMALFLUID CALIBRATION FILES
    unsortedCalibFiles = filter(lambda x: x.startswith('CalibNF'), allFilesInFolder)
    calibFiles = sorted(unsortedCalibFiles, key = lambda x: int(x.split('_')[1]), reverse = False)

    for fileName in calibFiles: 
        file = RawSensorData(folder, fileName)
        file.plots()
        
        rowList.append(file.getDict())
    
    ### ROOMTEMPERATURE FILE
    File = RawSensorData(folder, r'RT_1000_0mbar_RH0_0mA_SH0_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200908_095535.csv')
    File.plots()
    rowList.append(File.getDict())
    
    ### LIQUID NITROGEN FILE
    File = RawSensorData(folder, r'LN2_1000_0mbar_RH0_0mA_SH0_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200908_124031.csv')
    File.plots()
    rowList.append(File.getDict())
    
    
    dataCalib = pd.DataFrame(rowList)    
    dataCalib.to_csv('Analysed_Calibration.csv')
    return
    
    
class SensorData():
    #  get the data from the analysed file.
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        #OBS file with data must have these columns! 
        df = pd.read_csv('{}/{}'.format(folder, file))
        self.data = df
        
        self.temperature = df['Temperature']
        self.resistancePot = df['meanResPot']
        self.resistanceBR = df['meanResBR']
        self.resistanceTL = df['meanResTL']
        self.resistanceBL = df['meanResBL']
        self.resistanceTR = df['meanResTR']
       
        self.pressure = df['filePressure']
        self.number = df['Unnamed: 0']
        
        
    def getGroup(self, groupby, value): 
        self.data = self.data.groupby(groupby).get_group(value)
        self.temperature = self.data['Temperature']
        self.resistancePot = self.data['meanResPot']
        self.resistanceBR = self.data['meanResBR']
        self.resistanceTL = self.data['meanResTL']      
        self.resistanceBL = self.data['meanResBL']
        self.resistanceTR = self.data['meanResTR']  
        self.pressure = self.data['filePressure']
        self.number = self.data['Unnamed: 0']



def plotCalibrationPoints(file, log = False): 
    folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
    sensorData = SensorData(folder, file)
    
    plt.title('Pot sensor')
    plt.scatter(sensorData.resistancePot, sensorData.temperature, c = sensorData.number)
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    cbar = plt.colorbar()
    cbar.set_label('index')
    if log == True: 
        plt.loglog()
    plt.show()
    
    # #Check for estimated 77 K point
    # poly = np.polynomial.polynomial.Polynomial.fit(np.log10(sensorData.temperature), np.log10(sensorData.resistancePot), 1)
    # print('Pot', 10**poly(np.log10(77)),'Ohm')
    
    plt.title('Bottom Right sensor')
    plt.scatter(sensorData.resistanceBR, sensorData.temperature, c = sensorData.number)
    plt.grid() 
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    cbar = plt.colorbar()
    cbar.set_label('index')
    if log == True: 
        plt.loglog()
    plt.show()

    # #Check for estimated 77 K point
    # poly = np.polynomial.polynomial.Polynomial.fit(np.log10(sensorData.temperature), np.log10(sensorData.resistanceBR), 1)
    # print('BR', 10**poly(np.log10(77)),'Ohm')
    
    plt.title('Top Left sensor')
    plt.scatter(sensorData.resistanceTL, sensorData.temperature, c = sensorData.number)
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    cbar = plt.colorbar()
    cbar.set_label('index')
    if log == True: 
        plt.loglog()
    plt.show()
    
    # #Check for estimated 77 K point
    # poly = np.polynomial.polynomial.Polynomial.fit(np.log10(sensorData.temperature), np.log10(sensorData.resistanceTL), 1)
    # print('TL', 10**poly(np.log10(77)),'Ohm')
    
    plt.title('Bottom Left sensor')
    plt.scatter(sensorData.resistanceBL, sensorData.temperature, c = sensorData.number)
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    cbar = plt.colorbar()
    cbar.set_label('index')
    if log == True: 
        plt.loglog()
    plt.show()
    
    # #Check for estimated 77 K point
    # poly = np.polynomial.polynomial.Polynomial.fit(np.log10(sensorData.temperature), np.log10(sensorData.resistanceBL), 1)
    # print('BL', 10**poly(np.log10(77)),'Ohm')
    
    plt.title('Top Right sensor')
    plt.scatter(sensorData.resistanceTR, sensorData.temperature, c = sensorData.number)
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    cbar = plt.colorbar()
    cbar.set_label('index')
    if log == True: 
        plt.loglog()
    plt.show()
    
    # #Check for estimated 77 K point
    # poly = np.polynomial.polynomial.Polynomial.fit(np.log10(sensorData.temperature), np.log10(sensorData.resistanceTR), 1)
    # print('TR', 10**poly(np.log10(77)),'Ohm')
    
    
    plt.title('All sensors')
    plt.scatter(sensorData.resistancePot, sensorData.temperature, label = 'Pot')
    plt.scatter(sensorData.resistanceBR, sensorData.temperature, label = 'BR')
    plt.scatter(sensorData.resistanceTL, sensorData.temperature, label = 'TL')
    plt.scatter(sensorData.resistanceBL, sensorData.temperature, label = 'BL')
    plt.scatter(sensorData.resistanceTR, sensorData.temperature, label = 'TR')
    plt.grid()
    plt.legend()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    if log == True: 
        plt.loglog()
    plt.show()


class Cheby(): 
    def __init__(self, data, degree): 
        self.data = data
        
        self.potCheby = np.polynomial.chebyshev.Chebyshev.fit(np.log10(self.data.resistancePot), self.data.temperature, degree)
        self.potDomain = 10**self.potCheby.domain
        
        self.BRCheby = np.polynomial.chebyshev.Chebyshev.fit(np.log10(self.data.resistanceBR), self.data.temperature, degree)
        self.BRDomain = 10**self.BRCheby.domain
        
        self.TLCheby = np.polynomial.chebyshev.Chebyshev.fit(np.log10(self.data.resistanceTL), self.data.temperature, degree)
        self.TLDomain = 10**self.TLCheby.domain
        
        self.BLCheby = np.polynomial.chebyshev.Chebyshev.fit(np.log10(self.data.resistanceBL), self.data.temperature, degree)
        self.BLDomain = 10**self.BLCheby.domain
        
        self.TRCheby = np.polynomial.chebyshev.Chebyshev.fit(np.log10(self.data.resistanceTR), self.data.temperature, degree)
        self.TRDomain = 10**self.TRCheby.domain

    def Pot(self, resistance):
        return self.potCheby(np.log10(resistance))
    def BR(self, resistance):
        return self.BRCheby(np.log10(resistance))
    def TL(self, resistance):
        return self.TLCheby(np.log10(resistance))
    def BL(self, resistance):
        return self.BLCheby(np.log10(resistance))
    def TR(self, resistance):
        return self.TRCheby(np.log10(resistance))
    pass


class PolyFit():
    def __init__(self, data, degree = 9): 
        self.data = data
        
        self.potPoly = np.polynomial.polynomial.Polynomial.fit(1/np.log10(self.data.resistancePot), np.log10(self.data.temperature), degree)
        self.potDomain = 10**((1/self.potPoly.domain))
        
        self.BRPoly = np.polynomial.polynomial.Polynomial.fit(1/np.log10(self.data.resistanceBR), np.log10(self.data.temperature), degree)
        self.BRDomain = 10**((1/self.BRPoly.domain))

        self.TLPoly = np.polynomial.polynomial.Polynomial.fit(1/np.log10(self.data.resistanceTL), np.log10(self.data.temperature), degree)
        self.TLDomain = 10**(1/self.TLPoly.domain)
        
        self.BLPoly = np.polynomial.polynomial.Polynomial.fit(1/np.log10(self.data.resistanceBL), np.log10(self.data.temperature), degree)
        self.BLDomain = 10**(1/self.BLPoly.domain)

        self.TRPoly = np.polynomial.polynomial.Polynomial.fit(1/np.log10(self.data.resistanceTR), np.log10(self.data.temperature), degree)
        self.TRDomain = 10**(1/self.TRPoly.domain)

    def Pot(self, resistance):
        return 10**(self.potPoly(1/np.log10(resistance)))
    def BR(self, resistance):
        return 10**(self.BRPoly(1/np.log10(resistance)))
    def TL(self, resistance):
        return 10**(self.TLPoly(1/np.log10(resistance)))
    def BL(self, resistance):
        return 10**(self.BLPoly(1/np.log10(resistance)))
    def TR(self, resistance):
        return 10**(self.TRPoly(1/np.log10(resistance)))
    pass


class PolyFitTVO():
    def __init__(self, data, degree = 2): 
        self.data = data
        
        self.potPoly = np.polynomial.polynomial.Polynomial.fit(1000/self.data.resistancePot, self.data.temperature, degree)
        self.potDomain = 1000/self.potPoly.domain
        
        self.BRPoly = np.polynomial.polynomial.Polynomial.fit(1000/self.data.resistanceBR, self.data.temperature, degree)
        self.BRDomain = 1000/self.BRPoly.domain
        
        self.TLPoly = np.polynomial.polynomial.Polynomial.fit(1000/self.data.resistanceTL, self.data.temperature, degree)
        self.TLDomain = 1000/self.TLPoly.domain
        
        self.BLPoly = np.polynomial.polynomial.Polynomial.fit(1000/self.data.resistanceBL, self.data.temperature, degree)
        self.BLDomain = 1000/self.BLPoly.domain
        
        self.TRPoly = np.polynomial.polynomial.Polynomial.fit(1000/self.data.resistanceTR, self.data.temperature, degree)
        self.TRDomain = 1000/self.TRPoly.domain

    def Pot(self, resistance):
        return self.potPoly(1000/resistance)
    def BR(self, resistance):
        return self.BRPoly(1000/resistance)
    def TL(self, resistance):
        return self.TLPoly(1000/resistance)
    def BL(self, resistance):
        return self.BLPoly(1000/resistance)
    def TR(self, resistance):
        return self.TRPoly(1000/resistance)
    pass


class SensorCalibration(): 
    def __init__(self, folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'): 
        ## Superfluid domain: 
        self.dataSF = SensorData(folder, r'CalibrationSF.csv')
        self.fitSF = Cheby(self.dataSF, 10)
        ## MKS + 4.2 K domain: 
        self.data4K = SensorData(folder, r'Calibration4K.csv')
        self.fit4K = Cheby(self.data4K, 5)
        ## [1.7 K, 15 K] domain: 
        self.data15K = SensorData(folder, r'Calibration77K.csv')
        self.fit15K = PolyFitTVO(self.data15K, 2)
        
        ## [1.7 K, 15 K] domain: 
        self.data15K_TR = SensorData(folder, r'Calibration295K.csv')
        self.fit15K_TR = PolyFitTVO(self.data15K_TR, 2)
        
    def Pot(self, resArray):
        if np.min(resArray) >= np.min(self.fitSF.potDomain): 
            return self.fitSF.Pot(resArray)
        elif np.min(resArray) >= np.min(self.fit4K.potDomain): 
            return self.fit4K.Pot(resArray)
        else: 
            return self.fit15K.Pot(resArray)
        
    def BR(self, resArray):
        if np.min(resArray) >= np.min(self.fitSF.BRDomain): 
            return self.fitSF.BR(resArray)
        elif np.min(resArray) >= np.min(self.fit4K.BRDomain): 
            return self.fit4K.BR(resArray)
        else: 
            return self.fit15K.BR(resArray)
        
    def TL(self, resArray):
        if np.min(resArray) >= np.min(self.fitSF.TLDomain): 
            return self.fitSF.TL(resArray)
        elif np.min(resArray) >= np.min(self.fit4K.TLDomain): 
            return self.fit4K.TL(resArray)
        else: 
            return self.fit15K.TL(resArray)
        
    def TR(self, resArray):
        if np.min(resArray) >= np.min(self.fitSF.TRDomain): 
            return self.fitSF.TR(resArray)
        elif np.min(resArray) >= np.min(self.fit4K.TRDomain): 
            return self.fit4K.TR(resArray)
        else: 
            return self.fit15K_TR.TR(resArray)
        
    def BL(self, resArray):
        if np.min(resArray) >= np.min(self.fitSF.BLDomain): 
            return self.fitSF.BL(resArray)
        elif np.min(resArray) >= np.min(self.fit4K.BLDomain): 
            return self.fit4K.BL(resArray)
        else: 
            return self.fit15K.BL(resArray)
        
        
        
#DIVERE TESTS OF THE CALIBRATION: 


    
class ThreePointCalibration():
    def __init__(self, sensor): 
        self.degree = 11

        if sensor == 'TL':
            self.c = [0.0160117116, -0.529951612, 7.85425581, -68.7397115, 394.328751, -1555.0961, 4296.50658, -8303.23385, 10977.0431, -9427.66564, 4713.44899, -1028.79422]
        if sensor == 'TLnew':
             self.c = [2.60054198329723,	-101.531174365932,	1781.10005856969,	-18487.1226163326,	125734.416718120,	-585494.082310502,	1890554.63977116,	-4180370.62558734,	6059994.32523248,	-5206173.72433701,	2023979.77754778,	-15706.2812663871]
        if sensor == 'BR':
            self.c = [0.00234108904240316, -0.0734598319975664, 1.00459852843688, -7.79037185782785, 37.0417357934436, -106.451533365372, 151.047757233412, 69.8155276829130, -723.113946215836, 1347.16435354136, -1173.79024996956, 417.562650764724]
        if sensor == 'TR':
            pass
        if sensor == 'BL':
            pass
        self.poly = np.polynomial.polynomial.Polynomial(self.c[::-1])
        
    def getTemp(self, resistance): 
        return 10**self.poly(np.log10(resistance))
        
def powerFunction(x, a, b, c): 
    return a * np.power(x, b) + c


class PowerFit(): 
    def __init__(self, data): 
        boundary = ([0, -np.inf, 0], [np.inf, 0, np.inf])
        guess = [4000,-1, 1]
        
        self.data = data
        self.potFitCoeff, self.potFitCov = curve_fit(powerFunction, self.data.resistancePot, self.data.temperature, p0 = guess, bounds = boundary)
        self.BRFitCoeff, self.BRFitCov = curve_fit(powerFunction, self.data.resistanceBR, self.data.temperature, p0 = guess, bounds = boundary)
        self.TLFitCoeff, self.TLFitCov = curve_fit(powerFunction, self.data.resistanceTL, self.data.temperature, p0 = guess, bounds = boundary)
        self.BLFitCoeff, self.BLFitCov = curve_fit(powerFunction, self.data.resistanceBL, self.data.temperature, p0 = guess, bounds = boundary)
        self.TRFitCoeff, self.TRFitCov = curve_fit(powerFunction, self.data.resistanceTR, self.data.temperature, p0 = guess, bounds = boundary)
        
    def Pot(self, resistance):
        return powerFunction(resistance, *self.potFitCoeff)
    def BR(self, resistance):
        return powerFunction(resistance, *self.BRFitCoeff)
    def TL(self, resistance):
        return powerFunction(resistance, *self.TLFitCoeff)
    def BL(self, resistance):
        return powerFunction(resistance, *self.BLFitCoeff)
    def TR(self, resistance):
        return powerFunction(resistance, *self.TRFitCoeff)
    pass

    

def plotFit3pointCalib(folder, file, degree):
    df = SensorData(folder, file)    
    
    fitCheby = Cheby(df, 5)
    
    fitPolyCx = PolyFit(df, 3)
    
    fitPolyTVO = PolyFitTVO(df, 2)
    
    # fitPower = PowerFit(df)
    
    #Three point calibration
    threePointCalib_BR = ThreePointCalibration('BR')
    
    
    
    # minus value on resistance range determined by resistances in sensor. (so that it dont go to 0 resistance)
    
    #-2000 in resistance to se what happends for lower resitances aka higher temperatures
    resBR = np.arange(int(np.min(fitCheby.BRDomain)), int(np.max(fitCheby.BRDomain)+1))

    TempBR = np.array(threePointCalib_BR.getTemp(resBR), dtype = int)
    index = np.where(TempBR == 10)
    print('10 K = ', np.mean(resBR[index]), 'Ohm')
    
    
    # plt.title('3-point calibration BR')
    plt.title('BR fits')
    
    plt.plot(resBR, fitPolyCx.BR(resBR), label = 'PolyCxfit 3-degree')
    plt.plot(resBR, fitPolyTVO.BR(resBR), label = 'PolyTVOfit 2-degree')
    plt.plot(resBR, fitCheby.BR(resBR), label = 'Cheby 5-degree')
    # plt.plot(resBR, fitPower.BR(resBR), label = 'Powerfit')

    # plt.plot(resBR, threePointCalib_BR.getTemp(resBR), label = '3-point calib, 11 degree fit')
    plt.scatter(df.resistanceBR, df.temperature, label = 'measurement')
    
    plt.legend()
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.xlim([1500,20000])
    plt.ylim([1.7,10])
    plt.show()
    
    resTL = np.arange(int(np.min(fitCheby.TLDomain)), int(np.max(fitCheby.TLDomain)+1))
    
    plt.title('TL fits')
    
    plt.plot(resTL, fitPolyCx.TL(resTL), label = 'PolyCxfit 3-degree')
    plt.plot(resTL, fitPolyTVO.TL(resTL), label = 'PolyTVOfit 2-degree')
    plt.plot(resTL, fitCheby.TL(resTL), label = 'Cheby 5-degree')
    plt.scatter(df.resistanceTL, df.temperature, label = 'measurement')
    plt.legend()
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    # plt.xlim([1500,20000])
    plt.ylim([1.7,10])
    plt.show()
    
    resBL = np.arange(int(np.min(fitCheby.BLDomain)), int(np.max(fitCheby.BLDomain)+1))
    
    plt.title('BL fits')
    
    plt.plot(resBL, fitPolyCx.BL(resBL), label = 'PolyCxfit 3-degree')
    plt.plot(resBL, fitPolyTVO.BL(resBL), label = 'PolyTVOfit 2-degree')
    plt.plot(resBL, fitCheby.BL(resBL), label = 'Cheby 5-degree')
    plt.scatter(df.resistanceBL, df.temperature, label = 'measurement')
    plt.legend()
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    # plt.xlim([1500,20000])
    plt.ylim([1.7,10])
    plt.show()
    
    resTR = np.arange(int(np.min(fitCheby.TRDomain)), int(np.max(fitCheby.TRDomain)+1))
    
    plt.title('TR fits')
    
    plt.plot(resTR, fitPolyCx.TR(resTR), label = 'PolyCxfit 3-degree')
    plt.plot(resTR, fitPolyTVO.TR(resTR), label = 'PolyTVOfit 2-degree')
    plt.plot(resTR, fitCheby.TR(resTR), label = 'Cheby 5-degree')
    plt.scatter(df.resistanceTR, df.temperature, label = 'measurement')
    plt.legend()
    plt.grid()
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    # plt.xlim([1500,20000])
    plt.ylim([1.7,10])
    plt.show()
    
    

def checkOffset(): 
    os.chdir(r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files')
    
    fit = SensorCalibration()
        
    df2 = SensorData(r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files', 'Analysed_Calibration.csv')
    
    
    offsetPot = df2.temperature - fit.Pot(df2.resistancePot)
    offsetTR = df2.temperature - fit.TR(df2.resistanceTR)
    offsetBL = df2.temperature - fit.BL(df2.resistanceBL)
    offsetTL = df2.temperature - fit.TL(df2.resistanceTL)
    offsetBR = df2.temperature - fit.BR(df2.resistanceBR)
    
    auxDict = {'Temperature': df2.temperature, 
               'cheby2Temp_Pot': fit.Pot(df2.resistancePot), 
               'offsetPot': offsetPot, 
               'cheby2Temp_TR': fit.TR(df2.resistanceTR), 
               'offsetTR': offsetTR,
               'cheby2Temp_BL': fit.BL(df2.resistanceBL), 
               'offsetBL': offsetBL,
               'cheby2Temp_TL': fit.TL(df2.resistanceTL), 
               'offsetTL': offsetTL,
               'cheby2Temp_BR': fit.BR(df2.resistanceBR), 
               'offsetBR': offsetBR
               }
    
    data = pd.DataFrame.from_dict(auxDict)
    
    plt.scatter(data['Temperature'], data['offsetPot']*1000, label = 'cxPot')
    plt.scatter(data['Temperature'], data['offsetTR']*1000, label = 'cxTR')
    plt.scatter(data['Temperature'], data['offsetTL']*1000, label = 'cxTL')
    plt.scatter(data['Temperature'], data['offsetBR']*1000, label = 'cxBR')
    plt.scatter(data['Temperature'], data['offsetBL']*1000, label = 'cxBL')
    plt.title('Residual between Cheby 5-degree fit and All points')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Residual [mK]')
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([1.6,4.3])
    plt.show()

    
    # data.to_csv('Offset_Keller_MKS_{}_deg{}.csv'.format(fitType, degree))
    
    pass


def transition():
    df = SensorData(r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements', 'Analysed_Calibration_test.csv')    
    cheby = Cheby(df, 2)
    
    File = RawSensorData(r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements', r'pressGazo_46_5mbar_RH0_0mA_SH0_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200813_022337.csv')
    File.plots()
    
    plt.plot(File.time[50:-50], cheby.Pot(File.cxPot_R[50:-50]))
    plt.title('Cernox Pot, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    plt.plot(File.time[50:-50], cheby.TL(File.cxTL_R[50:-50]))
    plt.title('Cernox Top Left, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()

    plt.plot(File.time[50:-50], cheby.TR(File.cxTR_R[50:-50]))
    plt.title('Cernox Top Right, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()

    plt.plot(File.time[50:-50], cheby.BL(File.cxBL_R[50:-50]))
    plt.title('Cernox Bottom Left, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()

    plt.plot(File.time[50:-50], cheby.BR(File.cxBR_R[50:-50]))
    plt.title('Cernox Bottom Right, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    plt.plot(File.time[50:-50], cheby.Pot(File.cxPot_R[50:-50]), label ='Pot')
    plt.plot(File.time[50:-50], cheby.BR(File.cxBR_R[50:-50]), label = 'BR')
    plt.plot(File.time[50:-50], cheby.BL(File.cxBL_R[50:-50]), label = 'BL')
    plt.plot(File.time[50:-50], cheby.TR(File.cxTR_R[50:-50]), label = 'TR')
    plt.plot(File.time[50:-50], cheby.TL(File.cxTL_R[50:-50]), label = 'TL')
    plt.title('Cernox sensors, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.legend()
    plt.show()
    
    File = RawSensorData(r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Not used files', r'CalibNF85_47_0mbar_RH0_0mA_SH0_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200813_013732.csv')
    File.plots()
    
    plt.plot(File.time[100:-100], cheby.Pot(File.cxPot_R[100:-100]))
    plt.title('Cernox Pot, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    plt.plot(File.time[100:-100], cheby.TL(File.cxTL_R[100:-100]))
    plt.title('Cernox Top Left, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()

    plt.plot(File.time[100:-100], cheby.TR(File.cxTR_R[100:-100]))
    plt.title('Cernox Top Right, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()

    plt.plot(File.time[100:-100], cheby.BL(File.cxBL_R[100:-100]))
    plt.title('Cernox Bottom Left, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()

    plt.plot(File.time[100:-100], cheby.BR(File.cxBR_R[100:-100]))
    plt.title('Cernox Bottom Right, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    plt.plot(File.time[100:-100], cheby.Pot(File.cxPot_R[100:-100]), label ='Pot')
    plt.plot(File.time[100:-100], cheby.BR(File.cxBR_R[100:-100]), label = 'BR')
    plt.plot(File.time[100:-100], cheby.BL(File.cxBL_R[100:-100]), label = 'BL')
    plt.plot(File.time[100:-100], cheby.TR(File.cxTR_R[100:-100]), label = 'TR')
    plt.plot(File.time[100:-100], cheby.TL(File.cxTL_R[100:-100]), label = 'TL')
    plt.title('Cernox sensors, {}, {}mbar, {:.2f}K'.format(File.info, File.filePressure, File.fileTemp))
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.legend()
    plt.show()
    
    pass



def plotChebychev(folder, file, degree):
    df = SensorData(folder, file) 
    fit = Cheby(df, degree)

    resPot = np.arange(int(np.min(fit.potDomain)), int(np.max(fit.potDomain)+1))

    plt.title('Chebyfit Pot, degree = {}'.format(degree))
    plt.scatter(df.resistancePot, df.temperature, label = 'measurement')
    plt.plot(resPot, fit.Pot(resPot), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.show()
    
    residualPot = df.temperature - fit.Pot(df.resistancePot)
    plt.title('Residual chebychev Pot, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualPot*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    errorPot_RMSE = np.sqrt(np.mean(np.power(fit.Pot(df.resistancePot)-df.temperature,2)))
    print('The RMSE error for cernox pot sensor with a {} degree chebychev fit is: {} mK'.format(degree, errorPot_RMSE*1000))
    

    resBR = np.arange(int(np.min(fit.BRDomain)), int(np.max(fit.BRDomain)+1))
  
    plt.title('Chebyfit BR, degree = {}'.format(degree))
    plt.scatter(df.resistanceBR, df.temperature, label = 'measurement')
    plt.plot(resBR, fit.BR(resBR), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.show()
    
    residualBR = df.temperature - fit.BR(df.resistanceBR)
    plt.title('Residual chebychev BR, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualBR*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    errorBR_RMSE = np.sqrt(np.mean(np.power(fit.BR(df.resistanceBR)-df.temperature,2)))
    print('The RMSE error for cernox BR sensor with a {} degree chebychev fit is: {} mK'.format(degree, errorBR_RMSE*1000))

    resTL = np.arange(int(np.min(fit.TLDomain)), int(np.max(fit.TLDomain)+1))
   
    plt.title('Chebyfit TL, degree = {}'.format(degree))
    plt.scatter(df.resistanceTL, df.temperature, label = 'measurement')
    plt.plot(resTL, fit.TL(resTL), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.show()
    
    residualTL = df.temperature - fit.TL(df.resistanceTL)
    plt.title('Residual chebychev TL, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualTL*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    errorTL_RMSE = np.sqrt(np.mean(np.power(fit.TL(df.resistanceTL)-df.temperature,2)))
    print('The RMSE error for cernox TL sensor with a {} degree chebychev fit is: {} mK'.format(degree, errorTL_RMSE*1000))


    resBL = np.arange(int(np.min(fit.BLDomain)), int(np.max(fit.BLDomain)+1))
   
    plt.title('Chebyfit BL, degree = {}'.format(degree))
    plt.scatter(df.resistanceBL, df.temperature, label = 'measurement')
    plt.plot(resBL, fit.BL(resBL), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.show()
    
    residualBL = df.temperature - fit.BL(df.resistanceBL)
    plt.title('Residual chebychev BL, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualBL*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    errorBL_RMSE = np.sqrt(np.mean(np.power(fit.BL(df.resistanceBL)-df.temperature,2)))
    print('The RMSE error for cernox BL sensor with a {} degree chebychev fit is: {} mK'.format(degree, errorBL_RMSE*1000))


    resTR = np.arange(int(np.min(fit.TRDomain)), int(np.max(fit.TRDomain)+1))
    
    plt.title('Chebyfit TR, degree = {}'.format(degree))
    plt.scatter(df.resistanceTR, df.temperature, label = 'measurement')
    plt.plot(resTR, fit.TR(resTR), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.show()
    
    residualTR = df.temperature - fit.TR(df.resistanceTR)
    plt.title('Residual chebychev TR, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualTR*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.show()
    
    errorTR_RMSE = np.sqrt(np.mean(np.power(fit.TR(df.resistanceTR)-df.temperature,2)))
    print('The RMSE error for cernox TR sensor with a {} degree chebychev fit is: {} mK'.format(degree, errorTR_RMSE*1000))

    
def plotPolyCernox(folder, file, degree):
    df = SensorData(folder, file) 
    fit = PolyFit(df, degree)
     
    resPot = np.arange(int(np.min(fit.potDomain)), int(np.max(fit.potDomain)+1))
    
    plt.title('Polynomial Cernox fit Pot, degree = {}'.format(degree))
    plt.scatter(df.resistancePot, df.temperature, label = 'measurement')
    plt.plot(resPot, fit.Pot(resPot), label = 'fit')
    plt.ylabel('T [K]')
    plt.xlabel('R [Ohm]')
    plt.loglog()
    plt.legend()
    plt.grid()
    plt.show()
    
    residualPot = df.temperature - fit.Pot(df.resistancePot)
    plt.title('Residual Polynomial Cernox fit Pot, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualPot*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorPot_RMSE = np.sqrt(np.mean(np.power(fit.Pot(df.resistancePot)-df.temperature,2)))
    print('The RMSE error for cernox pot sensor with a {} degree polynomial cernox fit is: {} mK'.format(degree, errorPot_RMSE*1000))
    
    
    resBR = np.arange(int(np.min(fit.BRDomain)), int(np.max(fit.BRDomain)+1))
   
    plt.title('Polynomial Cernox fit BR, degree = {}'.format(degree))
    plt.scatter(df.resistanceBR, df.temperature, label = 'measurement')
    plt.plot(resBR, fit.BR(resBR), label = 'fit')
    plt.ylabel('T [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.show()
    
    residualBR = df.temperature - fit.BR(df.resistanceBR)
    plt.title('Residual Polynomial Cernox fit BR, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualBR*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorBR_RMSE = np.sqrt(np.mean(np.power(fit.BR(df.resistanceBR)-df.temperature,2)))
    print('The RMSE error for cernox BR sensor with a {} degree polynomial cernox fit is: {} mK'.format(degree, errorBR_RMSE*1000))

    
    resTL = np.arange(int(np.min(fit.TLDomain)), int(np.max(fit.TLDomain)+1))
   
    plt.title('Polynomial Cernox fit TL, degree = {}'.format(degree))
    plt.scatter(df.resistanceTL, df.temperature, label = 'measurement')
    plt.plot(resTL, fit.TL(resTL), label = 'fit')
    plt.ylabel('T [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.show()
    
    residualTL = df.temperature - fit.TL(df.resistanceTL)
    plt.title('Residual Polynomial Cernox fit TL, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualTL*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorTL_RMSE = np.sqrt(np.mean(np.power(fit.TL(df.resistanceTL)-df.temperature,2)))
    print('The RMSE error for cernox TL sensor with a {} degree polynomial cernox fit is: {} mK'.format(degree, errorTL_RMSE*1000))

 
    resBL = np.arange(int(np.min(fit.BLDomain)), int(np.max(fit.BLDomain)+1))
   
    plt.title('Polynomial Cernox fit BL, degree = {}'.format(degree))
    plt.scatter(df.resistanceBL, df.temperature, label = 'measurement')
    plt.plot(resBL, fit.BL(resBL), label = 'fit')
    plt.ylabel('T [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.show()
    
    residualBL = df.temperature - fit.BL(df.resistanceBL)
    plt.title('Residual Polynomial Cernox fit BL, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualBL*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorBL_RMSE = np.sqrt(np.mean(np.power(fit.BL(df.resistanceBL)-df.temperature,2)))
    print('The RMSE error for cernox BL sensor with a {} degree polynomial cernox fit is: {} mK'.format(degree, errorBL_RMSE*1000))

    resTR = np.arange(int(np.min(fit.TRDomain)), int(np.max(fit.TRDomain)+1))
   
    plt.title('Polynomial Cernox fit TR, degree = {}'.format(degree))
    plt.scatter(df.resistanceTR, df.temperature, label = 'measurement')
    plt.plot(resTR, fit.TR(resTR), label = 'fit')
    plt.ylabel('T [K]')
    plt.xlabel('R [Ohm]')
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.show()
    
    residualTR = df.temperature - fit.TR(df.resistanceTR)
    plt.title('Residual Polynomial Cernox fit TR, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualTR*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorTR_RMSE = np.sqrt(np.mean(np.power(fit.TR(df.resistanceTR)-df.temperature,2)))
    print('The RMSE error for cernox TR sensor with a {} degree polynomial cernox fit is: {} mK'.format(degree, errorTR_RMSE*1000))



def plotPolyTVO(folder, file, degree):
    df = SensorData(folder, file) 
    fit = PolyFitTVO(df, degree)

    resPot = np.arange(int(np.min(fit.potDomain)), int(np.max(fit.potDomain)+1))
    
    plt.title('PolyTVOfit Pot, degree = {}'.format(degree))
    plt.scatter(df.resistancePot, df.temperature, label = 'measurement')
    plt.plot(resPot, fit.Pot(resPot), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    residualPot = df.temperature - fit.Pot(df.resistancePot)
    plt.title('Residual Polynomial TVO fit Pot, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualPot*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorPot_RMSE = np.sqrt(np.mean(np.power(fit.Pot(df.resistancePot)-df.temperature,2)))
    print('The RMSE error for cernox pot sensor with a {} degree polynomial TVO fit is: {} mK'.format(degree, errorPot_RMSE*1000))
    
 
    resBR = np.arange(int(np.min(fit.BRDomain)), int(np.max(fit.BRDomain)+1))
    
    plt.title('PolyTVOfit BR, degree = {}'.format(degree))
    plt.scatter(df.resistanceBR, df.temperature, label = 'measurement')
    plt.plot(resBR, fit.BR(resBR), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    residualBR = df.temperature - fit.BR(df.resistanceBR)
    plt.title('Residual Polynomial TVO fit BR, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualBR*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorBR_RMSE = np.sqrt(np.mean(np.power(fit.BR(df.resistanceBR)-df.temperature,2)))
    print('The RMSE error for cernox BR sensor with a {} degree polynomial TVO fit is: {} mK'.format(degree, errorBR_RMSE*1000))

    
 
    resTL = np.arange(int(np.min(fit.TLDomain)), int(np.max(fit.TLDomain)+1))
    
    plt.title('PolyTVOfit TL, degree = {}'.format(degree))
    plt.scatter(df.resistanceTL, df.temperature, label = 'measurement')
    plt.plot(resTL, fit.TL(resTL), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    residualTL = df.temperature - fit.TL(df.resistanceTL)
    plt.title('Residual Polynomial TVO fit TL, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualTL*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorTL_RMSE = np.sqrt(np.mean(np.power(fit.TL(df.resistanceTL)-df.temperature,2)))
    print('The RMSE error for cernox TL sensor with a {} degree polynomial TVO fit is: {} mK'.format(degree, errorTL_RMSE*1000))

    
    resBL = np.arange(int(np.min(fit.BLDomain)), int(np.max(fit.BLDomain)+1))
    
    plt.title('PolyTVOfit BL, degree = {}'.format(degree))
    plt.scatter(df.resistanceBL, df.temperature, label = 'measurement')
    plt.plot(resBL, fit.BL(resBL), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    residualBL = df.temperature - fit.BL(df.resistanceBL)
    plt.title('Residual Polynomial TVO fit BL, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualBL*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorBL_RMSE = np.sqrt(np.mean(np.power(fit.BL(df.resistanceBL)-df.temperature,2)))
    print('The RMSE error for cernox BL sensor with a {} degree polynomial TVO fit is: {} mK'.format(degree, errorBL_RMSE*1000))

    
    resTR = np.arange(int(np.min(fit.TRDomain)), int(np.max(fit.TRDomain)+1))
    
    plt.title('Polynomial TVO fit TR, degree = {}'.format(degree))
    plt.scatter(df.resistanceTR, df.temperature, label = 'measurement')
    plt.plot(resTR, fit.TR(resTR), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    residualTR = df.temperature - fit.TR(df.resistanceTR)
    plt.title('Residual Polynomial TVO fit TR, degree = {}'.format(degree))
    plt.scatter(df.temperature, residualTR*1000)
    plt.ylabel('Residual [mK]')
    plt.xlabel('Temperature [K]')
    plt.grid()
    plt.xlim([1.7,10])
    plt.show()
    
    errorTR_RMSE = np.sqrt(np.mean(np.power(fit.TR(df.resistanceTR)-df.temperature,2)))
    print('The RMSE error for cernox TR sensor with a {} degree polynomial TVO fit is: {} mK'.format(degree, errorTR_RMSE*1000))



def plotPower(folder, file, RT = True):
    df = SensorData(folder, file, False) 
    fit = PowerFit(df)
    
    if RT == True: 
        resPot = np.arange(np.min(df.resistancePot), np.max(df.resistancePot), 1)
    else: 
        resPot = np.arange(1400, np.max(df.resistancePot), 1)
     
    plt.title('Powerfit Pot')
    plt.scatter(df.resistancePot, df.temperature, label = 'measurement')
    plt.plot(resPot, fit.Pot(resPot), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    if RT == True: 
        resBR = np.arange(np.min(df.resistanceBR), np.max(df.resistanceBR), 1)
    else: 
        resBR = np.arange(1600,  np.max(df.resistanceBR), 1)
     
    plt.title('Powerfit BR')
    plt.scatter(df.resistanceBR, df.temperature, label = 'measurement')
    plt.plot(resBR, fit.BR(resBR), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    if RT == True: 
        resTL = np.arange(np.min(df.resistanceTL), np.max(df.resistanceTL), 1)
    else: 
        resTL = np.arange(1500, np.max(df.resistanceTL), 1)
    
    plt.title('Powerfit TL')
    plt.scatter(df.resistanceTL, df.temperature, label = 'measurement')
    plt.plot(resTL, fit.TL(resTL), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    if RT == True: 
        resBL = np.arange(np.min(df.resistanceBL), np.max(df.resistanceBL), 1)
    else: 
        resBL = np.arange(2000, np.max(df.resistanceBL), 1)
    
    plt.title('Powerfit BL')
    plt.scatter(df.resistanceBL, df.temperature, label = 'measurement')
    plt.plot(resBL, fit.BL(resBL), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    if RT == True: 
        resTR = np.arange(np.min(df.resistanceTR), np.max(df.resistanceTR), 1)
    else: 
        resTR = np.arange(4000, np.max(df.resistanceTR), 1)
     
    plt.title('Powerfit TR')
    plt.scatter(df.resistanceTR, df.temperature, label = 'measurement')
    plt.plot(resTR, fit.TR(resTR), label = 'fit')
    plt.ylabel('Temperature [K]')
    plt.xlabel('Resistance [Ohm]')
    plt.legend()
    plt.grid()
    plt.show()


# threePoint = ThreePointCalibration('TL')
# threePointNew = ThreePointCalibration(('TLnew'))

# # print(threePoint.getResistance(10))
    

    
# # CalibrationLoop()

# plotCalibrationPoints(r'Analysed_Calibration_test.csv', True)



# # transition()


# plotPower(folder, r'Analysed_Calibration_test.csv')

# degreeCheby = 5
# plotChebychev(folder, r'Calibration77K.csv', degreeCheby)
# # checkOffset('cheby', degreeCheby)

# degree = 2
# plotPolyTVO(folder, r'Calibration295K.csv', degree)
# checkOffset('polyTVO', degree)

# degree = 3
# plotPolyCernox(folder, r'Calibration77K.csv', degree)
# checkOffset()


# plotFit3pointCalib(folder, r'Calibration77K.csv', degree)


# File = RawSensorData(r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements', r'Noise_41_41mbar_RH0_0mA_SH0_0mA_Coil0_0V_0_0Hz_0_0Ohm__20200806_105034.csv')
# File.plots()
# print('R_pot = ', File.meanResPot)
# print('R_BR = ', File.meanResBR)
# print('R_TL = ', File.meanResTL)
# print('R_BL = ', File.meanResBL)
# print('R_TR = ', File.meanResTR)

# print('std_pot = ', File.stdResPot)
# print('std_BR = ', File.stdResBR)
# print('std_TL = ', File.stdResTL)
# print('std_BL = ', File.stdResBL)
# print('std_TR = ', File.stdResTR)
        
# degree = 10
# sensorData_SF = SensorData(folder, r'CalibrationSF.csv', False)
# fitSF = Cheby(sensorData_SF, degree)
# plotChebychev(folder, r'CalibrationSF.csv', degree, False)