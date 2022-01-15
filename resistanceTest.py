# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:13:27 2020

@author: lmurberg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


folder = r'G:\Workspaces\h\HeliumIIHeatTransferRutherford\6 - Measurements and Results\2020-07-08 D11T-CoilGE02\Measurements\Analysed files'
df = pd.read_csv('{}\{}'.format(folder, r'CircuitAnalysis.csv'))


R_readout = df.loc[(df['cardLimitReached'] == False) & (df['relErrorR_readout'] < 0.1), 'R_readout']
T_ss_ol = df.loc[(df['cardLimitReached'] == False) & (df['relErrorR_readout'] < 0.1), 'T_ss_ol']
fit = np.polynomial.polynomial.Polynomial.fit(T_ss_ol, R_readout, 1)

rmse = np.sqrt(np.mean(np.power(R_readout - fit(T_ss_ol), 2)))

T_ss_ol_test = np.arange(1.8,15,0.1)

print('Coefficients: ', fit.convert().coef)
        
print('Error: ', rmse, r'Ohm')
        
plt.plot(T_ss_ol_test, fit(T_ss_ol_test), label ='fit: R [Ohm] = {:.4f} + {:.4f} Tss_ol [K]'.format(fit.convert().coef[0],fit.convert().coef[1]))
plt.scatter(T_ss_ol, R_readout)
plt.xlabel('Outer layer ss temperature [K]')
plt.ylabel('Resistance of heater [$\Omega$]')
plt.grid()
plt.legend()
plt.show()


# R_readout = np.mean(df.loc[(df['cardLimitReached'] == False) & (df['relErrorR_readout']<0.1), 'R_readout'])


# R_readout_std = np.std(df.loc[(df['cardLimitReached'] == False) & (df['relErrorR_readout']<0.1), 'R_readout'])
# R_readout_SEM = R_readout_std/np.sqrt(len(df.loc[(df['cardLimitReached'] == False) & (df['relErrorR_readout']<0.1), 'R_readout']))
# R_readoutError = np.mean(df.loc[(df['cardLimitReached'] == False) & (df['relErrorR_readout']<0.1), 'relErrorR_readout'])

# R_readout_error = R_readoutError + R_readout_SEM

df_18K = df.groupby('filePressure').get_group(16.38)
df_19K = df.groupby('filePressure').get_group(22.99)
df_21K = df.groupby('filePressure').get_group(41.41)
df_20K = df.groupby('filePressure').get_group(31.29)


R_readout_18K = df_18K.loc[(df_18K['cardLimitReached'] == False), 'R_readout']
T_ss_ol_18K = df_18K.loc[(df_18K['cardLimitReached'] == False), 'T_ss_ol']
R_error_18K = df_18K.loc[(df_18K['cardLimitReached'] == False), 'relErrorR_readout']


R_readout_19K = df_19K.loc[(df_19K['cardLimitReached'] == False), 'R_readout']
T_ss_ol_19K = df_19K.loc[(df_19K['cardLimitReached'] == False), 'T_ss_ol']
R_error_19K = df_19K.loc[(df_19K['cardLimitReached'] == False), 'relErrorR_readout']


R_readout_20K = df_20K.loc[(df_20K['cardLimitReached'] == False), 'R_readout']
T_ss_ol_20K = df_20K.loc[(df_20K['cardLimitReached'] == False), 'T_ss_ol']
R_error_20K = df_20K.loc[(df_20K['cardLimitReached'] == False), 'relErrorR_readout']


R_readout_21K = df_21K.loc[(df_21K['cardLimitReached'] == False), 'R_readout']
T_ss_ol_21K = df_21K.loc[(df_21K['cardLimitReached'] == False), 'T_ss_ol']
R_error_21K = df_21K.loc[(df_21K['cardLimitReached'] == False), 'relErrorR_readout']



plt.errorbar(T_ss_ol_18K, R_readout_18K, yerr = R_error_18K, fmt ='o', label = '1.8 K', c = 'y')
plt.errorbar(T_ss_ol_19K, R_readout_19K, yerr = R_error_19K, fmt ='o', label = '1.9 K', c = 'r')
plt.errorbar(T_ss_ol_20K, R_readout_20K, yerr = R_error_20K, fmt ='o', label = '2.0 K', c = 'g')
plt.errorbar(T_ss_ol_21K, R_readout_21K, yerr = R_error_21K, fmt ='o', label = '2.1 K', c = 'b')

plt.title('Resistance +/- rel_error as function of outer layer temperature')
plt.grid()
plt.xlabel('Outer layer ss temperature [K]')
plt.ylabel('Resistance of heater [$\Omega$]')
plt.legend()
plt.show()

plt.errorbar(T_ss_ol_18K, R_readout_18K, yerr = R_error_18K*R_readout_18K, fmt ='o', label = '1.8 K', c = 'y')
plt.errorbar(T_ss_ol_19K, R_readout_19K, yerr = R_error_19K*R_readout_19K, fmt ='o', label = '1.9 K', c = 'r')
plt.errorbar(T_ss_ol_20K, R_readout_20K, yerr = R_error_20K*R_readout_20K, fmt ='o', label = '2.0 K', c = 'g')
plt.errorbar(T_ss_ol_21K, R_readout_21K, yerr = R_error_21K*R_readout_21K, fmt ='o', label = '2.1 K', c = 'b')

plt.title('Resistance +/- error as function of outer layer temperature')
plt.grid()
plt.xlabel('Outer layer ss temperature [K]')
plt.ylabel('Resistance of heater [$\Omega$]')
plt.legend()
plt.show()



R_readout_18K = df_18K.loc[(df_18K['cardLimitReached'] == False) & (df_18K['relErrorR_readout'] < 0.1), 'R_readout']
T_ss_ol_18K = df_18K.loc[(df_18K['cardLimitReached'] == False) & (df_18K['relErrorR_readout'] < 0.1), 'T_ss_ol']
R_error_18K = df_18K.loc[(df_18K['cardLimitReached'] == False) & (df_18K['relErrorR_readout'] < 0.1), 'relErrorR_readout']


R_readout_19K = df_19K.loc[(df_19K['cardLimitReached'] == False) & (df_19K['relErrorR_readout'] < 0.1), 'R_readout']
T_ss_ol_19K = df_19K.loc[(df_19K['cardLimitReached'] == False) & (df_19K['relErrorR_readout'] < 0.1), 'T_ss_ol']
R_error_19K = df_19K.loc[(df_19K['cardLimitReached'] == False) & (df_19K['relErrorR_readout'] < 0.1), 'relErrorR_readout']


R_readout_20K = df_20K.loc[(df_20K['cardLimitReached'] == False) & (df_20K['relErrorR_readout'] < 0.1), 'R_readout']
T_ss_ol_20K = df_20K.loc[(df_20K['cardLimitReached'] == False) & (df_20K['relErrorR_readout'] < 0.1), 'T_ss_ol']
R_error_20K = df_20K.loc[(df_20K['cardLimitReached'] == False) & (df_20K['relErrorR_readout'] < 0.1), 'relErrorR_readout']


R_readout_21K = df_21K.loc[(df_21K['cardLimitReached'] == False) & (df_21K['relErrorR_readout'] < 0.1), 'R_readout']
T_ss_ol_21K = df_21K.loc[(df_21K['cardLimitReached'] == False) & (df_21K['relErrorR_readout'] < 0.1), 'T_ss_ol']
R_error_21K = df_21K.loc[(df_21K['cardLimitReached'] == False) & (df_21K['relErrorR_readout'] < 0.1), 'relErrorR_readout']



plt.errorbar(T_ss_ol_18K, R_readout_18K, yerr = R_error_18K, fmt ='o', label = '1.8 K', c = 'y')
plt.errorbar(T_ss_ol_19K, R_readout_19K, yerr = R_error_19K, fmt ='o', label = '1.9 K', c = 'r')
plt.errorbar(T_ss_ol_20K, R_readout_20K, yerr = R_error_20K, fmt ='o', label = '2.0 K', c = 'g')
plt.errorbar(T_ss_ol_21K, R_readout_21K, yerr = R_error_21K, fmt ='o', label = '2.1 K', c = 'b')

plt.title('Resistance +/- rel_error as function of outer layer temperature where rel_error < 0.1')
plt.grid()
plt.xlabel('Outer layer ss temperature [K]')
plt.ylabel('Resistance of heater [$\Omega$]')
plt.legend()
plt.show()

plt.errorbar(T_ss_ol_18K, R_readout_18K, yerr = R_error_18K*R_readout_18K, fmt ='o', label = '1.8 K', c = 'y')
plt.errorbar(T_ss_ol_19K, R_readout_19K, yerr = R_error_19K*R_readout_19K, fmt ='o', label = '1.9 K', c = 'r')
plt.errorbar(T_ss_ol_20K, R_readout_20K, yerr = R_error_20K*R_readout_20K, fmt ='o', label = '2.0 K', c = 'g')
plt.errorbar(T_ss_ol_21K, R_readout_21K, yerr = R_error_21K*R_readout_21K, fmt ='o', label = '2.1 K', c = 'b')

plt.title('Resistance +/- error as function of outer layer temperature where rel_error < 0.1')
plt.grid()
plt.xlabel('Outer layer ss temperature [K]')
plt.ylabel('Resistance of heater [$\Omega$]')
plt.legend()
plt.show()