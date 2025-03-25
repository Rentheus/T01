#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:09:01 2025

@author: axel
"""
 
import numpy as np
import matplotlib.pyplot as plt
import iminuit
from iminuit import cost

plt.rcParams["figure.figsize"] = (10,7)

#%% ionisationskammer
ion_1 = np.genfromtxt("T01\\T01\\Ionisationskammer_1.csv", delimiter = ",")
ion_1[:,0] = ion_1[0,0] - ion_1[:,0]
ion_1[:,0] /= 10
ion_2 = np.genfromtxt("T01\\T01\\Ionisationskammer_2.csv", delimiter = ",")
ion_2[:,0] = ion_2[0,0] - ion_2[:,0]
ion_2[:,0] /= 10
plt.errorbar(ion_1[:,0], ion_1[:,1], ion_1[:,2], label = "Messreihe 1", fmt = "o")
plt.errorbar(ion_2[:,0], ion_2[:,1], ion_2[:,2], label = "Messreihe 2", fmt = "o")


plt.ylabel("$U$ [$nA$]")
plt.xlabel("$d-d_0$ [$mm$]")
plt.title("Ionisationskammer")
plt.legend()
plt.show()

#%% beta-absorber

beta_noise = np.genfromtxt("T01\\T01\\beta_noise.csv", delimiter = ",")
beta = np.genfromtxt("T01\\T01\\beta_abschirmung.csv", delimiter = ",", dtype=(float, int))

beta[:,1] - np.mean(beta_noise)
abschirmungen = {
    0:0,
    5:6.84,
    13:135,
    23:933,
    12:100,
    9:28.2,
    22:754,
    6:10.7,
    17:332,
    18:375,
    11:66.8,
    15:224,
    20:507,
    } # area_mass density mg/cm^2
for i in range(beta.shape[0]):
    beta[i,0] = abschirmungen[int(beta[i,0])]
    
beta_comb = np.zeros((beta.shape[0]//5,3))
for i in range(beta.shape[0]//5):
    beta_comb[i,0] = beta[i*5,0]
    beta_comb[i,1] = np.mean(beta[i*5:(i+1)*5,1])
    beta_comb[i,2] = np.std(beta[i*5:(i+1)*5,1], ddof= 1)
    

def beta_absorb(x, a1,b1 ,c ):
    return abs(a1)*np.exp(-x/b1)+c

print(beta_comb)
chi2 = cost.LeastSquares(beta_comb[:,0], beta_comb[:,1], beta_comb[:,2], beta_absorb)

start_val = (15000, 1.36, 10000)

m_beta = iminuit.Minuit(chi2, *start_val)

print(m_beta.migrad())
#m_beta.visualize()

#plt.show()
   
#print(beta_comb)

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,9), layout = "tight",gridspec_kw={'height_ratios': [5, 2]}, sharex=True)


ax[0].errorbar(beta_comb[:,0], beta_comb[:,1], beta_comb[:,2], fmt = ".", label = "average counts per 10s")
#plt.scatter(beta[:,0], beta[:,1])

fity = beta_absorb( beta_comb[:,0], m_beta.values["a1"],m_beta.values["b1"],m_beta.values["c"])
x = np.linspace(0,950)
fity2 = beta_absorb( x, m_beta.values["a1"],m_beta.values["b1"], m_beta.values["c"])
ax[0].plot(x,fity2, label = "fit")

ax[0].set_ylabel("counts")
#ax[0].set_yscale("log")
ax[0].legend(fontsize = 13)

ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
ax[1].errorbar(beta_comb[:,0], beta_comb[:,1]-fity, beta_comb[:,2], fmt = ".", label = "residuals")
ax[1].set_ylabel('$counts - counts_{fit}$ ')
ax[1].set_xlabel('$d_{Alu}$ [$mm$] ')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)


fig.text(0.5,0, f'I_0  = ({m_beta.values["a1"]:.2f} +- {m_beta.errors["a1"]:0.2f}) , b = ({m_beta.values["b1"]:.3f} +- {m_beta.errors["b1"]:.3f}) 1/mm,  chi2/dof  = {m_beta.fval:.1f} ', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)

ax[0].title.set_text("$beta absorber, Sr-90")
plt.show()

#TODO probelm one abstimmung ein ereigniss alle 4.3 \mu s ==>totzeit? haben wir szintillator oder gm verwendet