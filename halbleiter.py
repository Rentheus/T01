#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### FP Physik
#
# Tobias Sommer, 445306
# Axel Andrée, 422821

import numpy as np
import matplotlib.pyplot as plt
import iminuit
from iminuit import cost
import scipy
from praktikum.analyse import lineare_regression_xy
plt.rcParams["figure.figsize"] = (13,8)

#%% pn-sperrdiode

alpha = np.genfromtxt("axto-t1\\axto-t1\\axto-t1-halbleiter\\5 peaks_sehr nah.TKA")
print(alpha[0:2])
alpha = alpha[2:]

channels = np.arange(0, len(alpha), 1)

plt.errorbar(channels, alpha, np.sqrt(alpha), fmt = ".", label = "Am-241 measurements")
plt.annotate("Po-214 peak", xy = (12500, 500), xytext = (14000, 550), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))
plt.annotate("Po-218 peak", xy = (7650, 210), xytext = (7000, 75), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))
plt.annotate("Rn-222 & Po-210 peaks", xy = (5500, 330), xytext = (5500, 25), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))
plt.annotate("Ra-226 peak", xy = (2600, 130), xytext = (2400, 45), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))

plt.legend()
plt.title("Am-241 spectrum at minimal distance, $t = 240s$")
plt.ylabel("counts")
plt.xlabel("channel")
plt.savefig("alpha-spectrum.pdf")
plt.show()

#%% distanzen:
def load_spectrum(file):
    return np.genfromtxt(file)[2:]
    

p5 = alpha

p4 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\4 peaks.TKA")
p4plus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\4 peaks+.TKA")
p4minus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\4 peaks-.TKA")

p3 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\3 peaks.TKA")
p3plus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\3 peaks+.TKA")
p3minus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\3 peaks-.TKA")

p2 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\2 peaks.TKA")
p2plus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\2 peaks+.TKA")
p2minus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\2 peaks-.TKA")

p1 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\1 peaks.TKA")
p1plus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\1 peaks+.TKA")
p1minus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\1 peaks-.TKA")

p0 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\0 peaks.TKA")
p0plus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\0 peaks+.TKA")
p0minus = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\0 peaks-.TKA")

plist = [p4,p3,p2,p1,p0]
ppluslist = [p4plus,p3plus,p2plus,p1plus,p0plus]
pminuslist = [p4minus,p3minus,p2minus,p1minus,p0minus]
#  5, 4, 4+, 4-, 3 ....
dists = np.array([34.88, 35.16, 35.26, 35.06, 35.79, 35.89, 35.69, 36.05, 36.15, 35.95, 36.82, 36.92, 36.72, 38.91, 39.01, 38.81])
    
fig, ax = fig, ax = plt.subplots(6, 1, figsize=(12,12), layout = "tight")

ax[0].errorbar(channels, p5, np.sqrt(p5), label = "measurement minimal distance", fmt = ".", errorevery = 5, ms = 1, elinewidth = .5)
ax[0].set_ylabel("counts")
ax[0].set_xlabel("channels")
ax[0].title.set_text("Am-241, all peaks, x = " + str(dists[0]) + " cm")


for i in range(5):
    ax[i+1].errorbar(channels, plist[i], np.sqrt(plist[i]), label = f"measurement, x = {dists[i*3 +1]} cm", fmt = ".", zorder = 4, errorevery = 5, ms = 1, elinewidth = .5, )
    ax[i+1].errorbar(channels, ppluslist[i], np.sqrt(ppluslist[i]), label = "1mm after", fmt = "x", zorder = 2, errorevery = (2, 5), ms = 1, elinewidth = .5)
    ax[i+1].errorbar(channels, pminuslist[i], np.sqrt(pminuslist[i]), label = "1mm before", fmt = "2", zorder = 3, errorevery = (4,5), ms = 1, elinewidth = .5)

    
    ax[i+1].set_ylabel("counts")
    ax[i+1].set_xlabel("channels")
    ax[i+1].title.set_text("Am-241, "+ str(4-i) +" peaks, x = " + str(dists[i*3 +1]) + " cm")
    ax[i+1].legend()
    
ax[5].set_yscale("log")
plt.savefig("disappearing_peaks_alpha.pdf")
plt.show()
#%%
#reichweitenplot
energy = [4.78, 5.3, 5.49, 6 ,7.69] # MeV
reichweiten = np.array([35.16, 35.79, 36.05, 36.82, 38.91]) - 34.88

reichweiten_lit = np.array([4.083E-03, 4.770E-03, 5.034E-03, 5.772E-03, 8.540E-03, ]) #g/cm^2, CSDA for dry air, sea level,  https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html TODO zitieren
air_density =  1.204 / 1000 # https://en.wikipedia.org/wiki/Density_of_air
#rw_umgerechnet = air_density * reichweiten

reichw_lit_umgerechnet = reichweiten_lit/air_density
print(reichw_lit_umgerechnet - reichweiten)

plt.scatter(energy, reichweiten, label = "range of alpha particles")
plt.scatter(energy, reichw_lit_umgerechnet, label = "expected range of alpha particles")

plt.title("range of alpha particles in our detector")
plt.ylabel("average energy [$MeV$]")
plt.ylabel("range [$cm$]")
plt.legend()
plt.show()

#einfach diff oder fit für detektorintrinsische stopping power? ==> diff weil reach nicht linear -> graph von nistseite


intrins_absorpt = np.mean(reichw_lit_umgerechnet - reichweiten)
intrins_absorpt_err = np.std(reichw_lit_umgerechnet - reichweiten, ddof = 1)/np.sqrt(5)

print(f"intr absortpion thickness: {intrins_absorpt:.2f}+-{intrins_absorpt_err:.2f} cm of air equiv")

#%% energie deposit in 300 mikrometer si https://physics.nist.gov/cgi-bin/Star/ap_table.pl GRAPH ZITIEREN

stopping_power = np.array([6.363E+02, 5.946E+02, 5.809E+02, 5.474E+02, 4.622E+02]) # MeV cm^2/g
density_si = 2.329085 #g/cm^3 https://en.wikipedia.org/wiki/Silicon
absorbed_energy = stopping_power * density_si * 0.03
print(f"rough estimation of maximum absorbed energy : {absorbed_energy} MeV, bigger than patricle energy: {energy} MeV => full absorption of alpha particle")

#energy loss in detector -> intrinsic absorption * stopping power air 

#for 7.69MeV
stopping_power_air_7690keV = 5.614E+02	#MeV cm^2/g
e_loss_7690keV = intrins_absorpt * air_density * stopping_power_air_7690keV 
e_loss_err_7690keV = intrins_absorpt_err * air_density * stopping_power_air_7690keV
print(f"energy loss by detector setup @7.69MeV: ({e_loss_7690keV:.3f} +- {e_loss_err_7690keV:.3f}) MeV)")
e_po214 = 7.69 - e_loss_7690keV
e_po214_err = e_loss_err_7690keV
print(f"measured energy for Po214: ({e_po214:.3f} +- {e_po214_err:.3f}) MeV)")

#for 6MeV
stopping_power_air_6MeV = 6.700E+02 #Mev cm^2/g
e_loss_6MeV = intrins_absorpt * air_density * stopping_power_air_6MeV 
e_loss_err_6MeV = intrins_absorpt_err * air_density * stopping_power_air_6MeV
print(f"energy loss by detector setup @6MeV: ({e_loss_6MeV:.3f} +- {e_loss_err_6MeV:.3f}) MeV)")
e_po218 = 6 - e_loss_6MeV
e_po218_err = e_loss_err_6MeV
print(f"measured energy for Po214: ({e_po218:.3f} +- {e_po218_err:.3f}) MeV)")

#for 4.78MeV
stopping_power_air_4780keV = 7.853E+02 #Mev cm^2/g
e_loss_4780keV = intrins_absorpt * air_density * stopping_power_air_4780keV 
e_loss_err_4780keV = intrins_absorpt_err * air_density * stopping_power_air_4780keV
print(f"energy loss by detector setup @4780keV: ({e_loss_4780keV:.3f} +- {e_loss_err_4780keV:.3f}) MeV)")
e_ra226 = 4.78 - e_loss_4780keV
e_ra226_err = e_loss_err_4780keV
print(f"measured energy for Ra-226: ({e_ra226:.3f} +- {e_ra226_err:.3f}) MeV)")


#peaks bestimmen:

def gauss(x, mu, sigma, a, c):
    return  a*1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2) + c


c_po214 = cost.LeastSquares(channels[11900:12700], alpha[11900:12700], np.sqrt(alpha[11900:12700]), gauss)

m_po214 = iminuit.Minuit(c_po214, mu = 12000, sigma=10, a = 1, c= 0)

print(m_po214.migrad())

c_po218 = cost.LeastSquares(channels[7170:8900], alpha[7170:8900], np.sqrt(alpha[7170:8900]), gauss)

m_po218 = iminuit.Minuit(c_po218, mu = 7800, sigma=200, a = 1000, c= 1)

print(m_po218.migrad())

c_ra226 = cost.LeastSquares(channels[1700:3200], alpha[1700:3200], np.sqrt(alpha[1700:3200]), gauss)

m_ra226 = iminuit.Minuit(c_ra226, mu = 2700, sigma=200, a = 1000, c= 1)

print(m_ra226.migrad(ncall = 20000))


fig, ax = plt.subplots(2, 3, figsize=(16,7), layout = "tight",gridspec_kw={'height_ratios': [5, 2]})

fit_counts_214 = gauss(channels[11900:12700], m_po214.values["mu"], m_po214.values["sigma"], m_po214.values["a"], m_po214.values["c"])
fit_counts_218 = gauss(channels[7170:8900], m_po218.values["mu"], m_po218.values["sigma"], m_po218.values["a"], m_po218.values["c"])
fit_counts_226 = gauss(channels[1700:3200], m_ra226.values["mu"], m_ra226.values["sigma"], m_ra226.values["a"], m_ra226.values["c"])


ax[0,2].errorbar(channels[11900:12700], alpha[11900:12700], np.sqrt(alpha[11900:12700]), fmt = ".", label = "recorded Po-214-peak", zorder = 1)
ax[0,2].plot(channels[11900:12700],fit_counts_214, label = "fit")
ax[0,2].set_xlabel("channel")
ax[0,2].set_ylabel("counts")
ax[0,2].axvline(m_po214.values["mu"], label = "peak", zorder = 3, color = "black")
ax[0,2].legend()
ax[0,2].title.set_text("fit for peak location, Po-214")



ax[0,1].errorbar(channels[7170:8900], alpha[7170:8900], np.sqrt(alpha[7170:8900]), fmt = ".", label = "recorded Po-218-peak", zorder = 1)
ax[0,1].plot(channels[7170:8900],fit_counts_218, label = "fit")
ax[0,1].set_xlabel("channel")
ax[0,1].set_ylabel("counts")
ax[0,1].axvline(m_po218.values["mu"], label = "peak", zorder = 3, color = "black")
ax[0,1].legend()
ax[0,1].title.set_text("fit for peak location, Po-218")

ax[0,0].errorbar(channels[1700:3200], alpha[1700:3200], np.sqrt(alpha[1700:3200]), fmt = ".", label = "recorded Ra-226-peak", zorder = 1)
ax[0,0].plot(channels[1700:3200],fit_counts_226, label = "fit")
ax[0,0].set_xlabel("channel")
ax[0,0].set_ylabel("counts")
ax[0,0].axvline(m_ra226.values["mu"], label = "peak", zorder = 3, color = "black")
ax[0,0].legend()
ax[0,0].title.set_text("fit for peak location, Ra-226")




ax[1,2].errorbar(channels[11900:12700], alpha[11900:12700]- fit_counts_214, np.sqrt(alpha[11900:12700]), fmt = ".", label = "residuals", zorder = 1)
ax[1,2].axhline(0, color = "black", linestyle = "--",zorder = 2)
ymax = max([abs(x) for x in ax[1,2].get_ylim()])
ax[1,2].set_ylim(-ymax, ymax)
ax[1,2].set_ylabel("$counts - counts_{Fit}$")
ax[1,2].set_xlabel("channel")
ax[1,2].legend()
ax[1,2].title.set_text(f"peak at channel ({m_po214.values['mu']:.2f} +- {m_po214.errors['mu']:.2f}), $\chi^2/ndof$ = {m_po214.fval/m_po214.ndof :.1f}")


ax[1,1].errorbar(channels[7170:8900], alpha[7170:8900]- fit_counts_218, np.sqrt(alpha[7170:8900]), fmt = ".", label = "residuals", zorder = 1)
ax[1,1].axhline(0, color = "black", linestyle = "--",zorder = 2)
ymax = max([abs(x) for x in ax[1,1].get_ylim()])
ax[1,1].set_ylim(-ymax, ymax)
ax[1,1].set_ylabel("$counts - counts_{Fit}$")
ax[1,1].set_xlabel("channel")
ax[1,1].legend()
ax[1,1].title.set_text(f"peak at channel ({m_po218.values['mu']:.2f} +- {m_po218.errors['mu']:.2f}), $\chi^2/ndof$ = {m_po218.fval/m_po218.ndof :.1f}")


ax[1,0].errorbar(channels[1700:3200], alpha[1700:3200]- fit_counts_226, np.sqrt(alpha[1700:3200]), fmt = ".", label = "residuals", zorder = 1)
ax[1,0].axhline(0, color = "black", linestyle = "--",zorder = 2)
ymax = max([abs(x) for x in ax[1,0].get_ylim()])
ax[1,0].set_ylim(-ymax, ymax)
ax[1,0].set_ylabel("$counts - counts_{Fit}$")
ax[1,0].set_xlabel("channel")
ax[1,0].legend()
ax[1,0].title.set_text(f"peak at channel ({m_ra226.values['mu']:.2f} +- {m_ra226.errors['mu']:.2f}), $\chi^2/ndof$ = {m_ra226.fval/m_ra226.ndof :.1f}")


plt.savefig("fit_peaks_for_energyfit_closest.pdf")
plt.show()

###### Particle energy by reverse Lookup at ASTAR: 
print("gemessene reichweiten:")
print(reichweiten * air_density)
print("gemessene reichweiten + 0.5mm:")
print((reichweiten+0.05) * air_density)

ASTAR_energies = np.array([3.460E-01, 1.725E+00, 2.165E+00, 3.231E+00, 5.360E+00])
ASTAR_reach = np.array([3.372E-04, 1.096E-03, 1.409E-03, 2.336E-03, 4.853E-03])
ASTAR_energies_plus = np.array([0.458, 1.755E+00, 2.242E+00, 3.292E+00, 5.403])
ASTAR_reach_plus = np.array([3.973E-04, 1.116E-03, 1.468E-03, 2.396E-03	, 4.912E-03	])

ASTAR_energies_err = ASTAR_energies_plus - ASTAR_energies
print("ASTAR reichweiten: ")
print(ASTAR_reach)
print("ASTAR reichweiten bei +0.5 mm : ")
print(ASTAR_reach_plus)





def lin(x, a, b):
    return a*x+b

#a, ae, b, be, fval, cov  = lineare_regression_xy(np.array([m_ra226.values['mu'], m_po218.values['mu'], m_po214.values['mu']]), np.array([e_ra226, e_po218, e_po214]), np.array([m_ra226.errors['mu'], m_po218.errors['mu'], m_po214.errors['mu']]),  np.array([e_ra226_err, e_po218_err, e_po214_err]))
a, ae, b, be, fval, cov  = lineare_regression_xy(np.array([m_ra226.values['mu'], m_po218.values['mu'], m_po214.values['mu']]), ASTAR_energies[[True,False,False,True,True]], np.array([m_ra226.errors['mu'], m_po218.errors['mu'], m_po214.errors['mu']]), ASTAR_energies_err[[True,False,False,True,True]])

print(a, ae, b, be, fval, cov)

ener_channels = lin(channels, a, b)
ener_channels_err = np.sqrt((ae*channels)**2 + be**2)
# ener_channels_err =  lincalib(channels, m_po218.values["mu"], e_po218, m_po214.values["mu"], e_po214, m_po218.errors["mu"], e_po218_err, m_po214.errors["mu"], e_po214_err)

print(ener_channels_err)
plt.errorbar(ener_channels, alpha, np.sqrt(alpha), ener_channels_err, fmt = ".", label="measurement, calibrated energy")
plt.ylabel("counts")
plt.xlabel("$E_α$ [$MeV$]")
plt.title("spectrum at lowest distance, calibrated energy")
plt.legend()
plt.show()

#%% distance energy calibration

fig, ax = plt.subplots(2, 5, figsize=(23,7), layout = "tight",gridspec_kw={'height_ratios': [5, 2]})


ax[0,0].errorbar(channels[11900:12700], alpha[11900:12700], np.sqrt(alpha[11900:12700]), fmt = ".", label = "recorded Po-214-peak", zorder = 1)
ax[0,0].plot(channels[11900:12700],fit_counts_214, label = "fit")
ax[0,0].set_xlabel("channel")
ax[0,0].set_ylabel("counts")
ax[0,0].axvline(m_po214.values["mu"], label = "peak", zorder = 3, color = "black")
ax[0,0].legend()
ax[0,0].title.set_text("fit for peak location, Po-214, d = 0 cm")

ax[1,0].errorbar(channels[11900:12700], alpha[11900:12700]- fit_counts_214, np.sqrt(alpha[11900:12700]), fmt = ".", label = "residuals", zorder = 1)
ax[1,0].axhline(0, color = "black", linestyle = "--",zorder = 2)
ymax = max([abs(x) for x in ax[1,0].get_ylim()])
ax[1,0].set_ylim(-ymax, ymax)
ax[1,0].set_ylabel("$counts - counts_{Fit}$")
ax[1,0].set_xlabel("channel")
ax[1,0].legend()
ax[1,0].title.set_text(f"peak at channel ({m_po214.values['mu']:.2f} +- {m_po214.errors['mu']:.2f}), $\chi^2/ndof$ = {m_po214.fval/m_po214.ndof :.1f}")


upper = [12500, 12200, 12000, 11200, 9000]
lower = [11000, 10500, 10000, 7000, 2000]
mu_begin = [12000, 11500, 11000, 9500, 5000]
peak_channel = [m_po214.values["mu"]]
peak_channel_err = [m_po214.errors["mu"]]
sigmas_peak_channels = [m_po214.values["sigma"]]
sigmas_peak_channels_err = [m_po214.errors["sigma"]]

for i in range(4):
    f = plist[i][lower[i]:upper[i]] > 0
    c_peaks = cost.LeastSquares(channels[lower[i]:upper[i]][f],  plist[i][lower[i]:upper[i]][f], np.sqrt(plist[i][lower[i]:upper[i]][f]), gauss)

    m_peaks = iminuit.Minuit(c_peaks, mu = mu_begin[i], sigma=100, a = 1, c= 0)

    print(m_peaks.migrad())
    fit_counts_peaks = gauss(channels[lower[i]:upper[i]], m_peaks.values["mu"], m_peaks.values["sigma"], m_peaks.values["a"], m_peaks.values["c"])
    ax[0,i+1].errorbar(channels[lower[i]:upper[i]], plist[i][lower[i]:upper[i]], np.sqrt(plist[i][lower[i]:upper[i]]), fmt = ".", label = "recorded Po-214-peak", zorder = 1, alpha = 1)
    ax[0,i+1].plot(channels[lower[i]:upper[i]], fit_counts_peaks, label = "fit")
    ax[0,i+1].set_xlabel("channel")
    ax[0,i+1].set_ylabel("counts")
    ax[0,i+1].axvline(m_peaks.values["mu"], label = "peak", zorder = 3, color = "black")
    ax[0,i+1].legend()
    ax[0,i+1].title.set_text(f"fit for peak location, Po-214, d = {reichweiten[i]:.1f} cm")
    
    peak_channel.append(m_peaks.values["mu"])
    peak_channel_err.append(m_peaks.errors["mu"])
    sigmas_peak_channels.append(m_peaks.values["sigma"])
    sigmas_peak_channels_err.append(m_peaks.errors["sigma"])

    
    ax[1,i+1].errorbar(channels[lower[i]:upper[i]],  plist[i][lower[i]:upper[i]]- fit_counts_peaks, np.sqrt( plist[i][lower[i]:upper[i]]), fmt = ".", label = "residuals", zorder = 1)
    ax[1,i+1].axhline(0, color = "black", linestyle = "--",zorder = 2)
    ymax = max([abs(x) for x in ax[1,i+1].get_ylim()])
    ax[1,i+1].set_ylim(-ymax, ymax)
    ax[1,i+1].set_ylabel("$counts - counts_{Fit}$")
    ax[1,i+1].set_xlabel("channel")
    ax[1,i+1].legend()
    ax[1,i+1].title.set_text(f"peak at channel ({m_peaks.values['mu']:.2f} +- {m_peaks.errors['mu']:.2f}), $\chi^2/ndof$ = {m_peaks.fval/m_peaks.ndof :.1f}")

plt.savefig("fits_for_peaks_dist.pdf")    
plt.show()
peak_channel = np.array(peak_channel)
peak_channel_err = np.array(peak_channel_err)
sigmas_peak_channels = np.array(sigmas_peak_channels)
#%%
reichweiten_remaining_temp = reichweiten[-1]-reichweiten
rw_rem = np.array([reichweiten[-1], reichweiten_remaining_temp[0], reichweiten_remaining_temp[1], reichweiten_remaining_temp[2], reichweiten_remaining_temp[3]])

total_absorpt_layer_thickness = intrins_absorpt + reichweiten[-1] - rw_rem
tot_abs_lay_thi_err = np.sqrt(intrins_absorpt_err**2 + 2* (0.01/12**0.57)**2)
print(tot_abs_lay_thi_err)
plt.errorbar(total_absorpt_layer_thickness, peak_channel, peak_channel_err, fmt = "o")
plt.ylabel("peak position [channel]")
plt.xlabel("total absorption layer thickness [cm of air equivalent]")
plt.title("peak position vs total absorption layer thickness")
plt.savefig("total_absorption_layer_thickness.pdf")
plt.show()

plt.errorbar(rw_rem, peak_channel, peak_channel_err, fmt = "o")
plt.ylabel("peak position [channel]")
plt.xlabel("remaining particle reach from experiment [cm]")
plt.title("peak position vs remaining particle reach using maximum measured reach")
plt.savefig("remaining_particle_reach.pdf")
plt.show()

#print(rw_rem * air_density)

#stopping power air  = (E0- Enow)/(rw_rem*air_density)
ASTAR_part_energies = np.array([5.360, 5.111, 4.518, 4.257, 3.411])
ASTAR_part_energies_plus = np.array([5.408, 5.162, 4.574, 4.316, 3.479])
ASTAR_part_energies_err = ASTAR_part_energies_plus - ASTAR_part_energies

all_energies = np.array([7.69, *ASTAR_part_energies])
all_energies_err = np.array([0, *ASTAR_part_energies_err])
energy_diffs = all_energies[:-1] - all_energies[1:] 
energy_diffs_err = np.sqrt(all_energies[:-1]**2 + all_energies[1:]**2) 

diffs_tot_abs_lay_thi = total_absorpt_layer_thickness[1:] - total_absorpt_layer_thickness[0:-1]
diffs_tot_abs_lay_thi = np.array([intrins_absorpt, *diffs_tot_abs_lay_thi])



stopping_power_air = (energy_diffs)/((diffs_tot_abs_lay_thi)*air_density)
sto_pow_air_err =np.sqrt(((energy_diffs)/(total_absorpt_layer_thickness**2*air_density) * np.sqrt(2) *tot_abs_lay_thi_err)**2 + (1/((total_absorpt_layer_thickness)*air_density) * energy_diffs_err )) 
#print(stopping_power_air)
#print(sto_pow_air_err)

#plt.errorbar()

ASTAR_stopping_power_air = [7.251E+02, 7.496E+02, 8.163E+02, 8.501E+02, 9.860E+02]
#TODO stoppig power von air

for i in range(5):
    print(f"{stopping_power_air[i]:.2f} +- {sto_pow_air_err[i]:.2f}")

#energy calib
a2, a2e, b2, b2e, fval2, cov2  = lineare_regression_xy(peak_channel, ASTAR_part_energies, peak_channel_err,  ASTAR_part_energies_err)

print(a,ae,b,be)
print(a2,a2e,b2,b2e,fval2,cov2)


x_p_channel= np.linspace(2000, 13000)
plt.errorbar(np.array([m_ra226.values['mu'], m_po218.values['mu'], m_po214.values['mu']]), ASTAR_energies[[True,False,False,True,True]],  ASTAR_energies_err[[True,False,False,True,True]], np.array([m_ra226.errors['mu'], m_po218.errors['mu'], m_po214.errors['mu']]), fmt= "o", label = "calibration via different peaks", color = "red", ms = 7)
plt.errorbar(peak_channel, ASTAR_part_energies, np.array([0.001,0.001,0.001,0.001,0.001,]), peak_channel_err, label = "Calibration via distance variation", color = "blue", fmt = "X")
plt.plot(x_p_channel, a2*x_p_channel + b2, color = "dodgerblue", label = "fit, distance variation")
plt.plot(x_p_channel, a*x_p_channel + b, color = "indianred", label = "fit, different peaks")
plt.legend()
plt.xlabel("channel")
plt.ylabel("$E_α$ [$MeV$]")
plt.title("comparison energy fits")
plt.savefig("comp_energy_fits.pdf")
plt.show()

#TODO die calibs matchen nicht

#%%1.5 range 
#messung bei min_dist 240s, bei anderen 25s --> skalierung
integral_under_peak = [np.sum(alpha[int(peak_channel[0]-2*sigmas_peak_channels[0]):int(peak_channel[0]+2*sigmas_peak_channels[0])]) * 25/240]
for i in range(4):
    integral_under_peak.append(np.sum(plist[i][int(peak_channel[i+1]-2*sigmas_peak_channels[i+1]):int(peak_channel[i+1]+2*sigmas_peak_channels[i+1])]))
integral_under_peak.append(np.sum(plist[4][2000:]))

print(integral_under_peak)

#TODO errors 
tot_abs_lay_thi = np.array([*total_absorpt_layer_thickness, 7.04821927])
plt.scatter(tot_abs_lay_thi, integral_under_peak)
plt.show()
#TODO letzten wert noch einfügen

#%% besser 
def find_peak(channels, spectrum, start_val, vis = False):
    f = spectrum > 0
    c_fkt=cost.LeastSquares(channels[f], spectrum[f], np.sqrt(spectrum[f]), gauss)
    m_fkt = iminuit.Minuit(c_fkt, mu = start_val, sigma=100, a = 1, c= 0)
    m_fkt.migrad()
    print(f"chi2/ndof = {m_fkt.fval / m_fkt.ndof:.2f}")
    print(f" min is valid : {m_fkt.valid}")
    if vis == True:
        m_fkt.visualize()
        plt.show()
    return m_fkt.values["mu"], m_fkt.errors["mu"], m_fkt.values["sigma"], m_fkt.errors["sigma"]
    
integral_under_peak = [np.sum(alpha[int(peak_channel[0]-3*sigmas_peak_channels[0]):int(peak_channel[0]+3*sigmas_peak_channels[0])]) * 25/240]


for i in range(4):
    l = int(peak_channel[i+1]-4*sigmas_peak_channels[i+1])
    u = int(peak_channel[i+1]+4*sigmas_peak_channels[i+1])
    peak_min, peak_min_err, sigm, sigm_err = find_peak(channels[l:u], pminuslist[i][l:u], peak_channel[i+1])
    integral_under_peak.append(np.sum(pminuslist[i][int(peak_min-4*sigm):int(peak_min+4*sigm)]))
    
    
    integral_under_peak.append(np.sum(plist[i][int(peak_channel[i+1]-4*sigmas_peak_channels[i+1]):int(peak_channel[i+1]+4*sigmas_peak_channels[i+1])]))
    
    peak_plus, peak_plus_err, sigm, sigm_err = find_peak(channels[l:u], ppluslist[i][l:u], peak_channel[i+1])
    integral_under_peak.append(np.sum(ppluslist[i][int(peak_plus-4*sigm):int(peak_plus+4*sigm)]))

#integral_under_peak.append(np.sum(pminuslist[4][3000:]))
#integral_under_peak.append(np.sum(plist[4][3000:]))
#integral_under_peak.append(np.sum(ppluslist[4][3000:]))

integral_under_peak = np.array(integral_under_peak) 
integral_under_peak_err = np.sqrt(integral_under_peak)
#da err  = sqrt( sum sqrt(val) **2) ==> err = sqrt(sum(val))

#print(integral_under_peak)

#TODO errors 
tot_abs_lay_thi = [total_absorpt_layer_thickness[0]]
for i in range(4):
    tot_abs_lay_thi.append(total_absorpt_layer_thickness[i+1]-0.1)
    tot_abs_lay_thi.append(total_absorpt_layer_thickness[i+1])
    tot_abs_lay_thi.append(total_absorpt_layer_thickness[i+1]+0.1)

#tot_abs_lay_thi.append(7.04821927 - 0.1)
#tot_abs_lay_thi.append(7.04821927 )
#tot_abs_lay_thi.append(7.04821927 + 0.1)

tot_abs_lay_thi = np.array(tot_abs_lay_thi)

def integr_reach_model(x, a1, b1, c1,): #1 -integrated gaussian , 1-erf()
    return a1*scipy.special.erfc(b1*(x-c1)) 

c_erf = cost.LeastSquares(tot_abs_lay_thi, integral_under_peak, integral_under_peak_err, integr_reach_model)
m_erf = iminuit.Minuit(c_erf, a1 = 20000, b1 = 15, c1 = 4) 
print(m_erf.migrad())
#m_erf.visualize()
x_abs_lay_thick = np.linspace(0, 8)
x_abs_lay_linfit = np.linspace(2.5, 6)

#steigung von erf:
steigung_erf = - m_erf.values["a1"] * m_erf.values["b1"] * 2/np.sqrt(np.pi) * np.exp(-(0)**2)
steigung_erf_err = np.sqrt( ( m_erf.values["b1"] * 2/np.sqrt(np.pi) * m_erf.errors["a1"])**2 + ( m_erf.values["a1"] * 2/np.sqrt(np.pi) * m_erf.errors["b1"])**2 )

lin_approx = lin(x_abs_lay_linfit, steigung_erf, -steigung_erf * m_erf.values["c1"] + m_erf.values["a1"])

R_max_hl = -(-steigung_erf * m_erf.values["c1"] + m_erf.values["a1"])/steigung_erf
R_max_hl_err = np.sqrt( (m_erf.values["a1"]/steigung_erf**2 * steigung_erf_err)**2 + (m_erf.errors["c1"])**2 + (1/steigung_erf * m_erf.errors["a1"])**2)

print(R_max_hl)
print(R_max_hl_err)


fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})
ax[0].plot(x_abs_lay_thick, integr_reach_model(x_abs_lay_thick, m_erf.values["a1"], m_erf.values["b1"], m_erf.values["c1"]))
ax[0].plot(x_abs_lay_linfit, lin_approx, label = "tangent for $R_{ex}$", color = "black", ls = "--")
ax[0].axhline(0, color = "silver", ls = "--")
ax[0].scatter(R_max_hl, 0, color = "navy", label = "$R_{ex}$ = " + f"({R_max_hl:.2f} +- {R_max_hl_err:.2f}) cm of air equivalent")

ax[0].errorbar(tot_abs_lay_thi, integral_under_peak, integral_under_peak_err, fmt= ".", label = "numerically integrated area under Po-214-peaks")
ax[0].set_title("numerically integrated area under Po-214-peaks, from $(peak- 3\sigma_{fit})$ to $(peak + 3\sigma_{fit})$")
#ax[0].set_xlabel("total absorption thickness [cm of air equivalent]")
ax[0].set_ylabel("total counts under Po-214-Peak")
ax[0].vlines(m_erf.values["c1"],0, m_erf.values["a1"], color = "grey", label = f"<R> = ({m_erf.values['c1']:.2f} +- {m_erf.errors['c1']:.2f}) cm of air equivalent" )
ax[0].legend()


fity = integr_reach_model(tot_abs_lay_thi, m_erf.values["a1"], m_erf.values["b1"], m_erf.values["c1"])

ax[1].errorbar(tot_abs_lay_thi, integral_under_peak - fity, integral_under_peak_err, fmt= ".", label = "residuals")
ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
ax[1].set_ylabel('$total counts - total counts_fit$ ')
ax[1].set_xlabel('total absorption thickness [cm of air equivalent]')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)
fig.text(0.5,0, f'a = ({m_erf.values["a1"]:.2f} +- {m_erf.errors["a1"]:.2f}) , b = ({m_erf.values["b1"]:.2f} +- {m_erf.errors["b1"]:.2f}), $<R>$ = ({m_erf.values["c1"]:.2f} +- {m_erf.errors["c1"]:.2f}),  chi2/dof = {m_erf.fval:.1f} / {m_erf.ndof} = {m_erf.fval/m_erf.ndof:.1f} ', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)

plt.savefig("integrated_reach_alpha.pdf")
plt.show()
#TODO problem keine rauschmessung 
#TODO schlechte qualitiät mit allem liegt potentielle an schlechter ausrichtung des strahlers?



#TODO COLLIMATOR

#%%collimator
#TODO kollimator4.TKA ist doppelmessung von kollimator3.TKA

coll0 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator0.TKA")
coll1 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator1.TKA")
coll2 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator2.TKA")
coll3 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator3.TKA")
coll4 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator4.TKA")
coll5 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator5.TKA")
coll6 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator6.TKA")
coll7 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator7.TKA")
#coll8 = load_spectrum("axto-t1\\axto-t1\\axto-t1-halbleiter\\kollimator8.TKA")

coll = [coll0, coll1, coll2, coll3, coll4, coll5, coll6, coll7]
coll_dists = [36.595,37.095,37.595,38.095,38.595,39.095,39.595,40.095]

chann_coll = np.arange(0, coll0.shape[0], 1)
fig, ax = plt.subplots(8, 1, figsize = (12, 17), layout = "tight")

for i in range(8):
    
    ax[i].errorbar(chann_coll, coll[i], np.sqrt(coll[i]), label = "measured spectra with collimator", fmt = ".")
    ax[i].title.set_text("Am-241 with collimator, x = " + str(coll_dists[i]) + " cm")
    ax[i].set_ylabel("counts")
    ax[i].set_xlabel("channels")
    ax[i].legend()
    
plt.savefig("collimator.pdf")    
plt.show()

#%% ionisationskammer
ion_1 = np.genfromtxt("T01\\T01\\Ionisationskammer_1.csv", delimiter = ",")
ion_1[:,0] = ion_1[0,0] - ion_1[:,0]
ion_1[:,0] /= 10
ion_2 = np.genfromtxt("T01\\T01\\Ionisationskammer_2.csv", delimiter = ",")
ion_2[:,0] = ion_2[0,0] - ion_2[:,0]
ion_2[:,0] /= 10

#fit 

plt.errorbar(ion_1[:,0], ion_1[:,1], ion_1[:,2], label = "Messreihe 1", fmt = "o")
plt.errorbar(ion_2[:,0], ion_2[:,1], ion_2[:,2], label = "Messreihe 2", fmt = "o")
plt.axhline(0, color = "silver", ls = "--")

plt.ylabel("$I$ [$nA$]")
plt.xlabel("$d-d_0$ [$mm$]")
plt.title("Ionisationskammer")
plt.legend()
plt.show()
# keine teilchem erreichen ionisationskammer bei r  = 5.4cm
reichweite_max_ion = 5.4 #unsicherheit 0.3/sqrt(12)
reichweite_max_ion_err = 0.3/12**0.5

absorption_ion = reichw_lit_umgerechnet[-1] - reichweite_max_ion
print(f"additional absorption thickness = ({absorption_ion:.2f} +- {reichweite_max_ion_err:.2f}) cm air equiv.")
print("kleiner als hl-det --> gut")

def ion_fit(x, a1, b1, c1, a2, b2, c2,):
    return a1*scipy.special.erfc(b1*(x-c1)) + a2*scipy.special.erfc(b2*(x-c2)) 

c_ion = cost.LeastSquares(ion_1[:,0], ion_1[:,1], ion_1[:,2], ion_fit)

c_in = (0.15, 1, 45, 1, 1, 15)
m_ion = iminuit.Minuit(c_ion, *c_in)

print(m_ion.migrad())

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})

x_plot = np.linspace(0, 55)
fity = ion_fit(ion_1[:,0], m_ion.values["a1"], m_ion.values["b1"], m_ion.values["c1"], m_ion.values["a2"], m_ion.values["b2"], m_ion.values["c2"],)
fity_plot = ion_fit(x_plot, m_ion.values["a1"], m_ion.values["b1"], m_ion.values["c1"], m_ion.values["a2"], m_ion.values["b2"], m_ion.values["c2"],)

ax[0].errorbar(ion_1[:,0], ion_1[:,1], ion_1[:,2], label = "measurement", fmt = "o")
ax[0].plot(x_plot, fity_plot, label = "fit")
ax[0].vlines( m_ion.values["c1"], 0, m_ion.values["a1"], label = "$R_1$", color = "darkblue")
ax[0].vlines( m_ion.values["c2"], 0, m_ion.values["a2"] + 2*m_ion.values["a1"], label = "$R_2$", color = "slateblue")
ax[0].axhline(0, color = "silver", ls = "--")


ax[0].set_ylabel("$I$ [$nA$]")
ax[0].legend()
ax[0].title.set_text("Ionisation chamber, fit")

ax[1].axhline(0, color = "black", ls = "--")
ax[1].errorbar(ion_1[:,0], ion_1[:,1]- fity, ion_1[:,2], label = "residuals", fmt = "o")
ax[1].set_ylabel('$I- I_fit$ [$nA$] ')
ax[1].set_xlabel('$d-d_0$ [$mm$]')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)
fig.text(0.5,0, f'$R_1$ = ({m_ion.values["c1"]:.2f} +- {m_ion.errors["c1"]:.2f}) mm , $R_2$ = ({m_ion.values["c2"]:.2f} +- {m_ion.errors["c2"]:.2f}) mm , chi2/dof = {m_ion.fval:.1f} / {m_ion.ndof} = {m_ion.fval/m_ion.ndof:.1f} ', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)
plt.savefig("ionisation_chamber_fit.pdf")
plt.show()

print("mean reach po214, ion")
print(f" {m_ion.values['c1']:.2f} +- {m_ion.errors['c1']:.2f} mm")
print("plus absorpt")
r_ion_tot = absorption_ion * 10 + m_ion.values['c1']
r_ion_tot_err = np.sqrt((reichweite_max_ion_err*10)**2 +  m_ion.errors['c1']**2)
print(f" {r_ion_tot:.2f} +- {r_ion_tot_err:.2f} mm")

print("mean reach po214, semiconductor")
print(f" {m_erf.values['c1']:.2f} +- {m_erf.errors['c1']:.2f} cm")
print("plus absorpt")
r_hl_tot = intrins_absorpt + m_erf.values['c1']
r_hl_tot_err = np.sqrt(intrins_absorpt_err**2 +  m_erf.errors['c1']**2)
print(f" {r_hl_tot:.2f} +- {r_hl_tot_err:.2f} mm")

