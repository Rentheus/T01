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

print(f"intr absortpion thickness: {intrins_absorpt:.5f}+-{intrins_absorpt_err:.5f} cm of air equiv")

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

#TODO stopping power eigentlich nicht linear
#alphateilchen verlieren nichtlinear energie => besser über range? aber auch nichtlinear
# langee tabelle über astar: reach -> energy



#def lincalib(channels, p1, e1, p2, e2, p1err, e1err, p2err, e2err):
#    m = (e2-e1)/(p2-p1)
#    merr = np.sqrt( ((1)/(p2-p1) * (np.sqrt(e1err**2 + e2err*2)))**2 + ( (e2-e1)/(p2-p1)**2 * (np.sqrt(p1err**2 + p2err*2)) )**2 )
#    print(merr)
#    b = -m * p1 + e1
#    berr = np.sqrt( (m*p1err)**2 + (p1*merr)**2 + e1err**2 )
#    ener_calib = m*channels + b
#    ener_caliberr = np.sqrt((merr*channels)**2 + berr**2)
#    return ener_calib, ener_caliberr

#oder mit linfit, linfit wahrsch. besser

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

print(ASTAR_energies_err)


def lin(x, a, b):
    return a*x+b

#a, ae, b, be, fval, cov  = lineare_regression_xy(np.array([m_ra226.values['mu'], m_po218.values['mu'], m_po214.values['mu']]), np.array([e_ra226, e_po218, e_po214]), np.array([m_ra226.errors['mu'], m_po218.errors['mu'], m_po214.errors['mu']]),  np.array([e_ra226_err, e_po218_err, e_po214_err]))
a, ae, b, be, fval, cov  = lineare_regression_xy(np.array([m_ra226.values['mu'], m_po218.values['mu'], m_po214.values['mu']]), ASTAR_energies[True,False,False,True,True], np.array([m_ra226.errors['mu'], m_po218.errors['mu'], m_po214.errors['mu']]), ASTAR_energies_err[True,False,False,True,True])

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
    ax[0,i+1].errorbar(channels[lower[i]:upper[i]], plist[i][lower[i]:upper[i]], np.sqrt(plist[i][lower[i]:upper[i]]), fmt = ".", label = "recorded Po-214-peak", zorder = 1)
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
tot_abs_lay_thi_err = np.sqrt(intrins_absorpt_err**2 + 2* (0.01/12**0.5)**2)
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
stopping_power_air = (7.69 - ASTAR_part_energies)/((total_absorpt_layer_thickness)*air_density)
sto_pow_air_err =(7.69 - ASTAR_part_energies)/(total_absorpt_layer_thickness**2*air_density) * tot_abs_lay_thi_err
print(stopping_power_air)
print(sto_pow_air_err)

ASTAR_stopping_power_air = [7.251E+02, 7.496E+02, 8.163E+02, 8.501E+02, 9.860E+02]


#energy calib
a2, a2e, b2, b2e, fval2, cov2  = lineare_regression_xy(peak_channel, ASTAR_part_energies, peak_channel_err,  np.array([0.001,0.001,0.001,0.001,0.001,]))

print(a,ae,b,be)
print(a2,a2e,b2,b2e,fval2,cov2)

plt.scatter(peak_channel, ASTAR_part_energies)
plt.plot(peak_channel, a2*peak_channel + b2)
plt.plot(peak_channel, a*peak_channel + b)
plt.show()

#TODO die calibs matchen nicht

#%%1.5 range 
#messung bei min_dist 240s, bei anderen 25s --> skalierung
integral_under_peak = [np.sum(alpha[int(peak_channel[0]-sigmas_peak_channels[0]):int(peak_channel[0]+sigmas_peak_channels[0])]) * 25/240]
for i in range(4):
    integral_under_peak.append(np.sum(plist[i][int(peak_channel[i+1]-sigmas_peak_channels[i+1]):int(peak_channel[i+1]+sigmas_peak_channels[i+1])]))
print(integral_under_peak)

#TODO errors 

plt.scatter(total_absorpt_layer_thickness, integral_under_peak)
plt.show()
#TODO letzten wert noch einfügen
