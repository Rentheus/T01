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
from uncertainties import ufloat 
plt.rcParams["figure.figsize"] = (13,8)

#%% szintillator vergleich

sz1 = np.genfromtxt("axto-t1\\axto-t1\\axto-t1-szintillator\\sz1.TKA") # jeweils 300s
sz1rausch = np.genfromtxt("axto-t1\\axto-t1\\axto-t1-szintillator\\sz1rausch.TKA")
sz2 = np.genfromtxt("axto-t1\\axto-t1\\axto-t1-szintillator\\sz2.TKA")
sz2rausch = np.genfromtxt("axto-t1\\axto-t1\\axto-t1-szintillator\\sz2rausch.TKA")

sz1 = sz1[2:]
sz1rausch = sz1rausch[2:]
sz2 = sz2[2:]
sz2rausch = sz2rausch[2:]

sz1err = np.sqrt(sz1)
sz2err = np.sqrt(sz2)

sz1rauscherr = np.sqrt(sz1rausch) #jeweils 100s
sz2rauscherr = np.sqrt(sz2rausch)

#rauschmessung comp für hintergrundstrahlung

sz1 = sz1 - sz1rausch*3
sz1err = np.sqrt(sz1err**2 + (3*sz1rauscherr)**2)

sz2 = sz2 - sz2rausch*3
sz2err = np.sqrt(sz2err**2 + (3*sz2rauscherr)**2)


fig, ax = fig, ax = plt.subplots(2, 1, figsize=(16,8), layout = "tight")


channels = np.arange(0,sz1.shape[0], 1)
ax[0].errorbar(channels, sz1, sz1err, fmt = ".", label = "measured counts")
ax[0].annotate("photopeak", xy = (7700, 180), xytext = (7700, 40), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))


ax[1].errorbar(channels, sz2, sz2err, fmt = ".", label = "measured counts")
ax[1].annotate("photopeak", xy = (9400, 500), xytext = (9400, 40), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))
ax[1].annotate("compton edge", xy = (6000, 300), xytext = (6000, 40), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))
ax[1].annotate("backscatter peak", xy = (3100, 400), xytext = (3100, 40), arrowprops=dict(facecolor='black', shrink=0.05, width= 2))


ax[0].set_ylabel("counts")
ax[1].set_ylabel("counts")
ax[0].set_xlabel("channel")
ax[1].set_xlabel("channel")
ax[0].legend()
ax[1].legend()

ax[0].title.set_text("Szintillator 1 (Plastic), $t_{active} = 300s$, Cs-137, $t_{dead}$ ≈  4% ")
ax[1].title.set_text("Szintillator 2 (NaI:Tl), $t_{active} = 300s$, Cs-137, $t_{dead}$ ≈  20% ",)
plt.savefig("szintillator_vgl.pdf")

plt.show()


#%% rel detection probability: 
# höhe von photopeak mit fit von gaußkurve
#sz1
def gauss(x, mu, sigma, a, c):
    return  a*1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2) +c

c_sz1 = cost.LeastSquares(channels[6000:11000], sz1[6000:11000], sz1err[6000:11000], gauss)

m_sz1 = iminuit.Minuit(c_sz1, mu = 7000, sigma=10, a = 1, c= 0)

print(m_sz1.migrad(ncall = 50000))
#print(m_sz1.hesse())
fig, ax = plt.subplots(2, 1, figsize=(10,8), layout = "tight",gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
fit_counts = gauss(channels[6000:11000], m_sz1.values["mu"], m_sz1.values["sigma"], m_sz1.values["a"], m_sz1.values["c"])

ax[0].errorbar(channels[6000:11000], sz1[6000:11000], sz1err[6000:11000], label = "measurements", fmt = ".", zorder= 1)
ax[0].plot(channels[6000:11000], fit_counts, label = "fit", zorder = 2)
ax[0].axvline(m_sz1.values["mu"], label = "peak", zorder = 3, color = "black")
ax[0].set_ylabel("counts")
ax[0].legend()


ax[1].errorbar(channels[6000:11000], sz1[6000:11000] - fit_counts, sz1err[6000:11000] , label = "residuals", zorder = 1)

ax[1].axhline(0, color = "black", linestyle = "--",zorder = 2)

 
              
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].set_ylabel("$counts - counts_{Fit}$")

ax[1].set_xlabel("channel")



ax[1].legend()
ax[0].title.set_text("photopeak plastic szintillator")


fig.text(0.5,0, f'A = ({m_sz1.values["a"]:.2f} +- {m_sz1.errors["a"]:.2f}) ,'+' $\overline{n}$'+f' = ({m_sz1.values["mu"]:.2f} +- {m_sz1.errors["mu"]:.2f}), $δ$ = ({m_sz1.values["sigma"]:.2f} +- {m_sz1.errors["sigma"]:.2f}), $B$ = ({m_sz1.values["c"]:.2f} +- {m_sz1.errors["c"]:.2f}),  chi2/dof = {m_sz1.fval:.1f} / {m_sz1.ndof} = {m_sz1.fval/m_sz1.ndof:.1f} ', horizontalalignment = "center")

plt.savefig("photopeak_plastic.pdf")

plt.show()


#%% sz2

c_sz2 = cost.LeastSquares(channels[8500:12500], sz2[8500:12500], sz2err[8500:12500], gauss)

m_sz2 = iminuit.Minuit(c_sz2, mu = 9500, sigma=10, a = 100000, c= 160)

print(m_sz2.migrad(ncall = 20000))
fig, ax = plt.subplots(2, 1, figsize=(10,8), layout = "tight",gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
fit_counts = gauss(channels[8500:12500], m_sz2.values["mu"], m_sz2.values["sigma"], m_sz2.values["a"], m_sz2.values["c"])

ax[0].errorbar(channels[8500:12500], sz2[8500:12500], sz2err[8500:12500], label = "measurements", fmt = ".", zorder= 1)
ax[0].plot(channels[8500:12500], fit_counts, label = "fit", zorder = 2)
ax[0].axvline(m_sz2.values["mu"], label = "peak", zorder = 3, color = "black")
ax[0].set_ylabel("counts")
ax[0].legend()


ax[1].errorbar(channels[8500:12500], sz2[8500:12500] - fit_counts, sz2err[8500:12500] , label = "residuals", zorder = 1)

ax[1].axhline(0, color = "black", linestyle = "--",zorder = 2)

 
              
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].set_ylabel("$counts - counts_{Fit}$")

ax[1].set_xlabel("channel")



ax[1].legend()
ax[0].title.set_text("photopeak NaI:Tl szintillator")
fig.text(0.5,0, f'A = ({m_sz2.values["a"]:.2f} +- {m_sz2.errors["a"]:.2f}) ,'+' $\overline{n}$'+f' = ({m_sz2.values["mu"]:.2f} +- {m_sz2.errors["mu"]:.2f}), $δ$ = ({m_sz2.values["sigma"]:.2f} +- {m_sz2.errors["sigma"]:.2f}), $B$ = ({m_sz2.values["c"]:.2f} +- {m_sz2.errors["c"]:.2f}),  chi2/dof = {m_sz2.fval:.1f} / {m_sz2.ndof} = {m_sz2.fval/m_sz2.ndof:.1f} ', horizontalalignment = "center")

plt.savefig("photopeak_NaI.pdf")
plt.show()
#%% detection probability and light yield

# werte im umfeld von 7 channels um peak
peak_sz1_val = sz1[int(m_sz1.values["mu"])-3:int(m_sz1.values["mu"])+3]
peak_sz2_val = sz2[int(m_sz2.values["mu"])-3:int(m_sz2.values["mu"])+3]

peak_sz1_mean = np.mean(peak_sz1_val)
peak_sz2_mean = np.mean(peak_sz2_val)

peak_sz1_std = np.std(peak_sz1_val, ddof = 1)
peak_sz2_std = np.std(peak_sz2_val, ddof = 1)

eta = peak_sz1_mean/peak_sz2_mean
etaerr = np.sqrt((1/peak_sz2_mean*peak_sz1_std)**2 + (peak_sz1_mean/peak_sz2_mean**2*peak_sz2_std)**2)

peak_sz1 = m_sz1.values["a"]*1/(m_sz1.values["sigma"] * np.sqrt(2*np.pi)) + m_sz1.values["c"]
peak_sz2 = m_sz2.values["a"]*1/(m_sz2.values["sigma"] * np.sqrt(2*np.pi)) + m_sz2.values["c"]

peak_sz1_err = np.sqrt((1/(m_sz1.values["sigma"] * np.sqrt(2*np.pi)) * m_sz1.errors["a"])**2 + (m_sz1.values["a"]*1/(m_sz1.values["sigma"]**2 * np.sqrt(2*np.pi)) * m_sz1.errors["sigma"])**2 + m_sz1.errors["c"]**2)
peak_sz2_err = np.sqrt((1/(m_sz2.values["sigma"] * np.sqrt(2*np.pi)) * m_sz2.errors["a"])**2 + (m_sz2.values["a"]*1/(m_sz2.values["sigma"]**2 * np.sqrt(2*np.pi)) * m_sz2.errors["sigma"])**2 + m_sz2.errors["c"]**2)


epsilon = peak_sz1/peak_sz2
epsilonerr = np.sqrt((1/peak_sz2*peak_sz1_err)**2 + (peak_sz1/peak_sz2**2*peak_sz2_err)**2)



print(f"Detection probability: ({epsilon:.2f} +- {epsilonerr:.2f})")

print(m_sz1.values["mu"])
print(m_sz2.values["mu"])

#rel light yield: gain increase between pmt stages is approx linear with voltage ==> 10 Stages (fig 5 in manual) 
#sz 1: 750V
#sz 2: 732V
# ==> sz 1 gain prop (750)^10 , sz2 gain prop 732^10
#used amp: canberra 2015b
#gain on nim module: coarse 64 fine sz1: 7,11 sz2 10,2 

#light yield: (mean(sz1)/gain(sz1)) / (mean(sz2)/(gain(sz2))) mean entspricht channel

light_yield = m_sz1.values["mu"]/m_sz2.values["mu"] * (732**10* 10.2)/(750**10 * 7.11)

light_yield_err = np.sqrt(((732**10* 10.2)/(750**10 * 7.11))**2 * ((1/m_sz2.values["mu"] * m_sz1.errors["mu"])**2 +(m_sz1.values["mu"]/m_sz2.values["mu"]**2 * m_sz1.errors["mu"])**2)) 


print(f"rel light yield: ({light_yield:.4f} +- {light_yield_err:.4f})")

epsilon_u = ufloat(epsilon, epsilonerr)
eta_u = ufloat(light_yield, light_yield_err)

QE = epsilon_u * eta_u
print(f"rel quantum efficiency:")
print(QE)


#%% gamma absorption

noise = np.genfromtxt("abs_gamma_rad_in_lead\\Noise.TKA")
t_noise = noise[0]
noise = noise[2:]

lengths = [0,1,2,3,5,10,20]
measurements = []
m_err = []
for i in lengths:
    m = np.genfromtxt("abs_gamma_rad_in_lead\\L"+ str(i) + ".TKA")
    print(m[0])
    measurements.append(m[2:] - noise)
    m_err.append(np.sqrt( m[2:] + noise))

chan_gamma = np.arange(0,measurements[0].shape[0], 1)
plt.errorbar(chan_gamma, noise, np.sqrt(noise), fmt = ".", label = "noise")
plt.title("noise measurement, Pb-absorption")
plt.ylabel("counts")
plt.xlabel("channel")
plt.legend()
plt.show()


for i in range(len(measurements)):
    plt.errorbar(chan_gamma, measurements[i], m_err[i], fmt = ".", label = "$\gamma$-spectrum, $d_{absorber} = $" + str(lengths[i])+" mm")
    
plt.title("Pb-absorber measurements without noise, Pb-absorption")

plt.ylabel("counts")
plt.xlabel("channel")
plt.legend()
plt.savefig("gamma_initial_visualisation.pdf")
plt.show()


#fit peaks

fig, ax = fig, ax = plt.subplots(4, 4, figsize=(16,8), layout = "tight")

peak_intensity = []
p_i_err = []

for i in range(len(measurements)):
     c_gamma = cost.LeastSquares(chan_gamma[750:950], measurements[i][750:950], m_err[i][750:950], gauss)

     m_gamma = iminuit.Minuit(c_gamma, mu = 850, sigma=10, a = 100, c= 0)

     print(m_gamma.migrad())
     ax[(i//4)*2, i%4].errorbar(chan_gamma[750:950], measurements[i][750:950], m_err[i][750:950], fmt = ".", label = "measurement",zorder = 1)
     ax[(i//4)*2, i%4].title.set_text("$\gamma$-spectrum, $d_{absorber} = $" + str(lengths[i])+" mm")
     
     c_fit = gauss(chan_gamma[750:950], m_gamma.values["mu"], m_gamma.values["sigma"], m_gamma.values["a"], m_gamma.values["c"])
     ax[(i//4)*2, i%4].plot(chan_gamma[750:950], c_fit, label = "fit", zorder =2 )
     ax[(i//4)*2, i%4].set_ylabel("counts")
     
     
     ax[(i//4)*2+1, i%4].errorbar(chan_gamma[750:950], measurements[i][750:950] - c_fit, m_err[i][750:950] , label = "residuals", zorder = 1)

     ax[(i//4)*2+1, i%4].axhline(0, color = "black", linestyle = "--",zorder = 2)

      
                   
     ymax = max([abs(x) for x in ax[(i//4)*2+1, i%4].get_ylim()])
     ax[(i//4)*2+1, i%4].set_ylim(-ymax, ymax)
     ax[(i//4)*2+1, i%4].set_ylabel("$counts - counts_{Fit}$")

     ax[(i//4)*2+1, i%4].set_xlabel("channel")

        

     ax[(i//4)*2+1, i%4].title.set_text(f"$\mu$ = ({m_gamma.values['mu']:.2f}+-{m_gamma.errors['mu']:.2f}), $\chi^2/ndof$ = {m_gamma.fval/m_gamma.ndof:.2f} ")
     ax[(i//4)*2+1, i%4].legend()
     #peak_intensity.append(np.mean(measurements[i][int(m_gamma.values['mu'])-2:int(m_gamma.values['mu'])+2]))
     #p_i_err.append(np.std(measurements[i][int(m_gamma.values['mu'])-2:int(m_gamma.values['mu'])+2], ddof = 1))
     peak_intensity.append(m_gamma.values["a"]/(np.sqrt(2*np.pi) *m_gamma.values["sigma"] ) + m_gamma.values["c"])
     p_i_err.append(np.sqrt( m_gamma.errors["c"]**2 + (1/(np.sqrt(2*np.pi) *m_gamma.values["sigma"] ) * m_gamma.errors["a"])**2 + (m_gamma.values["a"]/(np.sqrt(2*np.pi) *m_gamma.values["sigma"]**2 ) * m_gamma.errors["sigma"])**2 ))
     

plt.savefig("gamma_absorption_peaks.pdf")
plt.show()

print(peak_intensity)
print(p_i_err)
#%% fit absorption
lengths = np.array(lengths)
peak_intensity = np.array(peak_intensity)
p_i_err = np.array(p_i_err)

def exp(x, I_0, mu, c):
    return I_0*np.exp(-x*mu) + c

    
c_absorpt = cost.LeastSquares(lengths, peak_intensity, p_i_err, exp)
m_absorpt = iminuit.Minuit(c_absorpt, I_0 = 400, mu = 1, c = 0)
print(m_absorpt.migrad())

fig, ax = plt.subplots(2, 1, figsize=(10,8), layout = "tight",gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
fit_counts = exp(lengths, m_absorpt.values["I_0"], m_absorpt.values["mu"], m_absorpt.values["c"])
linx = np.linspace(0,20)
fit_counts2 =  exp(linx, m_absorpt.values["I_0"], m_absorpt.values["mu"], m_absorpt.values["c"])
ax[0].errorbar(lengths, peak_intensity, p_i_err, label = "measurements", fmt = "o", zorder= 1)
ax[0].plot(linx, fit_counts2, label = "fit", zorder = 2)
ax[0].set_ylabel("counts $\propto$ $I_{\gamma}$")
ax[0].legend()


ax[1].errorbar(lengths, peak_intensity - fit_counts, p_i_err , label = "residuals", zorder = 1, fmt = "o")

ax[1].axhline(0, color = "black", linestyle = "--",zorder = 2)

 
              
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].set_ylabel("$counts - counts_{Fit}$")

ax[1].set_xlabel("$d_{Pb}$ [$mm$]")



ax[1].legend()
ax[0].title.set_text("absorption of $\gamma$-radiation by Pb")
fig.text(0.5,0, f'$I_0$ = ({m_absorpt.values["I_0"]:.2f} +- {m_absorpt.errors["I_0"]:0.2f}) , $\mu$ = ({m_absorpt.values["mu"]:.3f} +- {m_absorpt.errors["mu"]:.3f}) $1/cm$, B = ({m_absorpt.values["c"]:.1f} +- {m_absorpt.errors["c"]:.1f}),  chi2/dof  = {m_absorpt.fval:.1f}/{m_absorpt.ndof} = {m_absorpt.fval/m_absorpt.ndof:.1f} ', horizontalalignment = "center")

plt.savefig("absorption_gamma.pdf")
plt.show()
print("mass attenuation coefficient = mu / rho_Pb [cm^2/g](zitieren)")
print(f"{m_absorpt.values['mu']/11.342 :.4f} +- {m_absorpt.errors['mu']/11.342 :.4f} cm^2/g")