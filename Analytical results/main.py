import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

import fncs

############################### Parameters #####################################
ms = np.array([0])
L = 1
es = np.linspace(0.01, 0.99999, 20)
v_0s = np.logspace(-8, 8, 300)
nus = 1
ns = np.array([-1, 1])
xs = np.linspace(0, 1, 100)  ## Non-dimensional
ys = np.linspace(0, 1, 100)  ## Non-dimensional
Xs, Ys = np.meshgrid(xs, ys)
Ms = np.linspace(1, 200, 100)

###############################################################################

############################### Shape functions calculation ########################
D_Msss = []    ## D_M

for i in range(len(es)):
    D_Mss = []
    for id_v in range(len(v_0s)):
        D_Ms = []
        N = norm_fac(L, ms, ns, es[i], v_0s[id_v], nus)
        for k in range(len(Ms)):
            c_M = c_xy(Ms[k], Xs, Ys, L, ms, ns, es[i], v_0s[id_v], nus)
            D_M = D_Mfunc(c_M, N)
            D_Ms.append(D_M)
        D_Mss.append(D_Ms)
    D_Msss.append(D_Mss)

D_Msss = np.array(D_Msss)

C_0ss = []    ## C_0
C_n2M2sss = []    ## C_{2-2M}
C_nMsss = []    ## C_{-M}
Cdot_n2Msss = []    ## \dot{C}_{-2M}

for i in range(len(es)):
    C_0s = []
    C_n2M2ss = []
    C_nMss = []
    Cdot_n2Mss = []
    for id_v in range(len(v_0s)):
        C_n2M2s = []
        C_nMs = []
        Cdot_n2Ms = []
        N = norm_fac(L, ms, ns, es[i], v_0s[id_v], nus)
        C_0 = C_Mfunc(0, L, ms, ns, es[i], v_0s[id_v], nus, N)
        for k in range(len(Ms)):
            C_n2M2 = C_Mfunc(- 2 * Ms[k] + 2, L, ms, ns, es[i], v_0s[id_v], nus, N)
            C_nM = C_Mfunc(- Ms[k], L, ms, ns, es[i], v_0s[id_v], nus, N)
            Cdot_n2M = Cdot_Mfunc(- 2 * Ms[k], L, ms, ns, es[i], v_0s[id_v], nus, N)
            C_n2M2s.append(C_n2M2)
            C_nMs.append(C_nM)
            Cdot_n2Ms.append(Cdot_n2M)
        C_0s.append(C_0)
        C_n2M2ss.append(C_n2M2s)
        C_nMss.append(C_nMs)
        Cdot_n2Mss.append(Cdot_n2Ms)
    C_0ss.append(C_0s)
    C_n2M2sss.append(C_n2M2ss)
    C_nMsss.append(C_nMss)
    Cdot_n2Msss.append(Cdot_n2Mss)

C_0ss = np.array(C_0ss)
C_n2M2sss = np.array(C_n2M2sss)
C_nMsss = np.array(C_nMsss)
Cdot_n2Msss = np.array(Cdot_n2Msss)

E_Msss = []    ## E_M
C_n2M1sss = []    ## C_{1 - 2M}
Cdot_n2Mn1sss = []    ## \dot{C}_{-2M - 1}

for i in range(len(es)):
    C_n2M1ss = []
    Cdot_n2Mn1ss = []
    E_Mss = []
    for id_v in range(len(v_0s)):
        E_Ms = []
        C_n2M1s = []
        Cdot_n2Mn1s = []
        N = norm_fac(L, ms, ns, es[i], v_0s[id_v], nus)
        for k in range(len(Ms)):
            E_M = E_Mfunc(Ms[k], L, Xs, Ys, ms, ns, es[i], v_0s[id_v], nus, N)
            C_n2M1 = C_Mfunc(- 2 * Ms[k] + 1, L, ms, ns, es[i], v_0s[id_v], nus, N)
            Cdot_n2Mn1 = Cdot_Mfunc(- 2 * Ms[k] - 1, L, ms, ns, es[i], v_0s[id_v], nus, N)
            C_n2M1s.append(C_n2M1)
            Cdot_n2Mn1s.append(Cdot_n2Mn1)
            E_Ms.append(E_M)
        E_Mss.append(E_Ms)
        C_n2M1ss.append(C_n2M1s)
        Cdot_n2Mn1ss.append(Cdot_n2Mn1s)
    E_Msss.append(E_Mss)
    C_n2M1sss.append(C_n2M1ss)
    Cdot_n2Mn1sss.append(Cdot_n2Mn1ss)

C_n2M1sss = np.array(C_n2M1sss)
Cdot_n2Mn1sss = np.array(Cdot_n2Mn1sss)
E_Msss = np.array(E_Msss)

#################################################################################################

##################################### Plot shape functions ######################################

# D_M vs M
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')
id_e = 0
plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
    plt.plot(Ms, D_Msss[id_e, id_v, :], ls = lss[ids], marker = "o", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))

plt.plot(Ms, D_Msss[id_e, -1, :], marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel(r"$D_M$")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("DM_vs_M.pdf")
plt.style.use('default')

# E_M vs M
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 21,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')
id_e = 0
plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
    plt.plot(Ms, E_Msss[id_e, id_v, :], ls = lss[ids], marker = "o", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))

plt.plot(Ms, E_Msss[id_e, -1, :], marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel(r"$E_M$")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("EM_vs_M.pdf")
plt.style.use('default')

# C_0 vs e
plt.figure(figsize = (8, 8))
plt.plot(es, C_0ss[:, id_v], alpha = 0.5, marker = "o", ms = 3)

plt.xlabel("e")
plt.ylabel(r"$C_0$")
plt.title(r"$C_0$ vs e at $v_0 =$" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.tight_layout()
plt.savefig("C0_e_v0.pdf")

# C_{2 - 2 M} vs M
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')

id_e = 0

plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
     plt.plot(Ms, C_n2M2sss[id_e, id_v, :], marker = "o", ls = lss[ids], ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))
plt.plot(Ms, C_n2M2sss[id_e, -1, :], marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel(r"$C_{-2M+2}$")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("C2M2_M.pdf")
plt.style.use('default')

# C_{-M} vs M
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 21,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')
id_e = 0

plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
     plt.plot(Ms, C_nMsss[id_e, id_v, :], marker = "o", ls = lss[ids], ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))
plt.plot(Ms, C_nMsss[id_e, -1, :], marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel(r"$C_{-M}$")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("CnM_M.pdf")
plt.style.use('default')

# \dot{C}_{-2M} vs M
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')
id_e = 0

plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
     plt.plot(Ms, Cdot_n2Msss[id_e, id_v, :], marker = "o", ls = lss[ids], ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))
plt.plot(Ms, Cdot_n2Msss[id_e, -1, :], marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel(r"$\dot{C}_{-2M}$")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Cdotn2M_M.pdf")
plt.style.use('default')

##################################### Factors, upper bound and fraction calculation ###########################################
lam_facss = []    ## laminar factor from La
tur_fac2ss = []    ## Turbulence factor from Tu_2
tur_fac3ss = []    ## Turbulence factor from Tu_1
firstss = []    ## La
secondss = []    ## Tu_1
thirdss = []    ## Tu_2
rhsss = []    ## Full upper bound
fdrss = []    ## fraction of La
sdrss = []    ## fraction of Tu_1
tdrss = []    ## fraction of Tu_2

for i in range(len(Ms)):
    lam_facs = []
    tur_fac2s = []
    tur_fac3s = []
    firsts = []
    seconds = []
    thirds = []
    rhss = []
    fdrs = []
    sdrs = []
    tdrs = []
    for k in range(len(v_0s)):
        lam_fac = np.sqrt(C_0ss[:, k] * C_n2M2sss[:, k, i]) / (2 * C_nMsss[:, k, i])
        tur_fac3 = np.sqrt(C_0ss[:, k]) * D_Msss[:, k, i] / (2**(3 / 2) * C_nMsss[:, k, i])
        tur_fac2 = np.sqrt(C_0ss[:, k] * Cdot_n2Msss[:, k, i]) / (2 * C_nMsss[:, k, i])
        first = lam_fac * es**2 * v_0s[k]**2
        second = tur_fac3 * es**3 * v_0s[k]**3
        third = tur_fac2 * es**2 * v_0s[k]**3
        rhs = first + second + third
        fdr = first / rhs
        sdr = second / rhs
        tdr = third / rhs
        lam_facs.append(lam_fac)
        tur_fac2s.append(tur_fac2)
        tur_fac3s.append(tur_fac3)
        firsts.append(first)
        seconds.append(second)
        thirds.append(third)
        rhss.append(rhs)
        fdrs.append(fdr)
        sdrs.append(sdr)
        tdrs.append(tdr)
    lam_facss.append(lam_facs)
    tur_fac2ss.append(tur_fac2s)
    tur_fac3ss.append(tur_fac3s)
    firstss.append(firsts)
    secondss.append(seconds)
    thirdss.append(thirds)
    rhsss.append(rhss)
    fdrss.append(fdrs)
    sdrss.append(sdrs)
    tdrss.append(tdrs)

lam_facss = np.array(lam_facss)
tur_fac2ss = np.array(tur_fac2ss)
tur_fac3ss = np.array(tur_fac3ss)
firstss = np.array(firstss)
secondss = np.array(secondss)
thirdss = np.array(thirdss)
rhsss = np.array(rhsss)
fdrss = np.array(fdrss)
sdrss = np.array(sdrss)
tdrss = np.array(tdrss)

##################################### Lower bounds calculation ###########################################
Grs = []    ## Gr
lbss = []    ## Lower bound 2.30
lb1ss = []    ## Lower bound 2.29
trs = []

for id_v in range(len(v_0s)):
    N_norm = norm_fac(L, ms, ns, es[0], v_0s[id_v], nus)
    Gr = 2 * np.pi * v_0s[id_v] * es * N_norm
    b = np.sqrt(C_n2M2sss[:, id_v, :]) + v_0s[id_v] * np.sqrt(Cdot_n2Msss[:, id_v, :])
    b1 = np.sqrt(C_n2M1sss[:, id_v, :]) + v_0s[id_v] * np.sqrt(Cdot_n2Mn1sss[:, id_v, :])
    tr = (4 * D_Msss[:, id_v, :] * C_nMsss[:, id_v, :]) / b**2
    lbs = []
    lb1s = []
    for id_e in range(len(es)):
        lb = 2 * np.pi * (- b[id_e, :] + np.sqrt(b[id_e, :]**2 + 4 * D_Msss[id_e, id_v, :] * C_nMsss[id_e, id_v, :] * Gr[id_e])) / (2 * D_Msss[id_e, id_v, :])
        lb1 = 2 * np.pi * (- b1[id_e, :] + np.sqrt(b1[id_e, :]**2 + 4 * E_Msss[id_e, id_v, :] * C_nMsss[id_e, id_v, :] * Gr[id_e] / (2 * np.pi))) / (2 * E_Msss[id_e, id_v, :])
        lbs.append(lb**2)
        lb1s.append(lb1**2)
    lbss.append(lbs)
    lb1ss.append(lb1s)
    Grs.append(Gr)
    trs.append(tr)

lbss = np.array(lbss)
lb1ss = np.array(lb1ss)
Grs = np.array(Grs)
trs = np.array(trs)

##########################################################################################################

#################################### Plot factors, upper bound and lower bounds ########################################
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 17,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# La vs M
id_e = 0

plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
     plt.plot(Ms, lam_facss[:, id_v, id_e], alpha = 0.8, marker = "o", ls = lss[ids], ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))
plt.plot(Ms, lam_facss[:, -1, id_e], alpha = 0.8, marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel("Laminar factor")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("lamfac_M.pdf")

# Tu_1 vs M
id_e = 0

plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
     plt.plot(Ms, tur_fac3ss[:, id_v, id_e], alpha = 1, marker = "o", ls = lss[ids], ms = 2, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))
plt.plot(Ms, tur_fac3ss[:, -1, id_e], alpha = 0.8, marker = "^", ms = 4, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))


plt.xlabel("M")
plt.ylabel("Turbulence factor")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("turfac_M.pdf")

# Tu_2 vs M
id_e = 0

plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, id_v in enumerate(range(0, len(v_0s), 75)):
     plt.plot(Ms, tur_fac2ss[:, id_v, id_e], alpha = 0.8, marker = "o", ls = lss[ids], ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[id_v])))
plt.plot(Ms, tur_fac2ss[:, -1, id_e], alpha = 0.8, marker = "^", ms = 3, label = r"$\log(v_0)$ = %.2f" % (np.log10(v_0s[-1])))

plt.xlabel("M")
plt.ylabel("Turbulence factor")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("extrafac_M.pdf")

## Upper bound and fraction
id_M = 0

# Upper bound vs v_0
plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, i in enumerate(range(0, len(es), 6)):
    plt.plot(v_0s, (firstss + secondss + thirdss)[id_M, :, i], ls = lss[ids], label = "e = %.3f" % (es[i]))

plt.xlabel(r"$v_0$")
plt.ylabel("Upper bound")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("epsilon_v0_M0.pdf")

# Upper bound vs Gr
id_e = -1
id_M = 49
id_e1 = 0
plt.figure(figsize = (8, 8))
N_norm = norm_fac(L, ms, ns, es[id_e], v_0s, nus)
N_norm1 = norm_fac(L, ms, ns, es[id_e1], v_0s, nus)
v0s2 = np.sqrt(C_n2M2sss / Cdot_n2Msss)
v0s1 = np.sqrt(2 * C_n2M2sss) / (es[id_e1] * D_Msss)
plt.plot(2 * np.pi * es[id_e] * v_0s * N_norm, (firstss + secondss + thirdss)[id_M, :, id_e], label = f"e = {es[id_e]}")
plt.plot(2 * np.pi * es[id_e1] * v_0s * N_norm1, (firstss + secondss + thirdss)[id_M, :, id_e1], label = f"e = {es[id_e1]}")
plt.axvline((es[id_e1] * v0s1 * np.sqrt(1 / 2 * (v0s1**2 + (2 * np.pi)**2)))[id_e1, 0, id_M], ls = "--", label = r"$Gr_{La \to Tu_1}$")
plt.axvline((es[id_e1] * v0s2 * np.sqrt(1 / 2 * (v0s2**2 + (2 * np.pi)**2)))[id_e1, 0, id_M], ls = ":", color = "black", label = r"$Gr_{La \to Tu_2}$")

plt.xlabel(r"$Gr$")
plt.ylabel("Upper bound")
plt.yscale("log")
plt.xscale("log")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig("epsilon_ev0_M49.pdf")

# Fractions
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')

plt.figure(figsize = (8, 8))
id_e = -1
id_M = 49
N_norm = norm_fac(L, ms, ns, es[id_e], v_0s, nus)
v0s1 = np.sqrt(C_n2M2sss / Cdot_n2Msss)
v0s2 = np.sqrt(2 * C_n2M2sss) / (es[id_e] * D_Msss)

plt.plot(2 * np.pi * es[id_e] * v_0s * N_norm, fdrss[id_M, :, id_e], label = "laminar")
plt.plot(2 * np.pi * es[id_e] * v_0s * N_norm, sdrss[id_M, :, id_e], label = r"tur $e^3$")
plt.plot(2 * np.pi * es[id_e] * v_0s * N_norm, tdrss[id_M, :, id_e], label = r"tur $e^2$")
plt.axvline((es[id_e] * v0s2 * np.sqrt(1 / 2 * (v0s2**2 + (2 * np.pi)**2)))[id_e, 0, id_M], ls = "--", label = r"$Gr_{La \to Tu_1}$")
plt.axvline((es[id_e] * v0s1 * np.sqrt(1 / 2 * (v0s1**2 + (2 * np.pi)**2)))[id_e, 0, id_M], ls = ":", color = "black", label = r"$Gr_{La \to Tu_2}$")

plt.xlabel(r"$Gr$")
plt.ylabel("Fraction")
plt.xscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("fraction_ev0_ef.pdf")
plt.show()
plt.style.use('default')

# Lower bounds
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')
id_M = 50
plt.figure(figsize = (8, 8))
lss = ["-", "-.", "--", ":"]
for ids, i in enumerate(range(0, len(es), 10)):
    plt.plot(Grs[:, i], lb1ss[:, i, id_M], ls = lss[ids], label = r"$e_{low1}$ =" + "%.3f" % (es[i]))
    plt.plot(Grs[:, i], lbss[:, i, id_M], ls = lss[ids+2], label = r"$e_{low}$ =" + "%.3f" % (es[i]))

plt.xlabel(r"$Gr$")
plt.ylabel("Lower bound")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("lb_v0_M50.pdf")
plt.show()
plt.style.use("default")

# All bounds (please switch off the comment for the plot corressponding to id_e1 and comment id_e to have the plot of smallest e)
plt.rcdefaults()

mpl.rcParams.update({
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "legend.fontsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

plt.style.use('tableau-colorblind10')
id_e = -1
id_M = 49
id_e1 = 0

plt.figure(figsize = (8, 8))
N_norm = norm_fac(L, ms, ns, es[id_e], v_0s, nus)
N_norm1 = norm_fac(L, ms, ns, es[id_e1], v_0s, nus)
plt.plot(Grs[:, id_e], lbss[:, id_e, id_M], label = r"$e_{low}$ =" + f"{es[id_e]}")
plt.plot(Grs[:, id_e], lb1ss[:, id_e, id_M], label = r"$e_{low1}$ =" + f"{es[id_e]}", ls = "--")
plt.plot(Grs[:, id_e], (firstss + secondss + thirdss)[id_M, :, id_e], label = r"$e_{up}$ =" + f"{es[id_e]}")
# plt.plot(Grs[:, id_e1], lbss[:, id_e1, id_M], label = r"$e_{low}$ =" + f"{es[id_e1]}")
# plt.plot(Grs[:, id_e1], lb1ss[:, id_e1, id_M], label = r"$e_{low1}$ =" + f"{es[id_e1]}", ls = "--")
# plt.plot(Grs[:, id_e1], (firstss + secondss + thirdss)[id_M, :, id_e1], label = r"$e_{up}$ =" + f"{es[id_e1]}")

plt.xlabel(r"$Gr$")
plt.ylabel("Bounds")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("bounds_ev0_ef.pdf")
plt.show()
plt.style.use("default")

# Difference bw upper and lower bounds (please switch off the comment for the plot corressponding to id_e1 and comment id_e to have the plot of smallest e)
id_e = -1
id_M = 49
id_e1 = 0

plt.figure(figsize = (8, 8))
plt.plot(Grs[:, id_e1], (firstss + secondss + thirdss)[id_M, :, id_e1] - lb1ss[:, id_e1, id_M] > 0, label = f"e = {es[id_e1]}")
plt.plot(Grs[:, id_e1], (firstss + secondss + thirdss)[id_M, :, id_e1] - lbss[:, id_e1, id_M] > 0, label = f"e = {es[id_e1]}", ls = "--")
# plt.plot(Grs[:, id_e], (firstss + secondss + thirdss)[id_M, :, id_e] - lb1ss[:, id_e, id_M] > 0, label = f"e = {es[id_e]}")
# plt.plot(Grs[:, id_e], (firstss + secondss + thirdss)[id_M, :, id_e] - lbss[:, id_e, id_M] > 0, label = f"e = {es[id_e]}", ls = "--")

plt.xlabel(r"$Gr$")
plt.ylabel(r"Diff$> 0$")
plt.xscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("diff_ev0_M49_e0.pdf")
plt.show()