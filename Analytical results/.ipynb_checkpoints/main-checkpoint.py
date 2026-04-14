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
D_Msss = []

for i in range(len(es)):
    D_Mss = []
    for id_v in range(len(v_0s)):
        D_Ms = []
        N = np.sqrt(norm_fac(L, ms, ns, es[i], v_0s[id_v], nus))
        for k in range(len(Ms)):
            c_M = c_xy(Ms[k], Xs, Ys, L, ms, ns, es[i], v_0s[id_v], nus)
            D_M = D_Mfunc(c_M, N)    ### D_M
            D_Ms.append(D_M)
        D_Mss.append(D_Ms)
    D_Msss.append(D_Mss)

D_Msss = np.array(D_Msss)

C_0ss = []
C_n2M2sss = []
C_nMsss = []
Cdot_n2Msss = []

for i in range(len(es)):
    C_0s = []
    C_n2M2ss = []
    C_nMss = []
    Cdot_n2Mss = []
    for id_v in range(len(v_0s)):
        C_n2M2s = []
        C_nMs = []
        Cdot_n2Ms = []
        N = np.sqrt(norm_fac(L, ms, ns, es[i], v_0s[id_v], nus))
        C_0 = C_Mfunc(0, L, ms, ns, es[i], v_0s[id_v], nus, N)
        for k in range(len(Ms)):
            C_n2M2 = C_Mfunc(- 2 * Ms[k] + 2, L, ms, ns, es[i], v_0s[id_v], nus, N)    ### C_{2 - 2M}
            C_nM = C_Mfunc(- Ms[k], L, ms, ns, es[i], v_0s[id_v], nus, N)              ### C_{-M}
            Cdot_n2M = Cdot_Mfunc(- 2 * Ms[k], L, ms, ns, es[i], v_0s[id_v], nus, N)   ### \dot{C}_{-2M}
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

#################################################################################################

##################################### Plot shape functions ######################################
id_v = -1

# D_M vs M
plt.figure(figsize = (8, 8))
for id_e in range(len(es)):
    plt.plot(Ms, D_Msss[id_e, id_v, :], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel(r"$D_M$")
plt.yscale("log")
plt.title(r"$D_M$ vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("DM_vs_M_v0.pdf")

# C_0 vs e
plt.figure(figsize = (8, 8))
plt.plot(es, C_0ss[:, id_v], alpha = 0.5, marker = "o", ms = 3)

plt.xlabel("e")
plt.ylabel(r"$C_0$")
# plt.yscale("log")
plt.title(r"$C_0$ vs e at $v_0 =$" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.tight_layout()
plt.savefig("C0_e_v0.pdf")

# C_{2 M - 2} vs M
plt.figure(figsize = (8, 8))
for id_e in range(len(es)):
    plt.plot(Ms, C_n2M2sss[id_e, id_v, :], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel(r"$C_{-2M+2}$")
plt.yscale("log")
plt.title(r"$C_{-2M+2}$ vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Cn2M2_M_v0.pdf")

# C_{-M} vs M
plt.figure(figsize = (8, 8))
for id_e in range(len(es)):
    plt.plot(Ms, C_nMsss[id_e, id_v, :], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel(r"$C_{-M}$")
plt.yscale("log")
plt.title(r"$C_{-M}$ vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("CnM_M_v0.pdf")

# \dot{C}_{-2M} vs M
plt.figure(figsize = (8, 8))
for id_e in range(len(es)):
    plt.plot(Ms, Cdot_n2Msss[id_e, id_v, :], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel(r"$\dot{C}_{-2M}$")
plt.yscale("log")
plt.title(r"$\dot{C}_{-2M}$ vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Cdotn2M_M_v0.pdf")

##################################### Factors, upper bound and fraction calculation ###########################################
lam_facss = []
tur_fac2ss = []
tur_fac3ss = []
firstss = []
secondss = []
thirdss = []
rhsss = []
fdrss = []
sdrss = []
tdrss = []

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
        lam_fac = np.sqrt(C_0ss[:, k] * C_n2M2sss[:, k, i]) / (2 * C_nMsss[:, k, i])    ### Laminar factor
        tur_fac3 = np.sqrt(C_0ss[:, k]) * D_Msss[:, k, i] / (2**(3 / 2) * C_nMsss[:, k, i])    ### turbulent factor e^3
        tur_fac2 = np.sqrt(C_0ss[:, k] * Cdot_n2Msss[:, k, i]) / (2 * C_nMsss[:, k, i])    ### turbulent factor e^2
        first = lam_fac * es**2 * v_0s[k]**2    ### Laminar term La
        second = tur_fac3 * es**3 * v_0s[k]**3    ### Turbulent term Tu_1
        third = tur_fac2 * es**2 * v_0s[k]**3    ### Turbulent term Tu_2
        rhs = first + second + third    ### Upper bound
        fdr = first / rhs    ### Laminar fraction
        sdr = second / rhs    ### Turbulent fraction Tu_1
        tdr = third / rhs    ### Turbulent fraction Tu_2
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

##########################################################################################################

#################################### Plot factors and upper bound ########################################
id_v = 0

# La vs M
plt.figure(figsize = (10, 10))
for id_e in range(len(es)):
    plt.plot(Ms, lam_facss[:, id_v, id_e], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel("Laminar factor")
# plt.yscale("log")
plt.title("Laminar factor vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("lamfac_M_v0.pdf")

# Tu_1 vs M
plt.figure(figsize = (8, 8))
for id_e in range(len(es)):
    plt.plot(Ms, tur_fac3ss[:, id_v, id_e], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel("Turbulence factor")
# plt.yscale("log")
plt.title("Turbulence factor vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("turfac_M_v0.pdf")

# Tu_2 vs M
plt.figure(figsize = (8, 8))
for id_e in range(len(es)):
    plt.plot(Ms, tur_fac2ss[:, id_v, id_e], alpha = 0.5, marker = "o", ms = 3, label = "e = %.3f" % (es[id_e]))

plt.xlabel("M")
plt.ylabel("Turbulence factor")
# plt.yscale("log")
plt.title("Turbulence factor of extra term vs M at $v_0 = $" + "%.3f" %(v_0s[id_v]))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("extrafac_M_v0.pdf")

## Upper bound and fraction
id_M = 0

# Upper bound vs v_0
plt.figure(figsize = (8, 8))
for i in range(len(es)):
    plt.plot(v_0s, (firstss + secondss + thirdss)[id_M, :, i], marker = "o", alpha = 0.6, ms = 1, label = f"e = {es[i]}")

plt.xlabel(r"$v_0$")
plt.ylabel("rhs")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Dissipation upper limit vs $v_0$ at $M = $" + "%.3f" %(Ms[id_M]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("epsilon_v0_M0.pdf")

# Upper bound vs Re
id_e = -1
id_e1 = 0
plt.figure(figsize = (8, 8))
# plt.plot(es[id_e] * v_0s[v_0mid], (firstss + secondss + thirdss)[0, v_0mid, id_e], marker = "o", alpha = 0.6, ms = 1, label = f"e = {es[id_e]}")
plt.plot(es[id_e] * v_0s / np.sqrt(2), (firstss + secondss + thirdss)[id_M, :, id_e], marker = "o", alpha = 0.6, ms = 1, label = f"e = {es[id_e]}")
plt.plot(es[id_e1] * v_0s / np.sqrt(2), (firstss + secondss + thirdss)[id_M, :, id_e1], marker = "o", alpha = 0.6, ms = 1, label = f"e = {es[id_e1]}")
# plt.plot(es[id_e1] * v_0s, 500 * secondss[id_M, :, id_e1], ls = "--", alpha = 0.6, ms = 1, label = "second term")
plt.axvline((np.sqrt(C_n2M2sss) / D_Msss)[id_e1, -1, id_M] , ls = "--", alpha = 0.4, ms = 1, label = r"$Re_{La \to Tu_1} (e = 0.01)$")
plt.axvline((np.sqrt(C_n2M2sss / (2 * Cdot_n2Msss)) * es[id_e1])[id_e1, -1, id_M], ls = "--", color = "black", alpha = 0.4, ms = 1, label = r"$Re_{La \to Tu_2} (e = 0.01)$")

plt.xlabel(r"$Re$")
plt.ylabel("rhs")
plt.yscale("log")
plt.xscale("log")
plt.title(r"Dissipation upper limit vs $Re$ at $M = $" + "%.3f" %(Ms[id_M]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("epsilon_ev0_M0.pdf")

# Fractions
plt.figure(figsize = (8, 8))
id_e = 0

plt.plot(es[id_e] * v_0s  / np.sqrt(2), fdrss[id_M, :, id_e], marker = "o", alpha = 0.6, ms = 1, label = "laminar")
plt.plot(es[id_e] * v_0s / np.sqrt(2), sdrss[id_M, :, id_e], marker = "o", alpha = 0.6, ms = 1, label = r"tur $e^3$")
plt.plot(es[id_e] * v_0s / np.sqrt(2), tdrss[id_M, :, id_e], marker = "o", alpha = 0.6, ms = 1, label = r"tur $e^2$")
plt.axvline((np.sqrt(C_n2M2sss) / D_Msss)[id_e, -1, id_M], ls = "--", alpha = 0.4, ms = 1, label = r"$Re_{La \to Tu_1}$")
plt.axvline((np.sqrt(C_n2M2sss / (2 * Cdot_n2Msss)) * es[id_e])[id_e, 0, id_M], ls = "--", color = "black", alpha = 0.4, ms = 1, label = r"$Re_{La \to Tu_2}$")

plt.xlabel(r"$Re$")
plt.ylabel("Fraction")
# plt.yscale("log")
plt.xscale("log")
plt.title(r"Fraction of dissipation upper limit vs $Re$ at $M = $" + "%.3f" %(Ms[id_M]) + " and $e = $" + "%.6f" %(es[id_e]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("fraction_ev0_e0.pdf")