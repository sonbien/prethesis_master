## defining functions (Some of the parameters are no need to be input of functions, but we did not change it while updating the code to keep the compatibility of the whole code)

l_n = lambda x: (2 * np.pi * x)**2  ## Non-dimensional

def phihat_y(L, m, n, e, v_0, nu):
    p_y = 0
    if n == 1 or n == -1 and m == 0:
        p_y = v_0 / 2 - np.sign(n) * 1j * np.pi   ## Non-dimensional
        
    return p_y

def norm_fac(L, ms, ns, e, v_0, nu):
    S = 1 / (2 * np.pi) * np.sqrt(1 / 2 * (v_0**2 + (2 * np.pi)**2))  ## Non-dimensional
    return S

def c_xy(M, x, y, L, ms, ns, e, v_0, nu):
    S = 0
    for n in ns:
        for m in ms:
            S += 1 / (l_n(n))**M * 1j * n * 2 * np.pi * phihat_y(L, m, n, e, v_0, nu) * np.exp(1j * 2 * np.pi * n * x)  ## Non-dimensional

    return np.abs(S)

def D_Mfunc(c, norm):
    return np.max(1 / norm * np.sqrt(c**2))

def E_Mfunc(M, L, x, y, ms, ns, e, v_0, nu, norm):
    S = 0
    for n in ns:
        for m in ms:
            S += 1 / (l_n(n))**M * phihat_y(L, m, n, e, v_0, nu) * np.exp(1j * 2 * np.pi * n * x)  ## Non-dimensional

    return np.max(1 / norm * np.abs(S))

def C_Mfunc(M, L, ms, ns, e, v_0, nu, norm):
    S = 0
    for n in ns:
        for m in ms:
            S += (l_n(n))**M * np.abs(phihat_y(L, m, n, e, v_0, nu))**2  ## Non-dimensional
    return 1 / norm**2 * S

def phihatdot_y(L, m, n, e, v_0, nu):
    p_y = 0
    if n == 1 or n == -1 and m == 0:
        p_y = 2 * np.pi * (n * 1j * v_0 / 2 + np.pi)  ## Non-dimensional

    return p_y

def Cdot_Mfunc(M, L, ms, ns, e, v_0, nu, norm):
    S = 0
    for n in ns:
        for m in ms:
            S += (l_n(n))**M * np.abs(phihatdot_y(L, m, n, e, v_0, nu))**2

    return 1 / norm**2 * S