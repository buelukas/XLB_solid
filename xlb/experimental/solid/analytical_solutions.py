import numpy as np
import sympy as sp
if __name__ == "__main__":    

    def phi_x(U, c_k, c_m):
        return [c_k*U[2]+c_m*U[3], c_m*U[4], c_k*U[0], c_m*U[0], c_m*U[1]]
    
    def phi_y(U, c_k, c_m):
        return [c_m*U[4], c_k*U[2]-c_m*U[3], c_k*U[1], -c_m*U[1], c_m*U[0]]

    x, y, t, c_k, c_m, i, j, c, delta_t = sp.symbols('x y t c_k c_m i j c delta_t')
    ux = sp.sin(4*sp.pi*(x-0.3*t)) * sp.cos(2*sp.pi*(y-0.8*t)) * sp.sin(4*sp.pi*(t-0.1))
    uy = sp.cos(4*sp.pi*(x-0.7*t)) * sp.sin(2*sp.pi*(y-0.1*t)) * sp.cos(4*sp.pi*(t+0.4))
    duxdt = sp.diff(ux, t)
    duydt = sp.diff(uy, t) 
    duxdx = sp.diff(ux, x)
    duydx = sp.diff(uy, x)
    duxdy = sp.diff(ux, y)
    duydy = sp.diff(uy, y)
    U = [duxdt, duydt, -c_k*(duxdx + duydy), -c_m*(duxdx - duydy), -c_m*(duxdy + duydx)]
    phixU = phi_x(U, c_k, c_m)
    phiyU = phi_y(U, c_k, c_m)

    dUdx = [sp.diff(f, x) for f in U]
    dUdy = [sp.diff(f, y) for f in U]
    dUdt = [sp.diff(f, t) for f in U]
    dphixUdx = [sp.diff(f, x) for f in phixU]
    dphiyUdy = [sp.diff(f, y) for f in phiyU]
    phixdUdx = phi_x(dUdx, c_k, c_m)
    phixphixdUdx = phi_x(phixdUdx, c_k, c_m)
    phiyphixdUdx = phi_y(phixdUdx, c_k, c_m)
    phiydUdy = phi_x(dUdx, c_k, c_m)
    phixphiydUdy = phi_x(phiydUdy, c_k, c_m)
    phiyphiydUdy = phi_y(phiydUdy, c_k, c_m)

    #B = dUdt + dphixUdx + dphiyUdy
    B = [dUdt[k] + dphixUdx[k] + dphiyUdy[k] for k in range(len(U))]
    phixB = phi_x(B, c_k, c_m)
    phiyB = phi_y(B, c_k, c_m)
    bx = B[0]
    by = B[1]

    f_ij = [
        0.25 * (U[k] + 2/c * (i * phixU[k] + j * phiyU[k])) -
        delta_t / 8 * (
            (B[k] + 2/c * (i * phixB[k] + j * phiyB[k])) +
            c * (
                i * dUdx[k] + (2*i**2 - 1)/c * phixdUdx[k] - 2/c**2 * (i * phixphixdUdx[k] + j * phiyphixdUdx[k]) +
                j * dUdy[k] + (2*j**2 - 1)/c * phiydUdy[k] - 2/c**2 * (i * phixphiydUdy[k] + j * phiyphiydUdy[k])
            )
        )
        for k in range(len(U))
    ]
    f_ij0 = [f_ij[k].evalf(subs={t: 0})for k in range(len(U))]
