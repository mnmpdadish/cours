import numpy as np
from scipy.special import jn, jn_zeros
import matplotlib.pyplot as plt

#############

n = 2      # nombre de diamètres nodaux (n >= 0)
m = 3      # nombre de cercles nodaux, bord inclus (m >= 1)
contour_levels = 31

#############

def normal_mode(n, m, r, theta):
    """Mode propre (n,m) d'une membrane circulaire de rayon 1, fixée au bord.

    u(r, theta) = J_n(k_nm r) cos(n theta)
    où k_nm = j_{n,m} est le m-ième zéro de la fonction de Bessel J_n,
    ce qui impose u = 0 au bord (r = 1).
    """
    assert(n >= 0)
    assert(m > 0)
    k = jn_zeros(n, m)[-1]          # m-ième zéro de J_n
    return jn(n, k*r) * np.cos(n*theta)

# grille polaire
rr = np.linspace(0, 1, 101)
tt = np.linspace(0, 2*np.pi, 201)

values = normal_mode(n, m, rr[:, None], tt[None, :])

# conversion en coordonnées cartésiennes pour le tracé
xx = rr[:, None] * np.cos(tt[None, :])
yy = rr[:, None] * np.sin(tt[None, :])

print(values.shape)
vmax = np.abs(values).max()
print(vmax)

# fréquence relative au mode fondamental (0,1) : f_nm / f_01 = j_nm / j_01
print("f_{%d,%d} / f_{0,1} = %.4f" % (n, m, jn_zeros(n, m)[-1] / jn_zeros(0, 1)[0]))


fig, ax = plt.subplots()

# couleurs
CS = ax.contourf(xx, yy, values, contour_levels, cmap='bwr', vmin=-vmax, vmax=vmax)
ax.set_box_aspect(1)
ax.set_aspect('equal')

# lignes nodales :
CS2 = ax.contour(xx, yy, values, levels=[0],
                 colors='black', linestyles='dashed')

# bord du tambour
phi = np.linspace(0, 2*np.pi, 400)
ax.plot(np.cos(phi), np.sin(phi), 'k-', lw=1.5)

ax.set_title("Mode (n=%d, m=%d)" % (n, m))
plt.show()
