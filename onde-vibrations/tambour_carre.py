import numpy as np
from scipy.special import jn, jn_zeros
import matplotlib.pyplot as plt

#############

n=2
m=4
contour_levels = 31

#############

def normal_mode(n, m, x,y):
    assert(n>0)
    assert(m>0)
    return np.sin(np.pi*n*x) * np.sin(np.pi*m*y)

xx = np.linspace(0, 1, 101)
yy = np.linspace(0, 1, 101)

values = normal_mode(n,m,xx[:,None], yy[None,:])

print(xx[:,None])
print(yy[None,:])
print(values)

print(values.shape)
max = values.max()
print(max)


fig, ax = plt.subplots()

# colors
CS = ax.contourf(xx,yy,values, contour_levels, cmap='bwr', vmin=-max, vmax=max)
ax.set_box_aspect(1)

# nodal lines:
CS2 = ax.contour(CS, levels=[0],
                  colors='black',linestyles='dashed')
plt.show()

