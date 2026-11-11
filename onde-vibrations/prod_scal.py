import numpy as np
#############

N=100 # number of masses
n=4   # mode number n
p=4   # mode number p

#############

# initialization:
mode_n = np.zeros((N+1))
mode_p = np.zeros((N+1))

for mm in range(N+1): # includes mm=0, but has no contribution at all.
  print(mm)
  mode_n[mm] = np.sin(np.pi*n*mm/(N+1))
  mode_p[mm] = np.sin(np.pi*p*mm/(N+1))

# scalar product
print( (2./(N+1)) * mode_p.dot(mode_n)) 

