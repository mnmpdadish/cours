import scipy.linalg as la
import numpy as np
np.set_printoptions(precision=4, suppress = True)

H = np.array([[ 2,-1, 0],
              [-1, 2,-1],
              [ 0,-1, 2]])

S = np.array([[ 1, 0, 0],
              [ 0, 1, 0],
              [ 0, 0, 1]])

valeurs, vecteurs = la.eigh(H,S)

print(f'valeurs = \n{valeurs}')
print(f'\nvecteurs = \n{vecteurs}')

