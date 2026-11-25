import numpy as np
import turtle
import time

#############################################
# Corde de guitare pincée en x = L/4 et relâchée au repos
#   equation d'onde : d2psi/dt2 = c^2 d2psi/dx2,  psi(0)=psi(L)=0
#############################################
Nx = 31          # nombre de points interieurs (les extremites sont fixes)
Nt = 600          # nombre d'images
L_corde = 1.0     # longueur de la corde
c = 1.0           # vitesse des ondes c = sqrt(T/rho)
max_t = 6.0       # duree simulee (periode fondamentale = 2L/c = 2)
h = 0.1           # hauteur du pincement
a = L_corde/4     # position du pincement

eigen_show = False   # True : affiche les modes propres un par un
init_show = True     # True : affiche la forme initiale en gris
N_mode = Nx          # nombre de modes gardes (essayer 1, 2, 3, 5, 10...)
dt_sleep = 0.02
#############################################

# grille : seulement les points interieurs, psi = 0 aux extremites
x_full = np.linspace(0, L_corde, Nx+2)
x = x_full[1:-1]
dx = x[1]-x[0]
t = np.linspace(0, max_t, Nt)

# forme initiale : triangle de sommet (a, h)
y = np.where(x < a, h*x/a, h*(L_corde-x)/(L_corde-a))

def D_Dx2(dx, N):
  inv_dx2 = (1./dx**2)
  d_dx2 = np.identity(N)*(-2*inv_dx2)
  row = np.array(range(N-1))
  col = row+1
  d_dx2[row, col] = inv_dx2
  d_dx2[col, row] = inv_dx2
  return d_dx2

# operateur K = -c^2 d2/dx2 : ses valeurs propres sont omega_n^2
K = -c**2 * D_Dx2(dx, Nx)
e, states = np.linalg.eigh(K)
omega = np.sqrt(e)

# on choisit le signe des modes pour qu'ils ressemblent a +sin(n pi x / L)
states = states * np.sign(states[0, :])

# amplitudes des modes (projection de la forme initiale)
U = states[:, :N_mode].T
U_y = U.dot(y)

# comparaison avec le resultat analytique C_n = 32h/(3 n^2 pi^2) sin(n pi/4)
norme = np.sqrt(2/(Nx+1))   # mode discret = norme * sin(n pi x/L)
print(" n   f_n (num)   f_n (theo)   C_n (num)    C_n (theo)")
for n in range(1, 13):
  C_num = U_y[n-1]*norme if n <= N_mode else 0.0
  C_th = 32*h/(3*n**2*np.pi**2)*np.sin(n*np.pi/4)
  f_num = omega[n-1]/(2*np.pi)
  f_th = n*c/(2*L_corde)
  print("%2d  %9.5f   %9.5f   %+10.6f   %+10.6f" % (n, f_num, f_th, C_num, C_th))

#############################################
# animation :

screen = turtle.Screen()

height = 600
width = 600

turtle.screensize(canvwidth=None, canvheight=None, bg=None)
screen.setup(width, height)
screen.tracer(0)

def tcurve(color, size=3):
  turtle_i = turtle.Turtle()
  turtle_i.speed(0.0)
  turtle_i.ht()
  turtle_i.pensize(size)
  turtle_i.color(color)
  return turtle_i

turtle1 = tcurve('red')
turtle2 = tcurve('gray', 1)
turtle3 = tcurve('black', 8)

def draw_curve(x, y, max_y, t1):
  # on ajoute les extremites fixes psi(0) = psi(L) = 0
  xx = np.concatenate(([0.], x, [L_corde]))
  yy = np.concatenate(([0.], y, [0.]))
  factor_x = 0.9*width / L_corde
  factor_y = 0.4*height / max_y
  off_x = -0.45*width
  t1.clear()
  t1.penup()
  t1.setpos((factor_x*xx[0]+off_x, factor_y*yy[0]))
  t1.pendown()
  for i in range(len(xx)):
    t1.goto((factor_x*xx[i]+off_x, factor_y*yy[i]))
  screen.update()

max_y = 1.2*h

# chevalet et sillet (extremites fixes)
for xe in (0., L_corde):
  turtle3.penup()
  turtle3.setpos((0.9*width*xe/L_corde - 0.45*width, -15))
  turtle3.pendown()
  turtle3.goto((0.9*width*xe/L_corde - 0.45*width, 15))

if init_show:
  draw_curve(x, y, max_y, turtle2)

if eigen_show:
  for n in range(20):
    draw_curve(x, states[:, n]/norme*h, max_y, turtle1)
    time.sleep(0.5)
else:
  for n in range(Nt):
    # pas de vitesse initiale : seulement des cosinus (S_n = 0)
    phasors = np.cos(omega[:N_mode]*t[n])
    yp1 = (U.T).dot(phasors*U_y)
    draw_curve(x, yp1, max_y, turtle1)
    time.sleep(dt_sleep)

turtle.done()
