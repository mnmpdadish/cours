import numpy as np
import turtle
import time

#############################################
Nx=501
Nt=201
max_x = 20
max_t = 5
init_speed=10
mu = -4
sigma = 1.0

x = np.linspace(-2*max_x, 2*max_x, Nx)  # we give bigger range in order to avoid edge effects
t = np.linspace(0, max_t, Nt)
y = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*(((x-mu)/(sigma))**2) - 1.j*init_speed*x)

# harmonic oscillator:
# Vx = 0.5*x**2

# well:
Vx = 0*x
# Vx[(x<2)] = 20
# Vx[(x>8)] = 20

# barrier 1:
Vx = 0*x 
Vx[(x>2)] = 20

# barrier 2:
# Vx = 0*x +10
# Vx[(x<2)] = 0
# Vx[(x>8)] = 0

# ramp:
# Vx = 4*x

eigen_show = False
real_show = True

N_mode = Nx
#############################################


LARGE_VALUE = 1e3
Vx[0]=LARGE_VALUE
Vx[-1]=LARGE_VALUE

np.set_printoptions(linewidth=600)

#max_x = x[-1]

N = Nx
dx = x[1]-x[0]

def D_Dx2(dx,N):
  inv_dx2 = (1./dx**2)
  d_dx2 = np.identity(N)*(-2*inv_dx2)
  row = np.array(range(N-1))
  col = row+1
  d_dx2[row,col] = inv_dx2
  d_dx2[col,row] = inv_dx2
  return d_dx2


L = -0.5*D_Dx2(dx,N) + np.diag(Vx)
e,states = np.linalg.eigh(L)
  
#############################################
# animating:

screen = turtle.Screen()

height = 600
width = 600

turtle.screensize(canvwidth=None, canvheight=None, bg=None)
screen.setup(width, height)
screen.tracer(0)

def tcurve(color):
  turtle_i = turtle.Turtle()
  turtle_i.speed(0.0)
  turtle_i.ht()
  turtle_i.pensize(3)
  turtle_i.color(color)
  return turtle_i

turtle1 = tcurve('red')
turtle2 = tcurve('blue')
turtle3 = tcurve('green')

def draw_curve(x,y,max_x,max_y,t1):
  factor_x = 0.9*height / (2*max_x)
  factor_y = 0.9*height / (2*max_y)
  off_y = -width*0.2
  t1.setpos((factor_x*x[0],factor_y*y[0]+off_y))
  t1.clear()
  for i in range(len(x)):
    t1.goto((factor_x*x[i],factor_y*y[i]+off_y))
  screen.update()
#############################################

#draw potential:
turtle2.pensize(5)
draw_curve(x,Vx,max_x,np.amax(Vx[1:-2]),turtle2)

U = (states[:,:N_mode].T.conj())
U_y = U.dot(y) 
max_y = np.amax(np.absolute(y))

for n in range(Nt):
  if(eigen_show):
    draw_curve(x,states[:,n],max_x,np.amax(states[:,0:100]*2),turtle1)
    time.sleep(0.5)
  else:
    phasors = np.exp(1.j * e[:N_mode]*t[n])
    yp1 = (U.T.conj()).dot(np.diag(phasors).dot(U_y))
    draw_curve(x,np.absolute(yp1),max_x,max_y,turtle1)
    if(real_show):
      draw_curve(x,np.real(yp1),    max_x,max_y,turtle3)
    time.sleep(0.04)






