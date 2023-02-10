import numpy as np
import matplotlib.pyplot as plt

f1 = lambda t, x: np.array([x[1], -x[0]]) # x' = y, y' = -x
x0 = np.array([1, 1]) # Initial condition, x = 1, y = 0
def true_fun(t):
    x = np.cos(t) + np.sin(t)
    return x

def euler_step(f,xn,t,h):
    x = xn + h*f(t,xn)
    return x

#Re-do figures!!

def rk_step(f,xn,t,h):
    k1 = f(t,xn)
    k2 = f(t+h/2,xn + h*k1/2)
    k3 = f(t+h/2,xn+h*(k2/2))
    k4 = f(t+h,xn+(h*k3))
    x = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x

def solve_to_system(f, x0, t1, t2, h, step_function):
    t = np.arange(t1,t2+h,h)
    x = x0
    xval = [x]
    for i in range(1,len(t)):
        x = step_function(f,x,t,h)
        xval.append(x)
    return t, xval

h = 0.1
t,EulerValues = solve_to_system(f1,x0,0,100,h,euler_step)
t1,RKValues = solve_to_system(f1,x0,0,100,h,rk_step)

XEuler = [x[0] for x in EulerValues]
YEuler = [x[1] for x in EulerValues]

XRK = [x[0] for x in RKValues]
YRK = [x[1] for x in RKValues]

Xtrue = true_fun(t)

plt.plot(t,XEuler,label = 'X, Euler Approximation',color='red')
plt.plot(t1,XRK,label='X, RK Approximation', color='blue')
plt.plot(t,Xtrue,label='True solution', color ='black',linestyle =':',linewidth=5 )
plt.xlabel('t')
plt.ylabel('X')
plt.title('X Approximation for both methods with step size '+str(h))
plt.legend()
plt.grid()
plt.show()

plt.plot(t1,YRK,label='Y, RK Approximation',color='blue')
plt.plot(t,YEuler,label='Y, Euler Approximation', color='red')
plt.xlabel('t')
plt.ylabel('Y')
plt.title('Y Approximation for both methods'+str(h))
plt.legend()
plt.grid()
plt.show()


