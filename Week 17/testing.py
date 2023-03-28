# %%
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def HopfNormal(t,y,beta):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 - u1*((u1**2) + (u2**2))+0.0001
    du2_dt = u1 + beta*u2 - u2*((u1**2) + (u2**2))
    return [du1_dt, du2_dt]

def modHopfNormal(t,y,beta):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 + u1*(u1**2+u2**2)-u1*(u1**2+u2**2)**2
    du2_dt = u1 + beta*u2+ u2*(u1**2+u2**2)-u2*(u1**2+u2**2)**2
    return [du1_dt, du2_dt]

# %%


def ODENatural(ode,x0,t,p0,p1,Steps):
    p_range = np.linspace(p0,p1,Steps) #Returns a numpy array of length 100 containing the parameter values 
    #we vary 
    # We have x0, of the format [u1 u2], the initial guess for the solution with the initial parameter value
    # We need to start from a known solution, so we can input this into scipys solver to find the initial solution
    # using the guess as a starting point
    x0 = solve_ivp(ode,t_span=[0,t],y0=x0,t_eval=[0],args=(p0,)) #Returns a numpy array containing the initial solution of x0
    u1 = np.array([x0.y[0]]) 
    u2 = np.array([x0.y[1]])
    predicted = [u1[0][-1],u2[0][-1]]#Creating an empty numpy array for which we store solutions in 
    for i in range(1,Steps):
        p = p_range[i] #Parameter value we solve at 
        sol = solve_ivp(ode,t_span=[0,t],y0=predicted,t_eval=[0.1],args=(p,))
        u1 = np.append(u1,sol.y[0])
        u2 = np.append(u2,sol.y[1])
        predicted = [u1[-1],u2[-1]]
        print(predicted)
    return p_range, u1,u2        

p,u1,u2 = ODENatural(HopfNormal,[-1,1],0.1,-1,2,1000)
plt.plot(p,u1)
plt.plot(p,u2)
plt.show()


# %%
def ODEStep(ode,x0,p0,p1):
    p_range = np.linspace(p0,p1,1000)
    solutions = np.array([x0])
    v0 = np.array([])
    v1 = np.array([])
    initial = np.array([x0])
    p_value = np.array([p0])
    predicted = initial
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        sol = root(ode,x0=predicted,args=(p),tol=1e-6)
        predicted = sol.x
        if sol.success == True:
            solutions = np.vstack([solutions,sol.x])
            p_value = np.append(p_value,p)
    v0 = np.append(v0,p_value[1])
    v0 = np.append(v0,solutions[1])
    v1 = np.append(v1,p_value[2])
    v1 = np.append(v1,solutions[2])
    return v0,v1

x,y = ODEStep(HopfNormal,[0,0],100,10000)
print(x,y)
# %%
def PseudoODE(ode,par,x0,p0,p1):
    def conditions(input):
        Condition1=ode(input[par:3],input[0])
        Condition2 = np.dot(((input)-approx),secant)
        return [*Condition1,Condition2]
    first,second = ODEStep(ode,x0,p0,p1)
    solutions = np.array([first,second])
    v0 = first
    v1 = second
    secant = v1-v0
    approx = v1+secant
    p = p0
    while p >=p0 and p<=p1:
            sol = root(conditions,x0=approx,tol=1e-6)
            v0 = v1
            v1 = np.array(sol.x)
            secant = v1-v0
            approx = v1+secant+np.random.normal(size=(3,))
            if sol.success == True:
                solutions = np.vstack([solutions,sol.x])
                p = sol.x[0]
    return solutions[:,1],solutions[:,2],solutions[:,0]

x,y ,beta= PseudoODE(HopfNormal,[1.93999909e-03,  2.53999896e-03],-100,100)
plt.plot(beta,x,'o')
plt.plot(beta,y)
plt.show()
# %%

