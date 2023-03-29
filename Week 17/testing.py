# %%
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ode(t,y,beta):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 + u1*((u1**2) + (u2**2))
    du2_dt = u1 + beta*u2 + u2*((u1**2) + (u2**2))
    return [du1_dt, du2_dt]

def modHopfNormal(t,y,beta):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 + u1*(u1**2+u2**2)-u1*(u1**2+u2**2)**2
    du2_dt = u1 + beta*u2+ u2*(u1**2+u2**2)-u2*(u1**2+u2**2)**2
    return [du1_dt, du2_dt]

# %%
def shooting(x0,ode):
    """
    A function that uses the numerical shooting method in order to find the limit cycles of an ode

    SYNTAX
    ------
    The function is called the following way;

    shooting(x0,ode):

        x0: the x0 input is a list containing an initial guess of the initial values of the limit cycle for the specified ode, the form of
        this input should be as follows;

            x0 = [x0,y0,T]

        ode: the ode input is the ordinary differential equation for which we want to find the limit cycle of, the form of the ode
        input should be as follows;

            def PredPrey(t,x0): Defining the ode as a function with inputs t and x0, time and initial conditions
            a = 1     |
            b = 0.1   |  Defining any parameters inside the function handle, you could leave them out of the function but you
            d = 0.1   |  would have to define them as arguments when calling the function
            x = x0[0] Splitting the initial conditions array into its constituent parts
            y = x0[1]
            dx_dt = x*(1-x) - (a*x*y)/(d+x) #Defining the differential equations
            dy_dt = b*y*(1-(y/x))
            return [dx_dt, dy_dt] #Returns the value of the ode as an array 
    
    OUTPUT
    ------
        The shooting function returns the correct initial values of the limit cycle
    """
    def cons(x0,ode): #Finds the points and time period of the limit cycle
        condition_1 = x0[:2]- scipy.integrate.solve_ivp(ode,[0, x0[2]],x0[:2], rtol = 1e-4).y[:,-1]
        condition_2 = ode(0,x0[:2])[0] 
        return [*condition_1,condition_2]
    result = scipy.optimize.root(cons, x0 = x0, args=(ode,))
    print(result.x)
    return(result.x)



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

