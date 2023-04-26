# %%
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root


def euler_step(f,xn,t,h,*args):
    f = np.array(f(t,xn,*args))
    x = xn + h*f
    return x

def solve_toEU(f,x0,t1,t2,h,*args):
   if t1 <0:
    raise ValueError('Time must be a positive integer')
   if t2 < t1:
    raise ValueError("This function iterates forwards in time, please provide a correct time interval")
   deltat_max = 0.1
   if h > deltat_max:
    raise ValueError("Step size too large for accurate approximation")
   t = np.arange(t1,t2+h,h)
   x = np.array([x0])
   for i in range(1,len(t)):
        Value = euler_step(f,x[i-1],t[i-1],h,*args)
        x = np.vstack([x, Value]) 
   return t,x 

def rk_step(f,xn,t,h,*args):
    k1 = np.array(f(t,xn,*args))
    k2 = np.array(f(t+h/2,xn + h*k1/2,*args))
    k3 = np.array(f(t+h/2,xn+h*(k2/2),*args))
    k4 = np.array(f(t+h,xn+(h*k3),*args))
    x = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x

def solve_toRK(f,x0,t1,t2,h,*args):
   if t1 <0:
    raise ValueError('Time must be a positive integer')
   if t2 < t1:
    raise ValueError("This function iterates forwards in time, please provide a correct time interval")
   deltat_max = 0.25
   if h > deltat_max:
    raise ValueError("Step size too large for accurate approximation")
   t = np.arange(t1,t2+h,h)
   x = np.array([x0])
   for i in range(1,len(t)):
        Value = rk_step(f,x[i-1],t[i-1],h,*args)
        x = np.vstack([x,Value])
   return t,x   

def solve_to(f,x0,t1,t2,h,solver,*args):
    """
   A function that uses methods of approximation to estimate the values of an ODE between a given timespan

   SYNTAX
    ------
    The function is called in the following way:

    solve_to(f,x0,t1,t2,h);

    The function will display a message asking what approximation method the user would like to use, for which they enter either euler or runge-kutta.
    If the user provides an incorrect input, the program will exit the function.

    WHERE:

        f: f is the ODE you wish to approximate, the form of the ode
        input should be as follows;

            def PredPrey(t,x0,a,b,d): Defining the ode as a function with inputs t x0, a, b and d.
            x = x0[0] 
            y = x0[1]
            dx_dt = x*(1-x) - (a*x*y)/(d+x)
            dy_dt = b*y*(1-(y/x))
            return [dx_dt, dy_dt] 
        
        x0: x0 is the value of the ODE for which you wish to iterate from, it's initial conditions. The form of this input should be as follows;

            x0 = [x1,x2,...,xi], where i is the number of dimensions the ode has

        t1 & t2 : t1 & t2 are the time values for which you wish to approximate your ODE between, in the form of integers or floating point numbers

        h: h is the value of the timestep for which you want to evaluate the ODE for, a smaller step size results in a more
        accurate approximation to the ODE. If h is too large, the function will display an error message explaining that the
        chosen stepsize is too large for an accurate approximation to the ODE.

        solver: solver is a string describing the type of approximation method you'd like to use, being either 'euler' or 'rk4.

    OUTPUT
    ------

    The solve_to function returns a 2 values as a tuple. It returns the approximated values of the independent variables within the timespan
    and the values for the time at which they were approximated. 
   """
    def euler_step(f,xn,t,h,*args):
        f = np.array(f(t,xn,*args))
        x = xn + h*f
        return x
    def solve_toEU(f,x0,t1,t2,h,*args):
        deltat_max = 0.1
        if h > deltat_max:
            raise ValueError("Step size too large for accurate approximation")
        t = np.arange(t1,t2+h,h)
        x = np.array([x0])
        for i in range(1,len(t)):
            Value = euler_step(f,x[i-1],t[i-1],h,*args)
            x = np.vstack([x, Value]) 
        return t,x  
    def rk_step(f,xn,t,h,*args):
        k1 = np.array(f(t,xn,*args))
        k2 = np.array(f(t+h/2,xn + h*k1/2,*args))
        k3 = np.array(f(t+h/2,xn+h*(k2/2),*args))
        k4 = np.array(f(t+h,xn+(h*k3),*args))
        x = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        return x    
    def solve_toRK(f,x0,t1,t2,h,*args):
        deltat_max = 0.25
        if h > deltat_max:
            raise ValueError("Step size too large for accurate approximation")
        t = np.arange(t1,t2+h,h)
        x = np.array([x0])
        for i in range(1,len(t)):
            Value = rk_step(f,x[i-1],t[i-1],h,*args)
            x = np.vstack([x,Value])
        return t,x  
    
    if callable(f) == False:
        raise TypeError("'f' must be a function")
    if not isinstance(x0,list):
        raise ValueError("'x0' must be a list the same length as there are dependent variables in your system")
    if not isinstance(t1,(int,float)):
        raise ValueError("'t1' must be either a float or integer value")
    if not isinstance(t2,(float,int)):
        raise ValueError("'t2' must be either a float or integer value")
    if t1 <0:
            raise ValueError('Time must be a positive integer/float')
    if t2 < t1:
        raise ValueError("This function iterates forwards in time, please provide a correct time interval")
    if solver == 'euler':
        t,x = solve_toEU(f,x0,t1,t2,h,*args)
    elif solver == 'rk4':
        t,x = solve_toRK(f,x0,t1,t2,h,*args)
    else:
        raise ValueError("Please provide a documented numerical approximation technique")
    return t,x
     
def shooting(x,ode,*args):
    """
    A function that uses the numerical shooting method in order to find the limit cycles of an ode

    SYNTAX
    ------
    The function is called the following way;

    shooting(x0,ode):

        WHERE:

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
    if callable(ode) == False:
        raise TypeError("'ode' must be a callable function")
    if not isinstance(x,(list)):
        raise ValueError("'x' must be of the form [x00,x01,...,x0n,T]")
    def shooting1(x,ode,*args):
        Condition1 = x[1:len(x)]-solve_ivp(ode,[0,x[0]],x[1:len(x)],args=([*args])).y[:,-1]
        Condition2 = ode(0,x[1:len(x)],*args)[0]
        return [*Condition1,Condition2]
    Result = root(shooting1,x0 = x,args=(ode,*args))
    if Result.success == False:
       raise ValueError('Periodic Orbit does not exist')
    return(Result.x)

def CubicContStep(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,10000)
    solutions = np.array([x0])
    p_value = np.array([p0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        predicted_value = solutions[-1]
        sol = root(f,x0=predicted_value,args=(p,))
        if sol.success == True:
            solutions = np.append(solutions,sol.x)
            p_value = np.append(p_value,p)
    v0 = np.array([p_value[1],solutions[1]]) 
    v1 = np.array([p_value[2],solutions[2]])
    return v0,v1

def Continuation(f,x0,p0,p1,type):
    """
    A function that performs parameter continuation on an arbitrary 1D function using either natural parameter continuation or psuedo-arc-length continuation

    SYNTAX
    ------
    The function is called in the following way:

    Continuation(f,x0,p0,p1,type)

        WHERE:

        f: 'f' is the function you'd like to perform parameter continuation on, f must be callable. The form of this input should be as follows:

            def f(x,c):
                return x**3 -x + c

        x0: 'x0' is the initial condition you'd like to start your continuation from, the form of this input is as follows:

            x0 = 1.0

        p0: 'p0' is the initial parameter value you'd like to start the continuation from. The form of this input should be as follows:

            p0 = -2.0

        p1: 'p1' is the final parameter value you'd like to perform the continuation to  

        type: 'type' is a string specifying the type of continuation you'd like to perform on the function 'f'. The form of this input should be as follows:

         type = 'natural'

    OUTPUT 
    ------
        The function returns the parameter values and their corresponding solutions as 2 numpy arrays of equal length.
    """
    def NumCont(f,x0,p0,p1):
        p_range = np.linspace(p0,p1,1000)
        solutions = np.array([x0])
        p_value = np.array([p0])
        for i in range(0,len(p_range)-1):
            p = p_range[i]
            predicted_value = solutions[-1]
            sol = root(f,x0=predicted_value,args=(p))
            if sol.success == True:
                solutions = np.append(solutions,sol.x)
                p_value = np.append(p_value,p)
        return p_value[1:],solutions[1:]
    
    def PseudoCont(f,x0,p0,p1):
        def conditions(input):
            Condition1=f(input[0],input[1])
            Condition2=np.dot(((input)-approx),secant)
            return [Condition1,Condition2]
        first,second = CubicContStep(f,x0,p0,p1)
        solutions = np.array([first[1],second[1]])
        p_value = np.array([first[0],second[0]])
        v0 = first
        v1 = second
        p = p0
        while p >=p0 and p<=p1:
            secant = v1-v0
            approx = v1+secant
            sol = root(conditions,x0=approx,tol=1e-6)
            v0 = v1
            v1 = np.array([sol.x[0],sol.x[1]])
            secant = v1-v0
            approx = v1+secant
            if sol.success == True:
                solutions = np.append(solutions,sol.x[1])
                p_value = np.append(p_value,sol.x[0])   
            p = sol.x[0]
        return p_value, solutions
    if callable(f) == False:
        raise TypeError("'f' must be a callable function")
    if not isinstance(x0,(int,float)):
        raise ValueError("'x0' must be either a decimal or integer")
    if not isinstance(p0,(float,int)):
        raise ValueError("p0 must be either a decimal or integer")
    if not isinstance(p1,(float,int)):
        raise ValueError("p0 must be either a decimal or integer")
    if p1 < p0:
        raise ValueError("'p1' must be greater than 'p0'")
    if type == 'natural':
        if not isinstance(type, str):
            raise ValueError("'type' must be a string describing the type of continuation you'd like to perform, enter either 'natural' or 'psuedo'")
        x,y = NumCont(f,x0,p0,p1)
    elif type == 'pseudo':
        if not isinstance(type, str):
            raise ValueError("'type' must be a string describing the type of continuation you'd like to perform, enter either 'natural' or 'psuedo'")
        x,y = PseudoCont(f,x0,p0,p1)
    else:
        raise ValueError("Invalid input, please enter either 'natural' or 'pseudo'")
    return x,y 
