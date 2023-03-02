# %%
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy
# %%
def euler_step(f,xn,t,h):
    """
    A function that uses the euler approximation method to find a single step of the solution

    SYNTAX
    ------
    The function is called in the following way:

    euler_step(f,xn,t,h);

    WHERE:

        f: f is the ODE you wish to approximate, the form of the ode
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
        
        xn: xn is the value of the ODE for which you wish to iterate a step further, the form of this input should be as follows;

            xn = [x1,x2,...,xi], where i is the number of dimensions the ode has

        t: t is the time value for which you wish to approximate your ODE at, in the form of an integer or floating point number

        h: h is the value of the timestep for which you want to evaluate the ODE after, a smaller step size results in a more
        accurate approximation to the ODE

    OUTPUT
    ------

    The euler_step function returns a numpy array containing an approximation to the ODE at a time t+h

    """
    f = np.array(f(t,xn))
    x = xn + h*f
    return x
# %%
def solve_toEU(f,x0,t1,t2,h):
   """
   A function that uses the euler method of approximation to estimate the values of an ODE between a given timespan

   SYNTAX
    ------
    The function is called in the following way:

    solve_toEU(f,x0,t1,t2,h);

    WHERE:

        f: f is the ODE you wish to approximate, the form of the ode
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
        
        x0: x0 is the value of the ODE for which you wish to iterate from, it's initial conditions. The form of this input should be as follows;

            x0 = [x1,x2,...,xi], where i is the number of dimensions the ode has

        t1 & t2 : t1 & t2 are the time values for which you wish to approximate your ODE between, in the form of integers or floating point numbers

        h: h is the value of the timestep for which you want to evaluate the ODE for, a smaller step size results in a more
        accurate approximation to the ODE. If h is too large, the function will display an error message explaining that the
        chosen stepsize is too large for an accurate approximation to the ODE.

    OUTPUT
    ------

    The solve_toEU function returns a 2 values as a tuple. It returns the approximated values of the independent variables within the timespan
    and the values for the time at which they were approximated. 
   """
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toEU)
   t = np.arange(t1,t2+h,h)
   x = np.array([x0])
   for i in range(1,len(t)):
        Value = euler_step(f,x[i-1],t[i-1],h)
        x = np.vstack([x, Value]) 
   return t,x 
# %%
def rk_step(f,xn,t,h):
    """
    A function that uses the 4th order Runge-Kutta approximation method to find a single step of the solution

    SYNTAX
    ------
    The function is called in the following way:

    rk_step(f,xn,t,h);

    WHERE:

        f: f is the ODE you wish to approximate, the form of the ode
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
        
        xn: xn is the value of the ODE for which you wish to iterate a step further, the form of this input should be as follows;

            xn = [x1,x2,...,xi], where i is the number of dimensions the ode has

        t: t is the time value for which you wish to approximate your ODE at, in the form of an integer or floating point number

        h: h is the value of the timestep for which you want to evaluate the ODE after, a smaller step size results in a more
        accurate approximation to the ODE

    OUTPUT
    ------

    The rk_step function returns a numpy array containing an approximation to the ODE at a time t+h
    """
    k1 = np.array(f(t,xn))
    k2 = np.array(f(t+h/2,xn + h*k1/2))
    k3 = np.array(f(t+h/2,xn+h*(k2/2)))
    k4 = np.array(f(t+h,xn+(h*k3)))
    x = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x
"Creating the function for approximation using RK4 method"
# %%
def solve_toRK(f,x0,t1,t2,h):
   """
   A function that uses the 4th order Runge-Kutta method of approximation to estimate the values of an ODE between a given timespan

   SYNTAX
    ------
    The function is called in the following way:

    solve_toRK(f,x0,t1,t2,h);

    WHERE:

        f: f is the ODE you wish to approximate, the form of the ode
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
        
        x0: x0 is the value of the ODE for which you wish to iterate from, it's initial conditions. The form of this input should be as follows;

            x0 = [x1,x2,...,xi], where i is the number of dimensions the ode has

        t1 & t2 : t1 & t2 are the time values for which you wish to approximate your ODE between, in the form of integers or floating point numbers

        h: h is the value of the timestep for which you want to evaluate the ODE for, a smaller step size results in a more
        accurate approximation to the ODE. If h is too large, the function will display an error message explaining that the
        chosen stepsize is too large for an accurate approximation to the ODE.

    OUTPUT
    ------

    The solve_toRK function returns a 2 values as a tuple. It returns the approximated values of the independent variables within the timespan
    and the values for the time at which they were approximated. 
   """
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toRK)
   t = np.arange(t1,t2+h,h)
   x = np.array([x0])
   for i in range(1,len(t)):
        Value = rk_step(f,x[i-1],t[i-1],h)
        x = np.vstack([x,Value])
   return t,x   
# %%
def solve_to(f,x0,t1,t2,h):
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

            def PredPrey(t,x0): Defining the ode as a function with inputs t and x0, time and initial conditions
            a = 1     |
            b = 0.1   |  Defining any parameters inside the function handle, you could leave them out of the function but you
            d = 0.1   |  would have to define them as arguments when calling the function
            x = x0[0] Splitting the initial conditions array into its constituent parts
            y = x0[1]
            dx_dt = x*(1-x) - (a*x*y)/(d+x) #Defining the differential equations
            dy_dt = b*y*(1-(y/x))
            return [dx_dt, dy_dt] #Returns the value of the ode as an array 
        
        x0: x0 is the value of the ODE for which you wish to iterate from, it's initial conditions. The form of this input should be as follows;

            x0 = [x1,x2,...,xi], where i is the number of dimensions the ode has

        t1 & t2 : t1 & t2 are the time values for which you wish to approximate your ODE between, in the form of integers or floating point numbers

        h: h is the value of the timestep for which you want to evaluate the ODE for, a smaller step size results in a more
        accurate approximation to the ODE. If h is too large, the function will display an error message explaining that the
        chosen stepsize is too large for an accurate approximation to the ODE.

    OUTPUT
    ------

    The solve_to function returns a 2 values as a tuple. It returns the approximated values of the independent variables within the timespan
    and the values for the time at which they were approximated. 
   """
    deltat_max = 1
    if h >= deltat_max:
        print("Step size too large for accurate approximation")
        exit(solve_to)
    method = input("Which approximation method would you like to use? Please enter either Euler or Runge-Kutta ").lower()
    if method == ("Euler").lower():
        t,x = solve_toEU(f,x0,t1,t2,h)
        return t,x
    elif method == ("Runge-Kutta").lower():
        t,x = solve_toRK(f,x0,t1,t2,h)
        return t,x
    else:
        print("Please provide a correct input")
        exit(solve_to)
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
