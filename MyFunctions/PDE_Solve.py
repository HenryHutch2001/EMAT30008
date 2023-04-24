import numpy as np
from scipy.optimize import root
from math import ceil
from MyFunctions.ODE_Solve import solve_toRK

def CreateAandb(N,a,b,bc_type,bc_left,bc_right):
    """
        A function that generates matrices related to different boundary conditions for use in matrix-vector notation operations
    
    SYNTAX
    ------
    The function is called in the following way:

    CreateAandb(N,a,b,bc_type,bc_left,bc_right)

    WHERE:
    N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

    a/b: 'a' and 'b' are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            a = 0
            b = 1
    
    bc_type: 'bc_type' is a string describing the type of boundary conditions imposed on the equation, 
        inputs are either 'dirichlet', 'neumann' or 'robin'. The form of this input should be as follows;

            bc_type = 'neumann'

    alpha: 'alpha' is a decimal value defining the left boundary condition. f(a,t) = alpha.  
        The form of this input should be as follows;

            alpha = 0.0

    beta: 'beta' is a decimal value defining the right boundary condition. f(b,t) = beta.  
        The form of this input should be as follows;

            beta = 1.0

    OUTPUT
    ------

    The function outputs 4 variables; 2 numpy arrays containing the relevant matrix-vector formulations, 
    the interior gridpoints for which a solution would be solved for, and 'dx', the relevant step in the spatial domain related to a,b and N.

    """
    def CreateAandbDirichlet(N,a,b,bc_left,bc_right):
        A = np.zeros((N-1,N-1))
        dx = (b-a)/N
        for i in range(0,N-1):
            A[i,i] = -2
        for i in range(0,N-2):
            A[i,i+1] = A[i,i-1] = 1
        A[N-2,N-3] = 1
        B = np.zeros(N-1)
        B[0] = bc_left
        B[N-2] = bc_right
        return A,B.T,dx
    
    def CreateAandbNeumann(N,a,b,bc_left,bc_right):
        A = np.zeros((N,N))
        dx = (b-a)/N
        for i in range(0,N):
            A[i,i] = -2
        for i in range(0,N-1):
            A[i,i+1] = A[i,i-1] = 1
        A[N-1,N-2] = 2
        B = np.zeros(N)
        B[0] = bc_left
        B[N-1] = 2*bc_right*dx
        return A,B.T,dx
    
    def CreateAandbRobin(N,a,b,bc_left,bc_right):
        A = np.zeros((N,N))
        dx = (b-a)/N
        beta,gamma = bc_right
        for i in range(0,N-1):
            A[i,i] = -2
        A[N-1,N-1] = -2*(1+gamma*dx)
        for i in range(0,N-1):
            A[i,i+1] = A[i,i-1] = 1
        A[N-1,N-2] = 2
        B = np.zeros(N)
        B[0] = bc_left
        B[N-1] = 2*beta*dx
        return A,B.T,dx
    GridSpace = np.linspace(a,b,N+1)
    if bc_type == 'dirichlet':
        x_ints = GridSpace[1:-1]
        A_dd,b_dd,dx=CreateAandbDirichlet(N,a,b,bc_left,bc_right)
    elif bc_type == 'neumann':
        x_ints = GridSpace[1:]
        A_dd,b_dd,dx=CreateAandbNeumann(N,a,b,bc_left,bc_right)
    elif bc_type == 'robin':
        x_ints = GridSpace[1:]
        A_dd,b_dd,dx=CreateAandbRobin(N,a,b,bc_left,bc_right)
    else:
        raise ValueError("Boundary condition type is invalid, please input 'dirichlet','neumann' or 'robin'")
    return A_dd,b_dd,x_ints,dx

def FiniteDifference(source,D,N,bc_type,alpha,beta,*args):
    """
    A function that uses the finite difference method to provide a numerical approximation to the Diffusion equation

    SYNTAX
    ------
    The function is called in the following way:

    CreateAandb(N,a,b,bc_type,bc_left,bc_right)

    WHERE:
    N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

    a/b: 'a' and 'b' are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            a = 0
            b = 1
    
    bc_type: 'bc_type' is a string describing the type of boundary conditions imposed on the equation, 
        inputs are either 'dirichlet', 'neumann' or 'robin'. The form of this input should be as follows;

            bc_type = 'neumann'

    alpha: 'alpha' is a decimal value defining the left boundary condition. f(a,t) = alpha.  
        The form of this input should be as follows;

            alpha = 0.0

    beta: 'beta' is a decimal value defining the right boundary condition. f(b,t) = beta.  
        The form of this input should be as follows;

            beta = 1.0
    OUTPUT
    ------

    The function outputs an array containing the numerical approximation to the solution, and the interior gridpoints of the domain they correspond to.
    
    """

    def FiniteDifferenceRobin(source,D,N,alpha,betagamma,*args):
        a = 0 
        b = 1
        GridSpace = np.linspace(a,b,N+1)
        Interior = GridSpace[1:]
        dx = (b-a)/N
        Guess = Interior * 0.5
        beta,gamma = betagamma
        def SolveSource(u):
            F = np.zeros(N)
            F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+source(Interior[0],F[0],*args) #First internal gridpoint
            for i in range(1,N-1):
                F[i] = F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+source(Interior[i],F[i],*args)
            F[N-1] = D*(((-2*(1+gamma*dx)*u[N-1])+(2*u[N-2]))/(dx**2) + (2*beta)/dx + source(Interior[N-1],F[N-1],*args))
            return F
        Solution = root(SolveSource, x0=Guess)
        return Interior, Solution.x
    
    def FiniteDifferenceNeumann(source,D,N,alpha,beta,*args): 
        a = 0
        b = 1
        GridSpace = np.linspace(a,b,N+1)
        Interior = GridSpace[1:]
        Guess = Interior*0.5
        dx = (b-a)/N
        def SolveSource(u):
            F = np.zeros(N)
            F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+source(Interior[0],F[0],*args) #First internal gridpoint
            for i in range(1,N-1): 
                F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+source(Interior[i],F[i],*args)
            F[N-1] = D*((-2*(u[N-1])+2*u[N-2])/(dx**2)+((2*beta)/dx) +source(Interior[N-1],F[N-1],*args))
            return F
        Solution = root(SolveSource,x0 = Guess)
        return Interior,Solution.x
    
    def FiniteDifferenceDirichlet(source,D,N,alpha,beta,*args): 
        a = 0
        b = 1
        GridSpace = np.linspace(a,b,N+1)
        dx = (b-a)/N 
        Interior = GridSpace[1:-1]
        Guess = Interior * 0.5
        def SolveSource(u):
            F = np.zeros(N-1)
            F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+source(Interior[0],F[0],*args)
            for i in range(1,N-2):
                F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+source(Interior[i],F[i],*args)
            F[N-2] = D*((beta-2*u[N-2]+u[N-3])/(dx**2))+source(Interior[N-2],F[N-2],*args)
            return F
        Solution = root(SolveSource,x0 = Guess) 
        return Interior,Solution.x

    if N < 5:
        raise ValueError("'N' must be greater than 5 to provide an accurate representation of the solution")
    if not isinstance(N, int):
        raise ValueError("'N' must be an integer value")
    if N > 200:
        raise RuntimeWarning("The size of 'N' leads to large computational compexity")
    if not isinstance(D,(float,int)):
        raise ValueError("'D' must be either a positive decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    if not isinstance(alpha,float):
        raise ValueError("'alpha' must be a decimal value")
    if source == 'none':
        def source(x,*args):
            return 0
    elif callable(source) == False:
        raise TypeError("'source' must be a callable function")
    if bc_type == 'dirichlet':
        if not isinstance(alpha,float):
            raise ValueError("'alpha' must be a floating point number")
        if not isinstance(beta,float):
            raise ValueError("'beta' must be a floating point number")
        Interior, Solution = FiniteDifferenceDirichlet(source,D,N,alpha,beta,*args)
    elif bc_type == 'neumann':
        if not isinstance(alpha,float):
            raise ValueError("'alpha' must be a floating point number")
        if not isinstance(beta,float):
            raise ValueError("'beta' must be a floating point number")
        Interior,Solution = FiniteDifferenceNeumann(source,D,N,alpha,beta,*args)
    elif bc_type == 'robin':
        if not isinstance(alpha,float):
            raise ValueError("'alpha' must be a floating point number")
        if not isinstance(beta,list):
            raise ValueError("'beta' must be a list")
        Interior,Solution = FiniteDifferenceRobin(source,D,N,alpha,beta,*args)
    else:
        raise ValueError("Boundary condition type is invalid, please input 'dirichlet','neumann' or 'robin'")
    return Interior,Solution

def ExplicitEuler(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
    """
        A function that uses Explicit Euler method to approximate the solution to the diffusion equation with Dirichlet boundary conditions

    SYNTAX 
    ------
    The function is called in the following way:

    ImplicitEuler(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)

    WHERE:

        N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

        domain_start/domain_end: 'domain_start' and 'domain_end' are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            domain_start = 0
            domain_end = 1

        D: 'D' is an integer/floating point value describing the diffusion coefficient of the diffusion equation. 
        The form of this input should be as follows;

            D = 0.5

        t_final: 't_final' is an integer/floating point number defining the time that the approximation solves up to, starting at zero. 
        The form of this input should be as follows;

            t_final = 1

        dt: 'dt' is a decimal value defining the time step size, the time domain is split into sections equal 
        to the size of dt and the solition is calculated at each point. The form of this input should be as follows;

            dt = 0.1

        bc_left: 'bc_left' is a decimal value defining the left boundary condition. f(domain_start,t) = bc_left.  
        The form of this input should be as follows;

            bc_left = 0.0

        bc_right: 'bc_right' is a decimal value defining the right boundary condition. f(domain_end,t) = bc_right.  
        The form of this input should be as follows;

            bc_right = 1.0

        initial_condition: 'initial_condition' is a function imposing an initial condition on the solution, 
        if no initial condition is wanted, input 'none' and the function will remove any condition. 
        The form of this input should be as follows;

            def f(x,*args):
                return np.ones(size(x))

        source: 'source' is a function describing the source term 'q(x,t,u;parameters)' 

            def source(x,t,u):
                return np.sin(math.pi*x)
        
    OUTPUT
    ------

    Returns 2 numpy arrays of equal lengths, with the first array containing the interior 
    gridpoints and the second containing the solutions for these gridpoints.

    """
    if not isinstance(N, int) or N <= 5:
        raise ValueError("N must be a positive integer greater than 5 to provide an accurate representation of the solution")
    if N > 200:
        raise RuntimeWarning("The size of 'N' leads to large computational compexity")
    if not isinstance(domain_start,int):
        raise ValueError("'domain_start' must be an integer")
    if domain_start > domain_end:
        raise ValueError("The starting point of the domain must be before the endpoint")
    if not isinstance(domain_end,int):
        raise ValueError("'domain_end' must be an integer value")
    if not isinstance(D,(float,int)):
        raise ValueError("'D' must be either a decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    if not isinstance(t_final,(float,int)):
        raise ValueError("'t_final' must be either a decimal or integer")
    if t_final < 0:
        raise ValueError("'t_final' must be a positive value")
    if not isinstance(dt,float):
        raise ValueError("'dt' must be a decimal value") 
    if not isinstance(bc_left,float):
        raise ValueError("'bc_left must be a decimal value")
    if not isinstance(bc_right,float):
        raise ValueError("'bc_right must be a decimal value")
    if initial_condition == 'none':
        def initial_condition(x,*args):
            return 0
    elif callable(initial_condition) == False:
        raise TypeError("'initial_condition' must be a callable function")
    if source == 'none':
        def source(x,*args):
            return 0
    elif callable(source) == False:
        raise TypeError("'source' must be a callable function")
    a = domain_start 
    b = domain_end 
    GridSpace = np.linspace(a,b,N+1) 
    x_int = GridSpace[1:-1]
    dx = (b-a)/N
    C = dt*D/(dx**2) 
    if C > 0.5:
        raise ValueError("The stability condition is not satisfied, please use a smaller time step")
    N_t = ceil(t_final/dt) 
    t = dt * np.arange(N_t) 
    u = np.zeros((N_t+1,N-1))
    u[0,:] = initial_condition(x_int)
    for n in range(0,N_t):
        for i in range(0,N-1):
            if i == 0:
                u[n+1,0] = u[n,0] + C * (bc_left-2*u[n,0]+u[n,1]) + dt*source(x_int[i],t[n],u[n,0],*args)
            elif i < N-2:
                u[n+1,i] = u[n,i] + C * (u[n,i+1] - 2 * u[n,i] + u[n,i-1]) + dt*source(x_int[i],t[n],u[n,i],*args)
            else:
                u[n+1,N-2] = u[n, N-2] + C * (bc_right - 2 * u[n, N-2] + u[n, N-3]) + dt*source(x_int[N-3],t[n],u[n,N-2],*args)
    return x_int, u[-1,:]

def ImplicitEuler(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    """
        A function that uses Implicit Euler method to approximate the solution to the diffusion equation

    SYNTAX 
    ------
    The function is called in the following way:

    ImplicitEuler(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)

    WHERE:

        N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

        domain_start/domain_end: 'domain_start' and 'domain_end' are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            domain_start = 0
            domain_end = 1

        D: 'D' is an integer/floating point value describing the diffusion coefficient of the diffusion equation. 
        The form of this input should be as follows;

            D = 0.5

        t_final: 't_final' is an integer/floating point number defining the time that the approximation solves up to, starting at zero. 
        The form of this input should be as follows;

            t_final = 1

        dt: 'dt' is a decimal value defining the time step size, the time domain is split into sections equal 
        to the size of dt and the solition is calculated at each point. The form of this input should be as follows;

            dt = 0.1

        bc_type: 'bc_type' is a string describing the type of boundary conditions imposed on the equation, 
        inputs are either 'dirichlet', 'neumann' or 'robin'. The form of this input should be as follows;

            bc_type = 'neumann'

        bc_left: 'bc_left' is a decimal value defining the left boundary condition. f(domain_start,t) = bc_left.  
        The form of this input should be as follows;

            bc_left = 0.0

        bc_right: 'bc_right' is a decimal value defining the right boundary condition. f(domain_end,t) = bc_right.  
        The form of this input should be as follows;

            bc_right = 1.0

        initial_condition: 'initial_condition' is a function imposing an initial condition on the solution, 
        if no initial condition is wanted, input 'none' and the function will remove any condition. 
        The form of this input should be as follows;

            def f(x,*args):
                return np.ones(size(x))

        source: 'source' is a function describing the source term 'q(x,t,u;parameters)' 

            def source(x,t,u):
                return np.sin(math.pi*x)
        
    OUTPUT
    ------

    Returns 2 numpy arrays of equal lengths, with the first array containing the interior 
    gridpoints and the second containing the solutions for these gridpoints.

    """
    def ImplicitDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
    # Define the matrix for the implicit Euler method
        a = domain_start
        b = domain_end
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
        C = dt*D/(dx**2)
        dt_max = dx**2/(2*D)
        if dt > dt_max:
            raise ValueError("Time step is too large to satisfy stability condition.")
        N_t = ceil(t_final/dt)

        u = np.zeros((N_t+1, N-1))
        u[0,:] = initial_condition(x_int)
        M = np.identity(N-1)-C*A_dd
        for n in range(0,N_t):
            b = u[n] + C*b_dd + dt*source(x_int,(n+1)*dt,u[n],*args)
            u[n+1] = np.linalg.solve(M,b)
        return x_int,u[-1,:]

    def ImplicitNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
        a = domain_start
        b = domain_end
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
        C = dt*D/(dx**2)
        dt_max = dx**2/(2*D)
        if dt > dt_max:
            raise ValueError("Time step is too large to satisfy stability condition.")
        N_t = ceil(t_final/dt)
        u = np.zeros((N_t+1, N))
        u[0,:] = initial_condition(x_int)
        M = np.identity(N)-C*A_dd
        for n in range(0,N_t):
            b = u[n] + C*b_dd + dt*source(x_int,(n+1)*dt,u[n],*args)
            u[n+1] = np.linalg.solve(M,b)
        return x_int,u[-1,:]

    def ImplicitRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
        a = domain_start
        b = domain_end
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
        C = dt*D/(dx**2)
        dt_max = dx**2/(2*D)
        if dt > dt_max:
            raise ValueError("Time step is too large to satisfy stability condition.")
        N_t = ceil(t_final/dt)
        u = np.zeros((N_t+1, N))
        u[0,:] = initial_condition(x_int)
        M = np.identity(N)-C*A_dd
        for n in range(0,N_t):
            b = u[n] + C*b_dd + dt*source(x_int,(n+1)*dt,u[n],*args)
            u[n+1] = np.linalg.solve(M,b)
        return x_int,u[-1,:]

    if N < 5:
        raise ValueError("'N' must be greater than 5 to provide an accurate representation of the solution")
    if not isinstance(N, int):
        raise ValueError("'N' must be an integer value")
    if N > 300:
        raise RuntimeWarning("The size of 'N' leads to large computational compexity")
    if not isinstance(domain_start,int):
        raise ValueError("'domain_start' must be an integer")
    if domain_start > domain_end:
        raise ValueError("The starting point of the domain must be before the endpoint")
    if not isinstance(domain_end,int):
        raise ValueError("'domain_end' must be an integer value")
    if not isinstance(D,(float,int)):
        raise ValueError("'D' must be either a decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    if not isinstance(t_final,(float,int)):
        raise ValueError("'t_final' must be either a decimal or integer")
    if t_final < 0:
        raise ValueError("'t_final' must be a positive value")
    if not isinstance(dt,float):
        raise ValueError("'dt' must be a decimal value") 
    if not isinstance(bc_left,float):
        raise ValueError("'bc_left must be a decimal value")
    if initial_condition == 'none':
        def initial_condition(x,*args):
            return 0
    elif callable(initial_condition) == False:
        raise TypeError("'initial_condition' must be a callable function")
    if source == 'none':
        def source(x,*args):
            return 0
    elif callable(source) == False:
        raise TypeError("'source' must be a callable function")
    if bc_type == 'dirichlet':
        if not isinstance(bc_right,float):
            raise ValueError("'bc_right' must be a decimal value")
        x,y = ImplicitDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    elif bc_type == 'neumann':
        if not isinstance(bc_right,float):
            raise ValueError("'bc_right' must be a decimal value")
        x,y = ImplicitNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    elif bc_type == 'robin':
        if not isinstance(bc_right,list):
            raise ValueError("'bc_right' must be a list containing both beta and gamma values")
        ImplicitRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    else:
        raise ValueError("'bc_type' must be either 'dirichlet', 'neumann' or 'robin'")
    return x,y 

def CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    """
    A function that uses Crank-Nicolson method to approximate the solution to the diffusion equation

    SYNTAX 
    ------
    The function is called in the following way:

    CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)

    WHERE:

        N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

        domain_start/domain_end: 'domain_start' and 'domain_end are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            domain_start = 0
            domain_end = 1

        D: 'D' is an integer/floating point value describing the diffusion coefficient of the diffusion equation. 
        The form of this input should be as follows;

            D = 0.5

        t_final: 't_final' is an integer/floating point number defining the time that the approximation solves up to, starting at zero. 
        The form of this input should be as follows;

            t_final = 1

        dt: 'dt' is a decimal value defining the time step size, the time domain is split into sections equal 
        to the size of dt and the solition is calculated at each point. The form of this input should be as follows;

            dt = 0.1
        
        bc_type: 'bc_type' is a string describing the type of boundary conditions imposed on the equation, 
        inputs are either 'dirichlet', 'neumann' or 'robin'. The form of this input should be as follows;

            bc_type = 'neumann'

        bc_left: 'bc_left' is a decimal value defining the left boundary condition. f(domain_start,t) = bc_left.  
        The form of this input should be as follows;

            bc_left = 0.0

        bc_right: 'bc_right' is a decimal value defining the right boundary condition. f(domain_end,t) = bc_right.  
        The form of this input should be as follows;

            bc_right = 1.0

        initial_condition: 'initial_condition' is a function imposing an initial condition on the solution, 
        if no initial condition is wanted, input 'none' and the function will remove any condition. 
        The form of this input should be as follows;

            def f(x,*args):
                return np.ones(size(x))

        source: 'source' is a function describing the source term 'q(x,t,u;parameters)' 

            def source(x,t,u):
                return np.sin(math.pi*x)
        
    OUTPUT
    ------

    Returns 2 numpy arrays of equal lengths, with the first array containing the interior 
    gridpoints and the second containing the solutions for these gridpoints.

    """
    def CrankNicholsonDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):  
        a = domain_start
        b = domain_end
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
        C = dt*D/(dx**2)
        N_t = ceil(t_final/dt) 
        u = np.zeros((N_t+1, N-1))
        u[0,:] = initial_condition(x_int)
        LHS = ((np.identity(N-1))-((C/2)*A_dd))
        for n in range(0,N_t):
            RHS = ((np.identity(N-1))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt,u[n],*args)
            u[n+1] = np.linalg.solve(LHS,RHS)
        return x_int,u[-1,:]

    def CrankNicholsonNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
        a = domain_start
        b = domain_end
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
        C = dt*D/(dx**2)
        N_t = ceil(t_final/dt) 
        u = np.zeros((N_t+1, N))
        u[0,:] = initial_condition(x_int)
        LHS = ((np.identity(N))-((C/2)*A_dd))
        for n in range(0,N_t):
            RHS = ((np.identity(N))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt,u[n],*args)
            u[n+1] = np.linalg.solve(LHS,RHS)
        return x_int,u[-1,:]
    
    def CrankNicholsonRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
        a = domain_start
        b = domain_end
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
        C = dt*D/(dx**2)
        N_t = ceil(t_final/dt) 
        u = np.zeros((N_t+1, N))
        u[0,:] = initial_condition(x_int)
        LHS = ((np.identity(N))-((C/2)*A_dd))
        for n in range(0,N_t):
            RHS = ((np.identity(N))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt,u[n],*args)
            u[n+1] = np.linalg.solve(LHS,RHS)
        return x_int,u[-1,:]

    if N < 5:
        raise ValueError("'N' must be greater than 5 to provide an accurate representation of the solution")
    if not isinstance(N, int):
        raise ValueError("'N' must be an integer value")
    if N > 300:
        raise RuntimeWarning("The size of 'N' leads to large computational compexity")
    if not isinstance(domain_start,int):
        raise ValueError("'domain_start' must be an integer")
    if domain_start > domain_end:
        raise ValueError("The starting point of the domain must be before the endpoint")
    if not isinstance(domain_end,int):
        raise ValueError("'domain_end' must be an integer value")
    if not isinstance(D,(float,int)):
        raise ValueError("'D' must be either a decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    if not isinstance(t_final,(float,int)):
        raise ValueError("'t_final' must be either a decimal or integer")
    if t_final < 0:
        raise ValueError("'t_final' must be a positive value")
    if not isinstance(dt,float):
        raise ValueError("'dt' must be a decimal value") 
    if not isinstance(bc_left,float):
        raise ValueError("'bc_left must be a decimal value")
    if initial_condition == 'none':
        def initial_condition(x,*args):
            return 0
    elif callable(initial_condition) == False:
        raise TypeError("'initial_condition' must be a callable function")
    if source == 'none':
        def source(x,*args):
            return 0
    elif callable(source) == False:
        raise TypeError("'source' must be a callable function")
    if bc_type == 'dirichlet':
        if not isinstance(bc_right,float):
            raise ValueError("'bc_right' must be a decimal value")
        x,y = CrankNicholsonDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    elif bc_type == 'neumann':
        if not isinstance(bc_right,float):
            raise ValueError("'bc_right' must be a decimal value")
        x,y = CrankNicholsonNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    elif bc_type == 'robin':
        if not isinstance(bc_right,list):
            raise ValueError("'bc_right' must be a list containing both beta and gamma values")
        CrankNicholsonRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    else:
        raise ValueError("'bc_type' must be either 'dirichlet', 'neumann' or 'robin'")
    return x,y 

def RKSolver(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    """
    A function that uses Runge-Kutta method to approximate the solution to the diffusion equation after its discretisation into a system of ODEs.

    SYNTAX 
    ------
    The function is called in the following way:

    CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)

    WHERE:

        N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

        domain_start/domain_end: 'domain_start' and 'domain_end are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            domain_start = 0
            domain_end = 1

        D: 'D' is an integer/floating point value describing the diffusion coefficient of the diffusion equation. 
        The form of this input should be as follows;

            D = 0.5

        t_final: 't_final' is an integer/floating point number defining the time that the approximation solves up to, starting at zero. 
        The form of this input should be as follows;

            t_final = 1

        dt: 'dt' is a decimal value defining the time step size, the time domain is split into sections equal 
        to the size of dt and the solition is calculated at each point. The form of this input should be as follows;

            dt = 0.1
        
        bc_type: 'bc_type' is a string describing the type of boundary conditions imposed on the equation, 
        inputs are either 'dirichlet', 'neumann' or 'robin'. The form of this input should be as follows;

            bc_type = 'neumann'

        bc_left: 'bc_left' is a decimal value defining the left boundary condition. f(domain_start,t) = bc_left.  
        The form of this input should be as follows;

            bc_left = 0.0

        bc_right: 'bc_right' is a decimal value defining the right boundary condition. f(domain_end,t) = bc_right.  
        The form of this input should be as follows;

            bc_right = 1.0

        initial_condition: 'initial_condition' is a function imposing an initial condition on the solution, 
        if no initial condition is wanted, input 'none' and the function will remove any condition. 
        The form of this input should be as follows;

            def f(x,*args):
                return np.ones(size(x))

        source: 'source' is a function describing the source term 'q(x,t,u;parameters)' 

            def source(x,t,u):
                return np.sin(math.pi*x)
        
    OUTPUT
    ------

    Returns 2 numpy arrays of equal lengths, with the first array containing the interior 
    gridpoints and the second containing the solutions for these gridpoints.

    """
    if N < 5:
        raise ValueError("'N' must be greater than 5 to provide an accurate representation of the solution")
    if not isinstance(N, int):
        raise ValueError("'N' must be an integer value")
    if N > 300:
        raise RuntimeWarning("The size of 'N' leads to large computational compexity")
    if not isinstance(domain_start,int):
        raise ValueError("'domain_start' must be an integer")
    if domain_start > domain_end:
        raise ValueError("The starting point of the domain must be before the endpoint")
    if not isinstance(domain_end,int):
        raise ValueError("'domain_end' must be an integer value")
    if not isinstance(D,(float,int)):
        raise ValueError("'D' must be either a decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    if not isinstance(t_final,(float,int)):
        raise ValueError("'t_final' must be either a decimal or integer")
    if t_final < 0:
        raise ValueError("'t_final' must be a positive value")
    if not isinstance(dt,float):
        raise ValueError("'dt' must be a decimal value") 
    if not isinstance(bc_left,float):
        raise ValueError("'bc_left must be a decimal value")
    if initial_condition == 'none':
        def initial_condition(x,*args):
            return 0
    elif callable(initial_condition) == False:
        raise TypeError("'initial_condition' must be a callable function")
    if source == 'none':
        def source(x,*args):
            return 0
    elif callable(source) == False:
         raise TypeError("'source' must be a callable function")
    a = domain_start
    b = domain_end
    if bc_type == 'dirichlet':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
    elif bc_type == 'neumann':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
    elif bc_type == 'robin':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
    def system(t,u,source,*args):
        du_dt = (D/(dx**2))*(A_dd*u-b_dd-(dx**2)*source(t,x_int,u,*args))
        return du_dt
    t,u = solve_toRK(system,initial_condition(x_int),0,t_final,dt,source,*args)
    return x_int,u[0,:],t

def SolvePDE(Method,N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    if Method == 'implicit euler':
        x,y = ImplicitEuler(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)
    elif Method == 'crank nicolson':
        x,y = CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)
    elif Method == 'rk4':
        x,y = RKSolver(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)
    return x,y