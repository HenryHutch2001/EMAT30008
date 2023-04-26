# %% Importing packages 
from MyFunctions.ODE_Solve import shooting,solve_to,solve_toEU,solve_toRK, Continuation
import pytest
# %%
import numpy as np


def ode(t,y): 
    dx_dt = y[1]
    dy_dt = -y[0]
    return [dx_dt, dy_dt]

ode1 = 5

def ode2(t,y,a,b,d): 
    x = y[0]
    y = y[1]
    dx_dt = x*(1-x) - (a*x*y)/(d+x) 
    dy_dt = b*y*(1-(y/x))
    return [dx_dt, dy_dt] 

def function(x,c):
    return x**3-x+c

def test_EU():
    output = solve_toEU(ode,[1,1],0,1,0.1)
    assert isinstance(output, tuple)
    with pytest.raises(ValueError):
        solve_toEU(ode,[1,1],-1,1,0.1)
    with pytest.raises(ValueError):
        solve_toEU(ode,[1,1],0,1,0.5)
    with pytest.raises(ValueError):
        solve_toEU(ode,[1,1],1,0,0.1)

def test_output_typeRK():
    output = solve_toRK(ode,[1,1],0,1,0.1)
    assert isinstance(output, tuple)
    with pytest.raises(ValueError):
        solve_toRK(ode,[1,1],-1,1,0.1)
    with pytest.raises(ValueError):
        solve_toRK(ode,[1,1],0,1,0.5)
    with pytest.raises(ValueError):
        solve_toRK(ode,[1,1],1,0,0.1)

def test_SolveTo():
    t,x = solve_to(ode,[1,1],0,1,0.01,'rk4')
    assert isinstance(t,np.ndarray)
    assert isinstance(x,np.ndarray)
    with pytest.raises(ValueError):
        solve_to(ode,(1,1),0,1,0.01,'rk4')
        solve_to(ode,[1,1],'test',1,0.01,'rk4')
        solve_to(ode,[1,1],0,'test',0.01,'rk4')
        solve_to(ode,[1,1],1,0,0.01,'rk4')
        solve_to(ode,[1,1],0,1,0.2,'euler')
        solve_to(ode,[1,1],0,1,0.3,'rk4')

    with pytest.raises(TypeError):
        solve_to(ode1,[1,1],0,1,0.01,'rk4')

def test_Shooting():
    x = shooting([20,1,2],ode2,1,0.2,0.1)
    assert isinstance(x,np.ndarray)
    with pytest.raises(ValueError):
        x = shooting('test',ode2,1,0.2,0.1)
    with pytest.raises(TypeError):
        x = shooting([20,1,2],ode1,1,0.2,0.1)

def test_Continuation():
    x,y = Continuation(function,2,-2,2,'natural')
    assert isinstance(x,np.ndarray)
    assert isinstance(y,np.ndarray)
    with pytest.raises(ValueError):
        Continuation(function,'test',-2,2,'natural')
        Continuation(function,2,2,-2,'natural')
        Continuation(function,2,-2,2,'test')
        Continuation(function,2,'test',2,'natural')
        Continuation(function,2,-2,'test','natural')
    with pytest.raises(TypeError):
        Continuation(ode1,2,-2,2,'natural')






  


