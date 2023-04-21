# %% Importing packages 
from ODE_Solve import shooting,solve_to,solve_toEU,solve_toRK
import pytest
# %%
import numpy as np
from My_Functions import solve_toEU, solve_toRK

def ode(t,y): 
    dx_dt = y[1]
    dy_dt = -y[0]
    return [dx_dt, dy_dt]

ode1 = 5

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




  


