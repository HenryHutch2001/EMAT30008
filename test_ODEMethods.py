# %% Importing packages 
from My_Functions import shooting,solve_to
import pytest
# %%
import numpy as np
from My_Functions import solve_toEU, solve_toRK

def ode(t,y): 
    dx_dt = y[1]
    dy_dt = -y[0]
    return [dx_dt, dy_dt]

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
  


