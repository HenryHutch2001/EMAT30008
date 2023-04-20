import pytest
from PDE_Solve import CreateAandb,FiniteDifference,ExplicitEuler,ImplicitEuler,CrankNicolson
import numpy as np

def source1(x):
    return np.ones(np.size(x))

source2 = 5

def source3(x,t,u,mu=2):
    return np.exp(mu*u)

def initial_condition1(x):
    return np.ones(np.size(x))

initial_condition2 = 5

def test_FiniteDifference():
    x,y = FiniteDifference(source1,1,10,'neumann',0.0,1.0)
    assert isinstance(x,np.ndarray)
    assert isinstance(y,np.ndarray)
    with pytest.raises(ValueError):
        FiniteDifference(source1,1,3,'dirichlet',0.0,1.0)
    with pytest.raises(ValueError):
        FiniteDifference(source1,1,0.4,'dirichlet',0.0,1.0)
    with pytest.raises(RuntimeError):
        FiniteDifference(source1,1,300,'dirichlet',0.0,1.0)
    with pytest.raises(ValueError):
        FiniteDifference(source1,'hello',10,'dirichlet',0.0,1.0)
    with pytest.raises(ValueError):
        FiniteDifference(source1,-1,10,'dirichlet',0.0,1.0)
    with pytest.raises(ValueError):
        FiniteDifference(source1,1,10,'dirichlet',4,1.0)
    with pytest.raises(ValueError):
        FiniteDifference(source1,1,10,'dirichlet',0.0,3)
    with pytest.raises(TypeError):
        FiniteDifference(source2,1,10,'dirichlet',0.0,1.0)
    with pytest.raises(ValueError):
        FiniteDifference(source1,1,10,'testing',0.0,1.0)


