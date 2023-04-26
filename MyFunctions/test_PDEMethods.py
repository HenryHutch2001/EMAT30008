import pytest
from MyFunctions.PDE_Solve import CreateAandb,FiniteDifference,ExplicitEuler,ImplicitEuler,CrankNicolson,RKSolver
import numpy as np

def source1(x,f):
    return np.ones(np.size(x))

source2 = 5

def source3(x,t,u,mu=2):
    return np.exp(mu*u)

def source4(x,t,u):
    return np.ones(np.size(x))

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
    with pytest.raises(RuntimeWarning):
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

def test_ExplicitEuler():
    x,y = ExplicitEuler(10,0,1,1,1,0.001,0.0,0.0,initial_condition1,source3)
    assert isinstance(x,np.ndarray)
    assert isinstance(y,np.ndarray)
    with pytest.raises(ValueError):
        ExplicitEuler(3,0,1,1,1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(11.5,0,1,1,1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0.5,1,1,1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,1,0,1,1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1.5,1,1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,'testing',1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,-1,1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,1,'testing',0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,1,-1,0.001,0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,1,1,'testing',0.0,0.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,1,1,0.001,1,1.0,initial_condition1,source3)
        ExplicitEuler(10,0,1,1,1,0.001,1.0,1,initial_condition1,source3)
    with pytest.raises(RuntimeWarning):
        ExplicitEuler(201,0,1,1,1,0.001,0.0,0.0,initial_condition1,source3)
    with pytest.raises(TypeError):
        ExplicitEuler(10,0,1,1,1,0.001,0.0,0.0,initial_condition2,source3)
        ExplicitEuler(10,0,1,1,1,0.001,0.0,0.0,initial_condition1,source2)

def test_ImplicitEuler():
    x,y = ImplicitEuler(10,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
    assert isinstance(x,np.ndarray)
    assert isinstance(y,np.ndarray)
    with pytest.raises(ValueError):
        ImplicitEuler(3,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10.5,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0.5,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,1,0,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1.5,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,'test',1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,-1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,1,'test',0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,1,-1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,1,1,'test','dirichlet',0.0,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,1,1,0.0001,'dirichlet',1,1.0,initial_condition1,source4)
        ImplicitEuler(10,0,1,1,1,0.0001,'dirichlet',1.0,1,initial_condition1,source4)
        ImplicitEuler(10,0,1,1,1,0.0001,'test',0.0,1.0,initial_condition1,source4)
    with pytest.raises(RuntimeWarning):
        ImplicitEuler(301,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
    with pytest.raises(TypeError):
        ImplicitEuler(10,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition2,source4)
        ImplicitEuler(10,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source2)

def test_CrankNicoloson():
    x,y = CrankNicolson(10,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
    assert isinstance(x,np.ndarray)
    assert isinstance(y,np.ndarray)
    with pytest.raises(ValueError):
        CrankNicolson(4,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10.5,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0.5,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1.5,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,1,0,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1,'test',1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1,1,'test',0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1,1,-1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1,1,1,'test','dirichlet',0.0,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1,1,1,0.0001,'dirichlet',1,1.0,initial_condition1,source4)
        CrankNicolson(10,0,1,1,1,0.0001,'dirichlet',1.0,1,initial_condition1,source4)
        CrankNicolson(10,0,1,1,1,0.0001,'test',0.0,1.0,initial_condition1,source4)
    with pytest.raises(RuntimeWarning):
        CrankNicolson(301,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source4)
    with pytest.raises(TypeError):
        CrankNicolson(10,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition1,source2)
        CrankNicolson(10,0,1,1,1,0.0001,'dirichlet',0.0,1.0,initial_condition2,source4)

def test_RK():
    x,y,t = RKSolver(10,0,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
    assert isinstance(x,np.ndarray)
    assert isinstance(y,np.ndarray)
    assert isinstance(t,np.ndarray)
    with pytest.raises(ValueError):
        RKSolver(3,0,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver('test',0,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0.5,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,1,0,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1.5,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1,'test',1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1,-1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1,1,'test',0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1,1,-1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1,1,1,'test','dirichlet',0.0,0.0,initial_condition1,source4)
        RKSolver(10,0,1,1,1,0.01,'dirichlet','test',0.0,initial_condition1,source4)
        RKSolver(10,0,1,1,1,0.01,'dirichlet',0.0,'test',initial_condition1,source4)
        RKSolver(10,0,1,1,1,0.01,'neumann',0.0,'test',initial_condition1,source4)
        RKSolver(10,0,1,1,1,0.01,'neumann',0.0,'test',initial_condition1,source4)
        RKSolver(10,0,1,1,1,0.01,'robin','test',[0.0,0.0],initial_condition1,source4)
        RKSolver(10,0,1,1,1,0.01,'robin',0.0,'test',initial_condition1,source4)
    with pytest.raises(RuntimeWarning):
        RKSolver(301,0,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source4)
    with pytest.raises(TypeError):
        RKSolver(10,0,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition2,source4)
        RKSolver(10,0,1,1,1,0.01,'dirichlet',0.0,0.0,initial_condition1,source2)
    