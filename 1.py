import numpy as np
from scipy import optimize, minimum
from scipy.optimize import minimize

"""
minimize f(x) subject to

g_i(x) >= 0,  i = 1,...,m
h_j(x)  = 0,  j = 1,...,p
"""
"""
</p>
bounds None means 00
"""
def func(x, sign=1.0):
    """ Objective function """
    return sign*(10-x[0]**2-x[1]**2)
    '''
    
    '''
    """
    when sign=-1.0 maximum 
    """
def func_deriv(x, sign=1.0):
     """ Derivative of objective function """
     dfdx0 = sign*(-2*x[0])
     dfdx1 = sign*(-2*x[1])
     return np.array([ dfdx0, dfdx1 ])
cons = ({'type': 'eq',
        'fun' : lambda x: np.array([x[1]+x[0]]),
        'jac' : lambda x: np.array([1.0,1.0])},#map
        {'type': 'ineq',
        'fun' : lambda x: np.array([-x[0]**2+x[1]]),
        'jac' : lambda x: np.array([-2*x[0],1.0])}) #inep means >=0
#set

''' 
res = minimize(func, jac=func_deriv,
    method='SLSQP', options={'disp': True})
    '''
bonds=((-np.inf, np.inf), (-np.inf, np.inf))
res = minimize(fun=func, x0=[-2.0,2.0],bounds=bonds, jac=func_deriv,constraints=cons, method='SLSQP', options={'disp': True})

print res
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
       {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
       {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
bnds = ((0, None), (0, None))
res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
           constraints=cons)
#print res