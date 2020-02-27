import numpy as np
from scipy.interpolate import interp1d

### Spiral integration functions
def make_inverse_interpolant(function, tmin=0, tmax=2*np.pi, interpoints=100):
    '''returns primitive and inverse primitive between tmin and tmax
    if interpoints is int: number of equaly spaced points to take'''
    if isinstance(interpoints, int):
        xx = np.linspace(tmin, tmax, interpoints)
    else:
        xx = interpoints
    primitive = interp1d(xx, function(xx), kind='cubic')
    invprimitive = interp1d(primitive(xx), xx, kind='cubic')
    return primitive, invprimitive

def get_theta(area, t1, primitive, invprimitive):
    '''given area, t1 and promitive and invprimitive
    returns the angle t2 for which the area between t1 and t2
    is equal to area'''
    return invprimitive(primitive(t1) + area)

### Main Spiral class
class Spiral():
    def __init__(self):
        pass

    def integrate_spiral(self, x):
        p = x // (2*np.pi)
        try:
            temp = self.integral(x - 2. * np.pi * np.arange(p+1))
            return temp @ ((-1)**np.arange(p+1))
        except TypeError:
            return np.array([
                self.integral(xi - 2. * np.pi * np.arange(pi+1))
                @ ((-1)**np.arange(pi+1))
                for xi, pi in zip(x, p)])

    def get_area(self, t1, t2):
        return self.integrate_spiral(t2) - self.integrate_spiral(t1)

    def make_inverse_interpolant(self, tmin=0, tmax=2*np.pi, interpoints=100):
        return make_inverse_interpolant(
            self.integrate_spiral, tmin, tmax, interpoints)

### Spiral types
class PolySpiral(Spiral):
    def __init__(self, a=1, i=1, j=1, k=1):
        self.a = a
        self.i = i
        self.j = j
        self.k = k

    def radius(self, theta):
        r = self.a**(self.i/self.k) * theta**(self.j/self.k)
        return r

    def integral(self, theta):
        a, i, j, k = self.a, self.i, self.j, self.k
        return a**(2*i/k) / 2 * theta**(2*j/k + 1) / (2*j/k + 1)

    def inv_integral(self, area):
        a, i, j, k = self.a, self.i, self.j, self.k
        return (2 * area * (2*j/k + 1) / a**(2*i/k))**(1 / (2*j/k + 1))

class ArchimedesSpiral(PolySpiral):
    def __init__(self, a=1 / (2*np.pi)):
        self.a = a
        self.i, self.j, self.k = 1, 1, 1

class GalileeSpiral(PolySpiral):
    def __init__(self, a=1):
        self.a = a
        self.i, self.j, self.k = 1, 2, 1

class FermatSpiral(PolySpiral):
    def __init__(self, a=1):
        self.a = a
        self.i, self.j, self.k = 2, 1, 2

class HyperbolicSpiral(PolySpiral):
    def __init__(self, a=1):
        self.a = a
        self.i, self.j, self.k = 1, -0.5, 1

class LituusSpiral(PolySpiral):
    def __init__(self, a=1):
        self.a = a
        self.i, self.j, self.k = -2, 1, 2

class LogSpiral(Spiral):
    def __init__(self, a=1, b=1.27):
        self.a = a
        self.b = b

    def radius(self, theta):
        r = self.a * (self.b**theta - 1)
        return r

    def integral(self, theta):
        return self.a**3 / 3 * (self.b**theta - 1)

    def inv_integral(self, area):
        return np.log((3 * area / self.a**3)**0.5 + 1) / np.log(self.b)

class GoldenSpiral(LogSpiral):
    def __init__(self, a=1):
        self.a = a
        phi = (1 + np.sqrt(5)) / 2
        self.b = phi**(2 / np.pi)

class Circle(Spiral):
    def __init__(self, a=1):
        self.a = a

    def radius(self, theta):
        r = self.a
        return r + 0*theta

    def integral(self, theta):
        return self.a**2 / 2 * theta

    def inv_integral(self, area):
        return 2 * area / self.a**2
