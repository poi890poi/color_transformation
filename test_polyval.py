import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.special import comb

'''
l = a0 + a1*l + a2*a + a3*b
1 degree -> 1 + 3 = 4 coefficients

l = a0 + a1*l + a2*a + a3*b + a4*l*a + a5*l*b + a6*a*b + a7*l**2 + a8*a**2 + a9*b**2
2 degree -> 1 + 3 + 6 = 10 coefficients

3 degree -> 1 + 3 + 6 + 10 = 20 coefficients

a
b

1 + 3 + 6 + 
4
10
20
35
56
'''

COLOR_POLYN_DEGREE = 2

def array_to_coeffs(params, degree, channels=3):
    '''
    Some polynomial functions use array of N*D to store coefficients while
    acutal number of coefficients for the polynomial function is comb(N+D, D).
    This function takes array of shape (N, D) and return 1D array of the 
    coefficients.
    '''
    d = degree
    D = degree + 1
    coeffs = np.zeros((channels, D, D, D), dtype=np.float)
    for c in range(channels):
        p = 0
        for z in range(D):
            for y in range(D-z):
                l = D - y - z
                coeffs[c, 0:l, y, z] = params[c, p:p+l]
                p += l
    return coeffs

def coeffs_to_array(coeffs, degree, channels=3):
    '''
    Some polynomial functions use array of N*D to store coefficients while
    acutal number of coefficients for the polynomial function is comb(N+D, D).
    This function takes 1D array of the coefficients and return array of 
    shape (N, D).
    '''
    d = degree
    D = degree + 1
    params = np.zeros((channels, int(comb(d + channels, d))), dtype=np.float)
    for c in range(channels):
        p = 0
        for z in range(D):
            for y in range(D-z):
                l = D - y - z
                print(l, y, z)
                params[c, p:p+l] = coeffs[c, 0:l, y, z]
                p += l
    return params

params = np.arange(0, int(comb(COLOR_POLYN_DEGREE + 3, 3)) * 3)
params = params.reshape(3, -1)
print(params)

D = COLOR_POLYN_DEGREE + 1
coeffs = np.zeros((3, D, D, D), dtype=np.float)
p = 0
for j in range(D, 0, -1):
    for k in range(j, 0, -1):
        print(k)
        p += k

coeffs = array_to_coeffs(params, COLOR_POLYN_DEGREE)
print(coeffs)
print(coeffs_to_array(coeffs, COLOR_POLYN_DEGREE))

c = np.zeros((3, 3, 3, 3))
c[0][1][0][0] = 1
c[1][0][1][0] = 1
c[2][0][0][1] = 1
print(c)
p = coeffs_to_array(c, COLOR_POLYN_DEGREE)
print('P', p)
c = array_to_coeffs(p, COLOR_POLYN_DEGREE)
print('C', c)

raise

d = np.array([[3, 5, 7]], dtype=np.float)
c = np.zeros((3, 3, 3, 3))
c[0][1][0][0] = 1
c[1][0][1][0] = 1
c[2][0][0][1] = 1
print(poly.polyval3d(d[:, 0], d[:, 1], d[:, 2], c[0]),
    poly.polyval3d(d[:, 0], d[:, 1], d[:, 2], c[1]),
    poly.polyval3d(d[:, 0], d[:, 1], d[:, 2], c[2]),)

'''
(4, 2)
y^0 + y^1 + y^2 + y^3
x^1 + x*y + x*y^2 + x*y^3
x^2
'''