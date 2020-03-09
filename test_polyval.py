import numpy as np
import numpy.polynomial.polynomial as poly

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