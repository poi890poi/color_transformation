import numpy as np

#p_down [(552.0, None), array([ -19.75, 6352.5 ]), (393.0, None), (150.0, None), (231.0, None), (472.0, None)]

a = np.array([[  0.,  -1.,  62., 313., 279.],
       [  0.,   0.,  62., 313., 120.],
       [  0.,   1.,  62., 312.,  40.]])

d = np.vstack((a[:, 3:5], a[:, 3:5] + a[:, 1:3]))
print(d, np.mean(a[:, 1]))
p_ = np.polyfit(d[:, 0], d[:, 1], 1)
print(p_)
a, b = p_
for x, y in d:
    print(y, a*x + b)