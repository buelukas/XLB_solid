import numpy as np
_f = np.zeros((5, 4))
f = np.zeros(20)

for i in range(4):
    for j in range(5):
        _f[j,i] = 5*i + j

for i in range(4):
    for j in range(5):
        f[j + 5*i] = _f[j, i]

print(_f)
print(f)