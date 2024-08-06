import numpy as np

def vector_add(a,b,c):
    # Reference storage of c versus assigning a new temporary to c
    c[:] = a + b


n = 100
a = np.ones(n,dtype=np.float32)
b = np.linspace(start=0.0, stop=1.0, num=n, dtype=np.float32)

c = np.zeros_like(a)
#print('b',b)
#print('a',a)

vector_add(a,b,c)
print('c',c[0:10])
