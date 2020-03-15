import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Word, alphas, nums

source = open("scalarProductExtras_atomic.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    dim.append(line.parseString(_)[2])

print(time)
print(dim)

x_atom = np.array(dim, dtype = float)
y_atom = np.array(time, dtype = float)

source = open("scalarProductExtras_critical.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    dim.append(line.parseString(_)[2])

print(time)
print(dim)

x_crit = np.array(dim, dtype = float)
y_crit = np.array(time, dtype = float)

source = open("scalarProductExtras_locks.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    dim.append(line.parseString(_)[2])

print(time)
print(dim)

x_lock = np.array(dim, dtype = float)
y_lock = np.array(time, dtype = float)

source = open("scalarProductExtras_reduction.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    dim.append(line.parseString(_)[2])

print(time)
print(dim)

x_redu = np.array(dim, dtype = float)
y_redu = np.array(time, dtype = float)

fig = plt.figure()
plt.semilogx(x_atom,y_atom,"o-",color = 'r')
plt.semilogx(x_crit,y_crit,"o-",color = 'g')
plt.semilogx(x_lock,y_lock,"o-",color = 'b')
plt.semilogx(x_redu,y_redu,"o-",color = 'black')
plt.xlabel("dim")
plt.ylabel("time")
plt.title("f : time(dim) atomic - red, critical - green, lock - blue, reduction - black")
fig.savefig("simpsonsRule_threadstest.png")