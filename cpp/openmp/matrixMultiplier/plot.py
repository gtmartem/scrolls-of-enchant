import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Word, alphas, nums

source = open("matrixMultiplier_threadstest.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
threads = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    threads.append(line.parseString(_)[2])

print(time)
print(threads)

x = np.array(threads, dtype = float)
y = np.array(time, dtype = float)

fig = plt.figure()
plt.plot(x,y,"o-",color = 'r')
plt.xlabel("threads")
plt.ylabel("time")
plt.title("f : time(threads)")
fig.savefig("matrixMultiplier_threadstest.png")

source = open("matrixMultiplier_dimtest_40.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    dim.append(line.parseString(_)[2])

print(time)
print(threads)

x_40 = np.array(dim, dtype = float)
y_40 = np.array(time, dtype = float)

source = open("matrixMultiplier_dimtest_10.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[6])
for _ in src:
    dim.append(line.parseString(_)[2])

print(time)
print(threads)

x_10 = np.array(dim, dtype = float)
y_10 = np.array(time, dtype = float)

fig = plt.figure()
plt.semilogx(x_10,y_10,"o-",color = 'r')
plt.semilogx(x_40,y_40,"o-",color = 'g')
plt.xlabel("dim")
plt.ylabel("time")
plt.title("f : time(threads) red - 10 threads, green - 40 threads")
fig.savefig("matrixMultiplier_dimtest.png")