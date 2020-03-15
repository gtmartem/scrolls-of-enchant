import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Word, alphas, nums

source = open("scalarProduct_threadstest.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
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
plt.title("f : time(threads), dim : 10000000")
fig.savefig("scalarProduct_threadstest.png")

""" ################################### """

source = open("scalarProduct_dimtest_10.out", "r")
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

x_10 = np.array(dim, dtype = float)
y_10 = np.array(time, dtype = float)

source = open("scalarProduct_dimtest_1.out", "r")
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

x_1 = np.array(dim, dtype = float)
y_1 = np.array(time, dtype = float)

source = open("scalarProduct_dimtest_25.out", "r")
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

x_25 = np.array(dim, dtype = float)
y_25 = np.array(time, dtype = float)

fig = plt.figure()
plt.semilogx(x_1,y_1,"o-",color = 'r')
plt.semilogx(x_10,y_10,"o-",color = 'g')
plt.semilogx(x_25,y_25,"o-",color = 'b')
plt.xlabel("dim")
plt.ylabel("time")
plt.title("f : time(dim); red - 1 thread, green - 10 threads, blue - 25 threads")
fig.savefig("scalarProduct_dimtest.png")
