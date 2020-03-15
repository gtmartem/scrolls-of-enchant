import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Word, alphas, nums

source = open("maxFromLinesMins_threadstest.out", "r")
line = Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums) + ','
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
threads = list()
for _ in src:
    time.append(line.parseString(_)[10])
for _ in src:
    threads.append(line.parseString(_)[6])

print(time)
print(threads)

x = np.array(threads, dtype = float)
y = np.array(time, dtype = float)

fig = plt.figure()
plt.plot(x,y,"o-",color = 'r')
plt.xlabel("threads")
plt.ylabel("time")
plt.title("f : time(threads) dim = 10000")
fig.savefig("maxFromLinesMins_threadstest.png")

source = open("maxFromLinesMins_dimtest_10.out", "r")
line = Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums) + ','
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[10])
for _ in src:
    dim.append(line.parseString(_)[6])

print(time)
print(threads)

x_10 = np.array(dim, dtype = float)
y_10 = np.array(time, dtype = float)

source = open("maxFromLinesMins_dimtest_2.out", "r")
line = Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums) + ','
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[10])
for _ in src:
    dim.append(line.parseString(_)[6])

print(time)
print(threads)

x_2 = np.array(dim, dtype = float)
y_2 = np.array(time, dtype = float)

source = open("maxFromLinesMins_dimtest_1.out", "r")
line = Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums) + ','
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
time = list()
dim = list()
for _ in src:
    time.append(line.parseString(_)[10])
for _ in src:
    dim.append(line.parseString(_)[6])

print(time)
print(threads)

x_1 = np.array(dim, dtype = float)
y_1 = np.array(time, dtype = float)

fig = plt.figure()
plt.semilogx(x_10,y_10,"o-",color = 'r')
plt.semilogx(x_2,y_2,"o-",color = 'g')
plt.semilogx(x_1,y_1,"o-",color = 'b')
plt.xlabel("dim")
plt.ylabel("time")
plt.title("f : time(dim) (threads red - 10, green - 2, blue - 1)")
fig.savefig("maxFromLinesMins_dimtest.png")
