import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Word, alphas, nums

source = open("simpsonsRule_threadstest_100000.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ','
line = line + Word(alphas) + ':' + Word(nums)

src = [line.strip() for line in source]
time = list()
threads = list()
for _ in src:
    time.append(line.parseString(_)[10])
for _ in src:
    threads.append(line.parseString(_)[14])

print(time)
print(threads)

x_10 = np.array(threads, dtype = float)
y_10 = np.array(time, dtype = float)

source = open("simpsonsRule_threadstest_1410065408.out", "r")
line = Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ','
line = line + Word(alphas) + ':' + Word(nums)

src = [line.strip() for line in source]
time = list()
threads = list()
for _ in src:
    time.append(line.parseString(_)[10])
for _ in src:
    threads.append(line.parseString(_)[14])

print(time)
print(threads)

x_14 = np.array(threads, dtype = float)
y_14 = np.array(time, dtype = float)

fig = plt.figure()
plt.semilogy(x_10,y_10,"o-",color = 'r')
plt.semilogy(x_14,y_14,"o-",color = 'g')
plt.xlabel("threads")
plt.ylabel("time")
plt.title("f : time(threads) red - N = 100000, green - N = 1410065408")
fig.savefig("simpsonsRule_threadstest.png")