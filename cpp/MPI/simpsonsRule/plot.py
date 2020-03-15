import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Word, alphas, nums

source = open("simpsonsRule_acctest.out", "r")
line = Word(alphas.upper(), alphas.lower()) + ':' + Word('-'+nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums) + ','
line = line + Word(alphas) + ':' + Word(nums+'.')

src = [line.strip() for line in source]
acc = list()
N = list()
for _ in src:
    N.append(line.parseString(_)[10])
for _ in src:
    acc.append(line.parseString(_)[2])

print(N)
print(acc)

x = np.array(N, dtype = float)
y = np.array(acc, dtype = float)

fig = plt.figure()
plt.plot(x,y,"o-",color = 'r')
plt.xlabel("N")
plt.ylabel("acc")
plt.title("f : acc(N)")
fig.savefig("simpsonsRule_acctest.png")

source = open("simpsonsRule_timetest.out", "r")
line = Word(alphas) + ':' + Word(nums+'.') + ',' 
line = line + Word(alphas) + ':' + Word(nums) + ',' 
line = line + Word(alphas) + ':' + Word(nums)

src = [line.strip() for line in source]
time = list()
proc = list()
for _ in src:
    proc.append(line.parseString(_)[10])
for _ in src:
    time.append(line.parseString(_)[2])

print(proc)
print(time)

x = np.array(proc, dtype = float)
y = np.array(time, dtype = float)

fig = plt.figure()
plt.plot(x,y,"o-",color = 'r')
plt.xlabel("Processes")
plt.ylabel("Time")
plt.title("f : time(processes)")
fig.savefig("simpsonsRule_timetest.png")