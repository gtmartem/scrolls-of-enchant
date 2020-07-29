from memory_profiler import profile


N = 300


@profile
def for_func():
    b = []
    for i in range(N):
        b.append([1]*(8**7))
    print(len(b))
    del b


if __name__ == '__main__':
    for_func()
