from memory_profiler import profile


N = 300


@profile
def list_comprehension_func():
    b = [[1]*(8**7) for i in range(N)]
    print(len(b))
    del b


if __name__ == '__main__':
    list_comprehension_func()
