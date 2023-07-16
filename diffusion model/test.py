import time


def main(base_num: int) -> list:
    base_list = [i for i in range(2, base_num + 1)]

    prime_list = []
    while True:
        prime = base_list[0]
        prime_list.append(prime)

        base_list = sieve(prime, base_list)
        if len(base_list) == 0:
            break

    return prime_list

def sieve(prime, base_list) -> list:
    return list(filter(lambda x: x % prime != 0, base_list))

# 計測
t1 = time.perf_counter_ns() 
print(main(100000))
t2 = time.perf_counter_ns()

print(f"経過時間：{(t2- t1) / 1_000_000} ms")
