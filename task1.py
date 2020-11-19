def multiplicate(A):
    assert len(A) >= 1
    if len(A) == 1:
        return [0]

    forward_mul = A.copy()
    for i in range(1, len(forward_mul)):
        forward_mul[i] *= forward_mul[i-1]

    backward_mul = A.copy()
    for i in range(len(A)-2, -1, -1):
        backward_mul[i] *= backward_mul[i+1]

    result = []
    for i in range(len(A)):
        if i == 0:
            result.append(backward_mul[i+1])
        elif i == len(A) - 1:
            result.append(forward_mul[i-1])
        else:
            result.append(forward_mul[i-1] * backward_mul[i+1])

    return result


if __name__ == "__main__":
    A = [int(el) for el in input().split()]
    print(multiplicate(A))
