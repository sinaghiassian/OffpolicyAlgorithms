LEFT = 0
RIGHT = 1
N = 5


def init():
    return N // 2


def sample(s, a):
    s_prime = None
    if a == LEFT:
        s_prime = max(s - 1, -1)
    elif a == RIGHT:
        s_prime = min(s + 1, N)

    r = 0
    terminal = False

    if s_prime == -1:
        r = 0
        terminal = True
    elif s_prime == N:
        r = 1
        terminal = True

    return s_prime, r, terminal
