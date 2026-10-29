def speedup(t1, tp):
    return t1 / tp

def karp_flatt(t1, tp, p):
    """e(p) = ((p * Tp / T1) - 1) / (p - 1), p >= 2"""
    return ((p * tp / t1) - 1.0) / (p - 1.0)
