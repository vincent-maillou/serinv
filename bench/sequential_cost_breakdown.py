def potrf_b3(b: int, n: int):
    flops = 1 / 3 * pow(b, 3) + 1 / 2 * pow(b, 2) + 1 / 6 * b
    return n * flops


def potrf_a3(a: int, n: int):
    flops = 1 / 3 * pow(a, 3) + 1 / 2 * pow(a, 2) + 1 / 6 * a
    return n * flops


def gemm_b3(b: int, n: int):
    flops = 2 * pow(b, 3)
    return n * flops


def gemm_ab2(a: int, b: int, n: int):
    flops = 2 * a * pow(b, 2)
    return n * flops


def gemm_ba2(a: int, b: int, n: int):
    flops = 2 * b * pow(a, 2)
    return n * flops


def gemm_a3(a: int, n: int):
    flops = 2 * pow(a, 3)
    return n * flops


def trsm_b3(b: int, n: int):
    flops = pow(b, 3)
    return n * flops


def trsm_ab2(a: int, b: int, n: int):
    flops = a * pow(b, 2)
    return n * flops


def get_pobtaf_flops(a: int, b: int, n: int):

    cholesky = potrf_b3(b, n) + potrf_a3(a, 1)
    gemm = gemm_b3(b, 4 * (n - 1)) + gemm_ab2(a, b, 5 * (n - 1)) + gemm_ba2(a, b, n)
    trsm = trsm_b3(b, n - 1) + trsm_ab2(a, b, n)
    total = cholesky + gemm + trsm

    cholesky_pct = (cholesky / total) * 100
    gemm_pct = (gemm / total) * 100
    trsm_pct = (trsm / total) * 100

    return {"cholesky_pct": cholesky_pct, "gemm_pct": gemm_pct, "trsm_pct": trsm_pct}


def get_pobtasi_flops(a: int, b: int, n: int):

    gemm = (
        gemm_b3(b, n - 1)
        + gemm_ab2(a, b, n - 1)
        + gemm_ba2(a, b, n - 1)
        + gemm_a3(a, 1)
    )
    trsm = trsm_b3(b, n - 1) + trsm_ab2(a, b, 1)
    total = gemm + trsm

    gemm_pct = (gemm / total) * 100
    trsm_pct = (trsm / total) * 100

    return {"gemm_pct": gemm_pct, "trsm_pct": trsm_pct}


def get_d_pobtaf_flops(a: int, b: int, n: int):

    cholesky = potrf_b3(b, n)
    gemm = gemm_b3(b, 2 * n) + gemm_ab2(a, b, 2 * n) + gemm_ba2(a, b, 2 * n)
    trsm = trsm_b3(b, 2 * n) + trsm_ab2(a, b, n)
    total = cholesky + gemm + trsm

    cholesky_pct = (cholesky / total) * 100
    gemm_pct = (gemm / total) * 100
    trsm_pct = (trsm / total) * 100

    return {"cholesky_pct": cholesky_pct, "gemm_pct": gemm_pct, "trsm_pct": trsm_pct}


def get_d_pobtasi_flops(a: int, b: int, n: int):

    gemm = gemm_b3(b, 9 * (n - 1)) + gemm_ab2(a, b, 7 * (n - 1))
    trsm = trsm_b3(b, n - 1)
    total = gemm + trsm

    gemm_pct = (gemm / total) * 100
    trsm_pct = (trsm / total) * 100

    return {"gemm_pct": gemm_pct, "trsm_pct": trsm_pct}


if __name__ == "__main__":
    b = 2865
    a = 4
    n = 365

    pobtaf_flops = get_pobtaf_flops(a, b, n)

    print("pobtaf:", pobtaf_flops)

    pobtasi_flops = get_pobtasi_flops(a, b, n)

    print("pobtasi:", pobtasi_flops)

    d_pobtaf_flops = get_d_pobtaf_flops(a, b, n)

    print("d_pobtaf:", d_pobtaf_flops)

    d_pobtasi_flops = get_d_pobtasi_flops(a, b, n)

    print("d_pobtasi:", d_pobtasi_flops)
