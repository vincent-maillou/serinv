import time


import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


if __name__ == "__main__":
    t_ordering = np.array(
        [
            22.5237,
            22.5175,
            22.4671,
            22.4602,
            22.4625,
            22.5356,
            22.6629,
            22.4256,
            22.4837,
            22.7216,
        ]
    )
    t_pobtaf = np.array(
        [
            9.55964,
            9.66214,
            9.63006,
            9.54471,
            9.63873,
            9.72026,
            9.67979,
            9.53878,
            9.48534,
            9.59536,
        ]
    )
    t_pobtasi = np.array(
        [
            60.1223,
            55.5015,
            68.9829,
            49.1418,
            63.7999,
            50.1767,
            57.6179,
            54.283,
            55.8491,
            52.4062,
        ]
    )

    mean_ordering, lb_mean_ordering, ub_mean_ordering = mean_confidence_interval(
        t_ordering
    )
    mean_pobtaf, lb_mean_pobtaf, ub_mean_pobtaf = mean_confidence_interval(t_pobtaf)
    mean_pobtasi, lb_mean_pobtasi, ub_mean_pobtasi = mean_confidence_interval(t_pobtasi)

    print(
        f"Mean time ordering: {mean_ordering:.5f} sec, 95% CI: [{lb_mean_ordering:.5f}, {ub_mean_ordering:.5f}]"
    )

    print(
        f"Mean time pobtaf: {mean_pobtaf:.5f} sec, 95% CI: [{lb_mean_pobtaf:.5f}, {ub_mean_pobtaf:.5f}]"
    )

    print(
        f"Mean time pobtasi: {mean_pobtasi:.5f} sec, 95% CI: [{lb_mean_pobtasi:.5f}, {ub_mean_pobtasi:.5f}]"
    )
