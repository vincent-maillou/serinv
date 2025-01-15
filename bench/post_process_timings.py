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
            1.5594987738877535,
            1.5599795547313988,
            1.5598487770184875,
            1.5594719271175563,
            1.5596821592189372,
            1.560069088358432,
            1.5595460552722216,
            1.559863612987101,
            1.5597794889472425,
            1.5599749688990414,
            1.5596723891794682,
            1.5598031682893634,
            1.5606733108870685,
            1.5598723478615284,
            1.5597196500748396,
            1.5603729048743844,
            1.5598276779055595,
            1.5598690211772919,
            1.5605465481057763,
            1.5606088167987764,
        ]
    )
    t_pobtaf = np.array(
        [
            0.93309,
            0.93295,
            0.93279,
            0.93280,
            0.93276,
            0.93306,
            0.93323,
            0.93275,
            0.93309,
            0.93289,
            0.93288,
            0.93312,
            0.93313,
            0.93313,
            0.93304,
            0.93289,
            0.93329,
            0.93310,
            0.93305,
            0.93324,
        ]
    )
    t_pobtasi = np.array(
        [
            0.49623,
            0.49581,
            0.49434,
            0.49444,
            0.49438,
            0.49593,
            0.49552,
            0.49590,
            0.49610,
            0.49570,
            0.49416,
            0.49610,
            0.49598,
            0.49445,
            0.49440,
            0.49434,
            0.49463,
            0.49447,
            0.49450,
            0.49425,
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
