import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


# Load the .npy file
# n_p = 2
# d_pobtaf = np.load(f"timings_d_pobtaf_bs2865_as4_nb365_np{n_p}.npy")
# d_pobtasi = np.load(f"timings_d_pobtasi_bs2865_as4_nb365_np{n_p}.npy")
# d_pobtasi_rss = np.load(f"timings_d_pobtasi_rss_bs2865_as4_nb365_np{n_p}.npy")

d_pobtaf = np.load("timings_synthetic_pobtaf_b2865_a4_n365.npy")
d_pobtasi = np.load("timings_synthetic_pobtasi_b2865_a4_n365.npy")

mean_pobtaf, lb_mean_pobtaf, ub_mean_pobtaf = mean_confidence_interval(d_pobtaf)
mean_pobtasi, lb_mean_pobtasi, ub_mean_pobtasi = mean_confidence_interval(d_pobtasi)
# mean_pobtasi_rss, lb_mean_pobtasi_rss, ub_mean_pobtasi_rss = mean_confidence_interval(
#     d_pobtasi_rss
# )

print(
    f"Mean time pobtaf: {mean_pobtaf:.5f} sec, 95% CI: [{lb_mean_pobtaf:.5f}, {ub_mean_pobtaf:.5f}]"
)
print(
    f"Mean time pobtasi: {mean_pobtasi:.5f} sec, 95% CI: [{lb_mean_pobtasi:.5f}, {ub_mean_pobtasi:.5f}]"
)
# print(
#     f"Mean time pobtasi_rss: {mean_pobtasi_rss:.5f} sec, 95% CI: [{lb_mean_pobtasi_rss:.5f}, {ub_mean_pobtasi_rss:.5f}]"
# )
