import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200, suppress=True)

closes = np.loadtxt('./data/EURUSD_M1.csv', delimiter='\t', skiprows=1, usecols=4)
diffs = np.diff(closes)
closes = closes[1:]


window_size = 10
n = 20
n_test_data = 3

window = np.array(np.arange(0, window_size))
n_sequence = np.expand_dims(np.array(np.arange(0, n)), 1)

indexes = window + n_sequence 


window_start_indexes =  n_sequence * np.ones(window_size, np.int8)
window_percent_cng = (closes[indexes] - closes[window_start_indexes]) / closes[window_start_indexes]

print(window_percent_cng)
window_percent_cng_test_data = window_percent_cng[-10:-1]
window_percent_cng, window_percent_cng_test_data = np.split(window_percent_cng, [n - n_test_data])
window_percent_cng_test_data = np.expand_dims(window_percent_cng_test_data, 1)

window_percent_cng_test_data = window_percent_cng_test_data + np.expand_dims(np.zeros(n - n_test_data), 1)

diffs = np.sum(np.abs(window_percent_cng_test_data - window_percent_cng),axis=2)

#x_plot = np.tile(np.arange(0, window_size, 1), (n, 1))
#plt.plot(x_plot.T, window_percent_cng.T)
#plt.show()
