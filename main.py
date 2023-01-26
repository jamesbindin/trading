import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=200)

closes = np.loadtxt('./data/EURUSD_M1.csv', delimiter='\t', skiprows=1, usecols=4)
diffs = np.diff(closes)
closes = closes[1:]

window_size = 10
n = 20

window = np.array(np.arange(0, window_size))
n_sequence = np.expand_dims(np.array(np.arange(0, n)), 1)
indexes = window + n_sequence 

window_start_indexes =  n_sequence * np.ones(window_size, np.int8)
window_percent_cng = (closes[indexes] - closes[window_start_indexes]) / closes[window_start_indexes]

x = np.tile(np.arange(0, window_size, 1), (n, 1))

plt.plot(x.T, window_percent_cng.T)
plt.show()
