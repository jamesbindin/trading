import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200, suppress=True)
import sys; np.set_printoptions(threshold=sys.maxsize)


closes = np.loadtxt('./data/EURUSD_M1.csv', delimiter='\t', skiprows=1, usecols=4)
closes_i = np.arange(closes.shape[0])

train_n = 4000
test_n = 1000 
window_n = 20
target_step = 20
tolerance = 0.0001

offset = closes.shape[0] - (train_n + test_n + window_n*2 + target_step*2 - 2)

test_start_i = window_n + train_n + target_step - 1
train_I = np.arange(window_n).reshape(1, -1) + np.arange(train_n).reshape(-1, 1) + offset
train_TI = (train_I[:, -1] + target_step).reshape(-1, 1)

test_I = np.arange(test_n).reshape(-1, 1) + test_start_i + np.arange(window_n).reshape(1, -1) + offset
test_TI = (test_I[:, -1] + target_step).reshape(-1, 1)

#print(train_I)
#print(train_TI)
#print(test_I)
#print(test_TI)

train_start_I = train_I[:, 0].reshape(-1, 1) * np.ones(window_n, np.int8)
train = (closes[train_I] - closes[train_start_I]) / closes[train_start_I]

test_start_I = test_I[:, 0].reshape(-1, 1) * np.ones(window_n, np.int8)
test = (closes[test_I] - closes[test_start_I]) / closes[test_start_I]

#print(train_start_I)
#print(train)
#print(test_start_I)
#print(test)

train_end_I = train_I[:, -1].reshape(-1, 1)
train_T = (closes[train_TI] - closes[train_end_I]) / closes[train_end_I]

test_end_I = test_I[:, -1].reshape(-1, 1)
test_T = (closes[test_TI] - closes[test_end_I]) / closes[test_end_I]

#print(train_T)
#print(test_T)

test_EPD = np.expand_dims(test, 1) + np.expand_dims(np.zeros(train_n), 1)

diffs = np.mean(np.abs(test_EPD - train), axis=2).T

all_matching = np.where(diffs < tolerance, train_T * np.ones(test_n), 0)

positions_sum = np.sum(np.sign(all_matching), axis=0).reshape(-1, 1)
positions = np.sign(positions_sum)

positions_nz = positions[positions != 0]
targets_nz = test_T[positions != 0]
target_pos_nz = np.sign(targets_nz)
postions_checked = np.where(positions_nz == target_pos_nz, 1, -1)

profit = postions_checked * targets_nz
#print(np.cumsum(profit))



vals, counts = np.unique(np.sign(profit), return_counts=True)
val_counts = dict(zip(vals, counts))
print(val_counts)
print(val_counts[-1])
print(val_counts[1])
print(val_counts[1] / (val_counts[-1] + val_counts[1]))
#print(counts[vals == -1])
#print(counts[0])


