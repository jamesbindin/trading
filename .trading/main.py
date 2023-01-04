import numpy as np
import pandas as pd
import pickle
import seaborn as sns
#import pdb; pdb.set_trace()

arr = np.load('./data/r/EURUSD_M1.npy')[:1000]
sns.relplot(arr)


#x = np.array(np.arange(0, 10))
#y = np.expand_dims(np.array(np.arange(0, 20)), 1)
#Y = x + y
#
#ta = np.expand_dims(np.arange(0,20), 1)
#
#res = arr[Y]
#arr[0:10]


