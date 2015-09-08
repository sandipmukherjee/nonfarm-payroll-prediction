__author__ = 'sandip'


import pickle
from collections import Counter
import csv
import pandas as pd
import time


f = open('monthly_applications.pickle', 'rb')
daily_applications = pickle.load(f)




#out_f = open('sorted_applications.csv', 'wb')

out_df = open('data_frame', 'wb')

timeseries = {}
for k,v in daily_applications.items():
    d = time.strptime(k,"%Y-%m")
    value = int(v)
    timeseries[d] = value

df = pd.DataFrame.from_dict(timeseries, orient="index")
sorted_df = df.sort()
numbers = sorted_df[0:32].values
l = []
for number in numbers:
    l.append(number[0])
d = [161.0, 44.0, 332.0, 249.0, 307.0, 74.0, 32.0, 132.0, 162.0, 346.0, 65.0, 129.0, 134.0, 239.0, 134.0, 363.0, 175.0, 245.0, 373.0, 196.0, 67.0, 84.0, 337.0, 159.0, 277.0, 315.0, 280.0, 182.0, 23.0, 77.0, 207.0, 184.0, 157.0, 2.0, 210.0, 171.0, 238.0, 88.0, 188.0, 78.0, 144.0, 71.0, -33.0, -16.0, 85.0, 82.0, 118.0, 97.0, 15.0, -86.0, -80.0, -214.0, -182.0, -172.0, -210.0, -259.0, -452.0, -474.0, -765.0, -697.0, -798.0, -701.0, -826.0, -684.0, -354.0, -467.0, -327.0, -216.0, -227.0, -198.0, -6.0, -283.0, 18.0, -50.0, 156.0, 251.0, 516.0, -122.0, -61.0, -42.0, -57.0, 241.0, 137.0, 71.0, 70.0, 168.0, 212.0, 322.0, 102.0, 217.0, 106.0, 122.0, 221.0, 183.0, 164.0, 196.0, 360.0, 226.0, 243.0, 96.0, 110.0, 88.0, 160.0, 150.0, 161.0, 225.0, 203.0, 214.0, 197.0, 280.0, 141.0, 203.0, 199.0, 201.0, 149.0, 202.0, 164.0, 237.0, 274.0, 84.0, 144.0, 222.0, 203.0, 304.0, 229.0, 267.0, 243.0, 203.0]
zipped = zip(l,d[-32:])
columns = ['prev_5','prev_4','prev_3','prev_2','prev_1','nonfarm']
df_final = pd.DataFrame(columns=columns)
print(zipped)
print(len(zipped))
s = l
i_s = 0
for n in d[-27:]:
    df_final.loc[i_s] = [s[i_s], s[i_s+1], s[i_s+2], s[i_s+3], s[i_s+4], n]
    i_s += 1
df_final.save('data_frame')


data = pickle.load(open('dataset/daily_applications_industry.pickle','rb'))

print(data)