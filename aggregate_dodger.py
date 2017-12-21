import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/dodger.csv", header=0, skiprows=[1, ])

curdate = None
dates = []
counts = []

agg_lines = 3
cur_lines = 0
cur_cars = 0
has_data = False
for i in tqdm(xrange(len(df))):
    if curdate is None:
        curdate = df["timestamp"][i]
    cur_lines += 1
    if df["count"][i] > -1:
        has_data = True
        cur_cars += df["count"][i]
    if cur_lines == agg_lines:
        if not has_data:
            cur_cars = -1
        dates.append(curdate)
        counts.append(cur_cars)
        cur_lines = 0
        cur_cars = 0
        has_data = False
        curdate = None


raw_data = {"date": dates, "count": counts}
df2 = pd.DataFrame(raw_data, columns=['date', 'count'])
df2.to_csv('data/dodger15.csv', index=False)


