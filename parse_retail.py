import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    df = pd.read_csv("~/Downloads/retail.csv", header=0,  skiprows=[1,])

    curdate = None
    dates = []
    turnovers = []

    for i in tqdm(xrange(len(df))):
        if df["InvoiceDate"][i].split(":")[0] != curdate:
            if curdate is not None:
                dates.append(curdate)
                turnovers.append(turnover)
            curdate = df["InvoiceDate"][i].split(":")[0]
            turnover = 0
        turnover += df["Quantity"][i] * df["UnitPrice"][i]

    dates.append(curdate)
    turnovers.append(turnover)

    raw_data = {"date": dates, "turnover": turnovers}
    df2 = pd.DataFrame(raw_data, columns=['date', 'turnover'])
    df2.to_csv('retail.csv')


