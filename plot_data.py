import pandas as pd
from matplotlib import pyplot as plt
from sys import argv
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot data from csv.')
    parser.add_argument("filename", help="Name of csv file, may include path",
                        type=str)
    parser.add_argument("column", help="Index of column to use, starting from 1",
                        type=int, nargs="+")
    parser.add_argument("-d", "--delimiter", help="Delimiter character")
    parser.add_argument("-s", "--skip", help="Number of lines to skip at top", type=int)
    parser.add_argument("-x", "--xrange", help="X-axis range", nargs="+", type=float)
    parser.add_argument("-y", "--yrange", help="Y-axis range", nargs="+", type=float)

    args = parser.parse_args()

    labels = ["Actual", "Predicted"]
    handles = []

    df = pd.read_csv(args.filename, header=0, delimiter=args.delimiter, skiprows=[] if not args.skip else range(1, args.skip+1))
    plt.figure()

    styles = { 0: "--",
               1: "-"}

    colors = { 0: "darkorange",
               1: "forestgreen"}

    alphas = { 0: 0.6}

    counter = 0
    for col in args.column:
        try:
            style = styles[counter]
        except:
            style = "-"
        try:
            color = colors[counter]
        except:
            color = None

        try:
            alpha = alphas[counter]
        except:
            alpha = 1
        counter += 1

        sequence = df[df.keys().tolist()[col]]

        float_seq = []
        missing_count = 0
        for item in sequence:
            try:
                float_seq.append(float(item))
                if item == -1.0:
                    missing_count += 1
            except ValueError:
                pass

        old_len = len(sequence)
        new_len = len(float_seq)
        if old_len != new_len:
            print "", old_len-new_len, "of", old_len, "items could not be converted to float"
        if missing_count > 0:
            print "", missing_count, "of", new_len, "points exactly -1.0, possibly indicating missing data"
        if color is not None:
            handle, = plt.plot(float_seq, linestyle=style, alpha=alpha, color=color)
        else:
            handle, = plt.plot(float_seq, linestyle=style, alpha=alpha)
        handles.append(handle)
    if args.xrange:
        plt.xlim(args.xrange)
    if args.yrange:
        plt.ylim(args.yrange)

    plt.xlabel("Seconds since 2017-12-26 00:00:00")

    #days = ["2017-12-26", "2017-12-27", "2017-12-28", "2017-12-29", "2017-12-30", "2017-12-31", "2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-06", "2018-01-07", "2018-01-08", "2018-01-09", "2018-01-10", "2018-01-11", "2018-01-12", "2018-01-13", "2018-01-14", "2018-01-15"]
    #plt.xticks(np.arange(0, 60 * 24 * len(days) + 1, 24*60*3), days[0::3], rotation = 20)
    #for i in range(len(days)):
    #    plt.axvline(x=(i*24*60), lw=1, color='grey')
    plt.ylabel("Signal Strength (dBm)")

    if len(handles) <= 2:
        plt.legend(handles, labels)
    #plt.tight_layout()

    print "Plotting", len(float_seq), "data points ranging between", min(float_seq), "and", max(float_seq)
    plt.show(block=True)
