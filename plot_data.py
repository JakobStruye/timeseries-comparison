import pandas as pd
from matplotlib import pyplot as plt
from sys import argv
import argparse

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

    for col in args.column:
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
        handle, = plt.plot(float_seq)
        handles.append(handle)
    if args.xrange:
        plt.xlim(args.xrange)
    if args.yrange:
        plt.ylim(args.yrange)

    if len(handles) == 2:
        plt.legend(handles, labels)


    print "Plotting", len(float_seq), "data points ranging between", min(float_seq), "and", max(float_seq)
    plt.show(block=True)
