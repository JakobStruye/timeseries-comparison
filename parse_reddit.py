import pandas as pd
from tqdm import tqdm
import json
import datetime
from collections import OrderedDict

if __name__ == '__main__':

    #df = pd.read_json("")

    counts1 = OrderedDict()
    counts2 = OrderedDict()
    counts3 = OrderedDict()


    file_path = "/Users/jstr/Downloads/RC_2015-01"
    line_count = 53851542
    with open(file_path) as infile:
        for line in tqdm(infile, total=line_count):
            data = json.loads(line)
            timestamp_unix = data["created_utc"]
            timestamp = datetime.datetime.fromtimestamp(int(timestamp_unix))
            #timestamp_str1 = timestamp.strftime('%Y-%m-%d %H:') + str((int(timestamp.strftime("%M")) // 10) * 10).zfill(2)
            #timestamp_str2 = timestamp.strftime('%Y-%m-%d %H:') + str((int(timestamp.strftime("%M")) // 15) * 15).zfill(2)
            #timestamp_str3 = timestamp.strftime('%Y-%m-%d %H:') + str((int(timestamp.strftime("%M")) // 30) * 30).zfill(2)
            timestamp_str1 = timestamp.strftime('%Y-%m-%d %H:') + str((int(timestamp.strftime("%M")) // 5) * 5).zfill(2)

            try:
                cur_count1 = counts1[timestamp_str1]
            except:
                cur_count1 = 0
            #try:
            #    cur_count2 = counts2[timestamp_str2]
            #except:
            #    cur_count2 = 0
            #try:
            #    cur_count3 = counts3[timestamp_str3]
            #except:
            #    cur_count3 = 0
            counts1[timestamp_str1] = cur_count1 + 1
            #counts2[timestamp_str2] = cur_count2 + 1
            #counts3[timestamp_str3] = cur_count3 + 1



    with open('reddit4.txt', 'w') as outfile:
        outfile.write("time,count\n")
        for timestamp, count in tqdm(counts1.iteritems()):
            outfile.write(timestamp+","+str(count)+'\n')

    # with open('reddit5.txt', 'w') as outfile:
    #     outfile.write("time,count\n")
    #     for timestamp, count in tqdm(counts2.iteritems()):
    #         outfile.write(timestamp+","+str(count)+'\n')
    #
    # with open('reddit6.txt', 'w') as outfile:
    #     outfile.write("time,count\n")
    #     for timestamp, count in tqdm(counts3.iteritems()):
    #         outfile.write(timestamp+","+str(count)+'\n')


