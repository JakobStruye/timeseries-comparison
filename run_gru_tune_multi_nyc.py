import random
import operator
import threading
import subprocess
import time

random.seed(0)
l = threading.Lock()
kill = False

"""
Command line args
(0: filename)
(1: -d)
(2: dataset name)
3: nodes
4: retrain
5: lr
6: lookback
7: epochs
8: online
9: batch
10: lb_as_features
11: feature_count
12: implementation 
13: eps
"""

def run_and_add(results, results_closer):


    for i in range(1000):
        #nodes_log = random.random() * 2.5
        #nodes = max(1, int(round(10.0 ** nodes_log)))
        nodes = random.randint(40,400)
        #nodes = 53
        lr_log = random.random() * 3.0 - 4.0
        lr = 10 ** lr_log
        #lr = random.uniform(0.0001,0.001)
        #lr = 0.00077
        lookback = random.randint(25,125)
        #lookback = 52
        #retrain_interval = random.randint(1000,2000)
        retrain_interval = 2500
        epochs = 200
        #epochs = random.randint(60,130)
        online = False
        batch = 64
        lb_as_features = True
        feature_count = 1
        implementation = "lstm"
        eps = '1e-3' if random.random() <  0.5 else '1e-7'

        l.acquire()
        print "Running for", nodes, "nodes,", retrain_interval, "retrain,", lr, "learning rate,",\
            lookback, "lookback and", epochs, "epochs", "not" if not online else "", "online", \
            batch, "batch", "not" if lb_as_features else "", "lb_as_ft", feature_count, "feature_count", \
            implementation, eps, 'eps'
        l.release()
        return_val = subprocess.check_output(["python",  "run_gru_mase.py", "-d", "nyc_taxi", str(nodes),
                                              str(retrain_interval), str(lr), str(lookback), str(epochs),
                                              str(online), str(batch), str(lb_as_features), str(feature_count),
                                                                           implementation, eps])
        #mase = float(return_val.split(" ")[0])
        #closer_rate  = float(return_val.split(" ")[1])
        mase = float(return_val)

        l.acquire()
        results[(nodes, retrain_interval, lr, lookback, epochs, online, batch, lb_as_features,feature_count, implementation, eps)] = mase
        l.release()


def print_results(results, results_closer):
    while True:
        time.sleep(300)
        l.acquire()
        sorted_results = sorted(results.items(), key=operator.itemgetter(1))
        for result in sorted_results:
            print "Nodes:", result[0][0], "  Retrain:", result[0][1], " LR:", result[0][2], " Lookback:", result[0][3],\
                " Epochs:", result[0][4], " Online:", result[0][5], " Batch:", result[0][6], " as_ft:", result[0][7],\
                " feats:", result[0][8], " impl:", result[0][9], " eps:", result[0][10], "  MASE:", result[1]
        print "DONE"
        l.release()
        if kill:
            return



if __name__ == '__main__':


    results = dict()
    results_closer = dict()
    threads = []
    for _ in range(28):
        t = threading.Thread(target=run_and_add, args=(results,results_closer))
        threads.append(t)
        t.start()
    printer = threading.Thread(target=print_results, args=(results,results_closer))
    printer.start()

    for t in threads:
        t.join()
    kill = True
    printer.join()

    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    for result in sorted_results:
        print "Nodes:", result[0][0], "  Retrain:", result[0][1], " LR:", result[0][2], " Lookback:", result[0][3], \
            " Epochs:", result[0][4], " Online:", result[0][5], " Batch:", result[0][6], " as_ft:", result[0][7], \
            " feats:", result[0][8], " impl:", result[0][9], " eps:", result[0][10], "  MASE:", result[1]
