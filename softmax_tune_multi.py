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
"""

def run_and_add(results, results_closer):


    for i in range(3):
        nTrain = random.randint(500, 5000)

        batch = 2 ** random.randint(3,10)
        epochs = random.randint(20,200) if random.random() < 0.5 else random.randint(200,800)
        epochs_retrain = epochs if random.random() < 0.5 else random.randint(20, epochs+1)
        lr_log = (random.random() * -3.0) - 1
        lr = 10 ** lr_log * 5


        l.acquire()
        print "Running for", nTrain, "nTrain,", batch, "batch,", lr, "learning rate,", \
            epochs, "epochs and", epochs_retrain, "epochs_retrain"
        l.release()
        return_val = subprocess.check_output(["python",  "run_tm_softmax.py", "-d", "nyc_taxi", str(nTrain),
                                              str(batch), str(epochs), str(epochs_retrain), str(lr)])
        #mase = float(return_val.split(" ")[0])
        #closer_rate  = float(return_val.split(" ")[1])
        print "RET", return_val
        mase = float(return_val)

        l.acquire()
        results[(nTrain, batch, lr, epochs, epochs_retrain)] = mase
        l.release()


def print_results(results, results_closer):
    while True:
        time.sleep(5)
        l.acquire()
        sorted_results = sorted(results.items(), key=operator.itemgetter(1))
        for result in sorted_results:
            print "nTrain:", result[0][0], "  batch:", result[0][1], " LR:", result[0][2], " Epochs:", result[0][3],\
                " epochs_retrain:", result[0][4], "  MASE:", result[1]
        print "DONE"
        l.release()
        if kill:
            return



if __name__ == '__main__':


    results = dict()
    results_closer = dict()
    threads = []
    for _ in range(2):
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
        print "nTrain:", result[0][0], "  batch:", result[0][1], " LR:", result[0][2], " Epochs:", result[0][3], \
            " epochs_retrain:", result[0][4], "  MASE:", result[1]
