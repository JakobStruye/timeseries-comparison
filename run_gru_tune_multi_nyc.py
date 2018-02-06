import random
import operator
import threading
import subprocess
import time

l = threading.Lock()
kill = False

def run_and_add(results, results_closer):


    for i in range(3):
        #nodes_log = random.random() * 2.5
        #nodes = max(1, int(round(10.0 ** nodes_log)))
        nodes = random.randint(10,400)
        #nodes = 53
        lr_log = random.random() * -4.0
        lr = 10 ** lr_log
        #lr = random.uniform(0.0001,0.001)
        #lr = 0.00077
        lookback = random.randint(10,200)
        #lookback = 52
        retrain_interval = random.randint(1000,2000)
        retrain_interval = 1250
        epochs = 500
        #epochs = random.randint(60,130)

        l.acquire()
        print "Running for", nodes, "nodes,", retrain_interval, "retrain,", lr, "learning rate,", lookback, "lookback and", epochs, "epochs"
        l.release()
        return_val = subprocess.check_output(["python",  "run_gru_mase.py", "-d", "nyc_taxi", str(nodes), str(retrain_interval), str(lr), str(lookback), str(epochs)])
        #mase = float(return_val.split(" ")[0])
        #closer_rate  = float(return_val.split(" ")[1])
        mase = float(return_val)

        l.acquire()
        results[(nodes, retrain_interval, lr, lookback, epochs)] = mase
        l.release()


def print_results(results, results_closer):
    while True:
        time.sleep(5)
        l.acquire()
        sorted_results = sorted(results.items(), key=operator.itemgetter(1))
        for result in sorted_results:
            print "Nodes:", result[0][0], "  Retrain:", result[0][1], " LR:", result[0][2], " Lookback: ", result[0][3], " Epochs: ", result[0][4], "  MASE:", result[1]
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
        print "Nodes:", result[0][0], "  Retrain:", result[0][1], " LR:", result[0][2], " Lookback: ", result[0][3], " Epochs: ", result[0][4], "  MASE:", result[1]
