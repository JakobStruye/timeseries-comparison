import random
import operator
import threading
import subprocess
import time

l = threading.Lock()
kill = False

def run_and_add(results, results_closer):


    for i in range(1000):
        nodes_log = random.random() * 2.5
        nodes = max(1, int(round(10.0 ** nodes_log)))
        lr = random.random() * 4.0 / 1000.0
        lookback = random.randint(10,200)
        retrain_interval = random.randint(50,1500)
        epochs = random.randint(20,250)

        l.acquire()
        print "Running for", nodes, "nodes,", retrain_interval, "retrain,", lr, "learning rate,", lookback, "lookback and", epochs, "epochs"
        l.release()
        return_val = subprocess.check_output(["python",  "run_gru_mase.py", "-d", "rssi", str(nodes), str(retrain_interval), str(lr), str(lookback), str(epochs)])
        mase = float(return_val.split(" ")[0])
        closer_rate  = float(return_val.split(" ")[1])

        l.acquire()
        results[(nodes, retrain_interval, lr, lookback, epochs)] = mase
        results_closer[(nodes, retrain_interval, lr, lookback, epochs)] = closer_rate
        l.release()


def print_results(results, results_closer):
    while True:
        time.sleep(300)
        l.acquire()
        sorted_results = sorted(results.items(), key=operator.itemgetter(1))
        for result in sorted_results:
            print "Nodes:", result[0][0], "  Retrain:", result[0][1], " LR:", result[0][2], " Lookback: ", result[0][3], " Epochs: ", result[0][4], "  MASE:", result[1], " closer:", results_closer[result[0]]
        print "DONE"
        l.release()
        if kill:
            return



if __name__ == '__main__':
    random.seed(6)


    results = dict()
    results_closer = dict()
    threads = []
    for _ in range(30):
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
        print "Nodes:", result[0][0], "  Retrain:", result[0][1], " LR:", result[0][2], " Lookback: ", result[0][3], " Epochs: ", result[0][4], "  MASE:", result[1], " closer:", results_closer[result[0]]
