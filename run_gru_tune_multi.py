import random
import operator
import threading
import subprocess
import time

l = threading.Lock()
kill = False

def run_and_add(results):


    for i in range(3):
        nodes_log = random.random() * 2.5
        nodes = max(1, int(round(10.0 ** nodes_log)))
        batch_log = random.random() * 3.0
        batch = max(1, int(round(10.0 ** batch_log)))
        lr_log = random.random() * 3.0
        lr = 10.0 ** - lr_log
        lookback = random.randint(1,50)
        l.acquire()
        print "Running for", nodes, "nodes,", batch, "batch size,", lr, "learning rate and", lookback, "lookback"
        l.release()
        mase = float(subprocess.check_output(["python",  "run_gru_mase.py", "-d", "nyc_taxi", str(nodes), str(batch), str(lr), str(lookback)]))

        l.acquire()
        results[(nodes, batch, lr, lookback)] = mase
        l.release()


def print_results(results):
    time.sleep(15)
    l.acquire()
    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    for result in sorted_results:
        print "Nodes:", result[0][0], "  Batch size:", result[0][1], " LR:", result[0][2], " Lookback: ", result[0][3], "  MASE:", result[1]
    print "DONE"
    l.release()
    if kill:
        return



if __name__ == '__main__':
    random.seed(6)


    results = dict()
    threads = []
    for _ in range(2):
        t = threading.Thread(target=run_and_add, args=(results,))
        threads.append(t)
        t.start()
    printer = threading.Thread(target=print_results, args=(results,))
    printer.start()

    for t in threads:
        t.join()
    kill = True
    printer.join()

    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    for result in sorted_results:
        print "Nodes:", result[0][0], "  Batch size:", result[0][1], " LR:", result[0][2], " Lookback: ", result[0][3], "  MASE:", result[1]
