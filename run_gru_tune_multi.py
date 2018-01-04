import random
import operator
import threading
import subprocess
import time

l = threading.Lock()
kill = False

def run_and_add(results):


    for i in range(10):
        nodes_log = random.random() * 3.0
        nodes = int(round(10.0 ** nodes_log))
        batch_log = random.random() * 3.0
        batch = int(round(10.0 ** batch_log))
        l.acquire()
        print "Running for", nodes, "nodes and", batch, "batch size"
        l.release()
        nrmse = float(subprocess.check_output(["python",  "run_gru_nrmse.py", "-d", "reddit", "-e", "all_hours", str(nodes), str(batch)]))

        l.acquire()
        results[(nodes, batch)] = nrmse
        l.release()


def print_results(results):
    time.sleep(10)
    l.acquire()
    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    for result in sorted_results:
        print "Nodes:", result[0][0], "  Batch size:", result[0][1], "  NRMSE:", result[1]
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
        print "Nodes:", result[0][0], "  Batch size:", result[0][1], "  NRMSE:", result[1]