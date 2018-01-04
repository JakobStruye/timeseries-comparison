from run_gru import GruSettings, run_gru
import random
import operator

if __name__ == '__main__':
    random.seed(6)
    settings = GruSettings()
    settings.max_verbosity = 0
    settings.epochs = 1
    settings.online = False

    results = dict()
    for _ in range(100):
        nodes_log = random.random() * 3.0
        nodes = int(round(10.0 **  nodes_log))
        batch_log = random.random() * 3.0
        batch = int(round(10.0 ** batch_log))
        print "Running for", nodes, "nodes and", batch, "batch size"
        settings.nodes = nodes
        settings.batch_size = batch
        settings.finalize()
        nrmse = run_gru(settings)
        results[(nodes, batch)] = nrmse
    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    for result in sorted_results:
        print "Nodes:", result[0][0], "  Batch size:", result[0][1], "  NRMSE:", result[1]