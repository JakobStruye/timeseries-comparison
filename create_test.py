from math import pi,sin
import numpy as np

mult_min = 1.0
mult_max = 1.0
grow_thresh = 10000
therange = 30000.0

piseq = np.arange(-pi, pi, pi/1500.0)
for i in range(int(therange)):
    x = piseq[i % len(piseq)]
    if i < grow_thresh:
        val = mult_min
    else:
        val = (((i - grow_thresh) / therange) * (mult_max - mult_min)) + mult_min
    print "%f,%f" % (piseq[i % len(piseq)], val * sin(x))
