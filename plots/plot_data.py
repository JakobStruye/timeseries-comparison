import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as tck
from sys import argv

import numpy as np

if __name__ == '__main__':


    labels = ["Sine", "Cosine"]
    t = np.arange(-0.2, 1.2, 0.01)

    fig, ax = plt.subplots(1)

    plt.xlim((-0.1, 1.1))
    plt.ylim((-1.2, 1.2))
    plt
    plt.grid(True)
    ax.plot(t, np.sin(2*np.pi*t), linestyle='--', label='sine', color='black')
    ax.plot(t, np.cos(2*np.pi*t), linestyle='-.', label='cosine', color='black')

    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $p$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.25))

    plt.yticks([])
 

    #plt.axvline(x=0.25)
    #plt.axvline(x=0.5)
    #plt.axvline(x=0.75)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    
    plt.show(block=True)
