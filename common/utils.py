import torch.nn as nn
import matplotlib.pyplot as plt
from common.defaults import RESULTS_LOCATION

def Softupdate(tau,target: nn.Module, current: nn.Module):
    for tar,curr in zip(target.parameters(),current.parameters()):
        tar.data = tau*tar.data + (1-tau)*curr.data


def GenericPlot(list_of_plots, plot_name = "generic_plot", xlabel="xlabel", ylabel="ylabel"):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    
    for X,Y in list_of_plots:
        if X is not None:
            ax.plot(X,Y)
        else:
            ax.plot(Y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Returns vs Epochs")
    fig.savefig(plot_name + '.jpg')
    # plt.clf()