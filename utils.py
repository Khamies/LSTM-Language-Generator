import matplotlib.pyplot as plt
from data.ptb import PTB
from settings import training_setting
import torch


def get_batch(batch):
  sentences = batch["input"]
  target = batch["target"]
  sentences_length = batch["length"]

  return sentences, target, sentences_length



def plot(loss, mode):

    plt.plot(loss, label= mode)

    plt.legend()
    plt.show()


