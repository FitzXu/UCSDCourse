import os
import pickle
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

with open("./CelebA_cDCGAN_results/TenAEtrain_hist.pkl", "rb") as f:
    train_info = pickle.load(f)

D_Loss = train_info["D_losses"]
D_Loss = [item.item() for item in D_Loss]
G_Loss = train_info["G_losses"]
G_Loss = [item.item() for item in G_Loss]
auto_Loss = train_info['auto_losses']
auto_Loss = [item.item() for item in auto_Loss]
plt.plot(range(len(D_Loss)),D_Loss)
plt.plot(range(len(G_Loss)),G_Loss)
plt.plot(range(len(auto_Loss)),auto_Loss)
plt.savefig("./CelebA_cDCGAN_results/result/loss_curve_AECDCGAN5.png")