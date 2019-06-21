import os, time, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
# import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model_modi import *
import sys
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def variation_epoch(num_epoch):
    img_size = 64
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
    fill = torch.zeros([2, 2, img_size, img_size])
    for i in range(2):
        fill[i, i, :, :] = 1

    # fixed noise & label
    temp_z0_ = torch.randn(1, 100)

    fixed_z_ = torch.cat([temp_z0_, temp_z0_], 0)
    fixed_y_ = torch.cat([torch.zeros(1), torch.ones(1)], 0).type(torch.LongTensor).squeeze()

    fixed_z_ = fixed_z_.view(-1, 100, 1, 1) #generate a fixed noise with 16 x 100 x 1 x 1
    fixed_y_label_ = onehot[fixed_y_]
    with torch.no_grad():
        fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda()), Variable(fixed_y_label_.cuda())

    

    size_figure_grid = num_epoch//2
    fig, ax = plt.subplots(2, size_figure_grid,figsize=(20, 5))
    fig.tight_layout()
    plt.subplots_adjust(wspace =0.2, hspace =0.2)
    for i, j in itertools.product(range(2), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid):
        G = generator()
        G.cuda()
        G.load_state_dict(torch.load('./CelebA_cDCGAN_results/model/CelebA_cDCGAN_generator_e'+str(k*2)+'_param.pkl'))
        G.eval()
        test_images = G(fixed_z_, fixed_y_label_)
        G.train()
        ax[0, k].cla()
        ax[0, k].imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[1, k].cla()
        ax[1, k].imshow((test_images[1].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)


    label = 'Generated Faces Across different epochs'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig('./CelebA_cDCGAN_results/images/epoch_variation.png')

def generate_faces():
    img_size = 64
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
    fill = torch.zeros([2, 2, img_size, img_size])
    for i in range(2):
        fill[i, i, :, :] = 1
    
    
    G = generator()
    G.cuda()
    G.load_state_dict(torch.load('./CelebA_cDCGAN_results/model/AEAEgenerator_e1_param.pkl'))
    G.eval()
    G.train()
        

    size_figure_grid = 10
    fig, ax = plt.subplots(2, size_figure_grid,figsize=(20, 5))
    fig.tight_layout()
    plt.subplots_adjust(wspace =0.2, hspace =0.2)

    for i, j in itertools.product(range(2), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid):
                
        # fixed noise & label
        temp_z0_ = torch.randn(1, 100)
        fixed_z_ = torch.cat([temp_z0_, temp_z0_], 0)
        fixed_y_ = torch.cat([torch.zeros(1), torch.ones(1)], 0).type(torch.LongTensor).squeeze()
        fixed_z_ = fixed_z_.view(-1, 100, 1, 1) #generate a fixed noise with 16 x 100 x 1 x 1
        fixed_y_label_ = onehot[fixed_y_]
        with torch.no_grad():
            fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda()), Variable(fixed_y_label_.cuda())
        test_images = G(fixed_z_, fixed_y_label_)
        ax[0, k].cla()
        ax[0, k].imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[1, k].cla()
        ax[1, k].imshow((test_images[1].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)


    label = 'Generated faces of different identity'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig('./CelebA_cDCGAN_results/images/identity_variation.png')


def show_noise_morp(show=False, save=False, path='result.png'):
    source_z_ = torch.randn(10, 100)
    z_ = torch.zeros(100, 100)
    for i in range(5):
        for j in range(10):
            z_[i*20 + j] = (source_z_[i*2+1] - source_z_[i*2]) / 9 * (j+1) + source_z_[i*2]

    for i in range(5):
        z_[i*20+10:i*20+20] = z_[i*20:i*20+10]

    y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)], 0).type(torch.LongTensor).squeeze()
    y_ = torch.cat([y_, y_, y_, y_, y_], 0)
    y_label_ = onehot[y_]
    z_ = z_.view(-1, 100, 1, 1)
    y_label_ = y_label_.view(-1, 2, 1, 1)

    z_, y_label_ = Variable(z_.cuda(), volatile=True), Variable(y_label_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_, y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(img_size, img_size))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10 * 10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()

def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print ("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pickle.load(p)
    except ValueError:
        print ('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pickle.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pickle.dump(data_object, pickle_file)
    pickle_file.close()

def AE_diff_epoch(img_path):

    isCrop = False
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    img_PIL = Image.open(img_path).convert('RGB')
    ori = transform(img_PIL)


    # fixed noise & label    
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
    
    count = 0
    
    size_figure_grid = 3
    fig, ax = plt.subplots(2, size_figure_grid,figsize=(20, 5))
    fig.tight_layout()
    plt.subplots_adjust(wspace =0.2, hspace =0.2)

    for i, j in itertools.product(range(2), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    
    ori = torch.unsqueeze(ori, 0)

    ori = Variable(ori.cuda())
    ax[0, 0].imshow((ori[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    ax[1, 0].imshow((ori[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    for k in range(size_figure_grid-1):
        print("epochs: ",k)
        G = generator()
        G.load_state_dict(torch.load('./CelebA_cDCGAN_results/EnlargeAEgenerator_e'+str(k)+'_param.pkl'))        
        G.cuda()    
        G.eval()
        G.train()

        y_ = np.array([0])
        y_label_ = onehot[y_]
        y_label_ = Variable(y_label_.cuda())
        test_images = G(ori, y_label_)

        ax[0, k+1].imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

        y_ = np.array([1])
        y_label_ = onehot[y_]
        y_label_ = Variable(y_label_.cuda())
        test_images = G(ori, y_label_)

        ax[1, k+1].imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    plt.savefig('./CelebA_cDCGAN_results/images/EnlargeAE.png')

def AE_diff_identity(dir_path):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    # fixed noise & label    
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)

    # model
    G = generator()
    G.load_state_dict(torch.load('./CelebA_cDCGAN_results/twelveAEgenerator_e4_param.pkl'))
    G.cuda()    
    G.eval()
    G.train()

    # load images
    image_paths = [os.path.join(dir_path,img) for img in os.listdir(dir_path)]    
    size_figure_grid = len(image_paths)
    fig, ax = plt.subplots(2, size_figure_grid,figsize=(15, 5))
    fig.tight_layout()

    for i, j in itertools.product(range(2), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    

    for index, img_path in enumerate(image_paths):        
        img_PIL = Image.open(img_path).convert('RGB')
        ori = transform(img_PIL)
        ori = torch.unsqueeze(ori, 0)
        ori = Variable(ori.cuda())

        ax[0, index].imshow((ori[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        
        y_ = np.array([0])
        y_label_ = onehot[y_]
        y_label_ = Variable(y_label_.cuda())
        test_images = G(ori, y_label_)

        ax[1, index].imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        print("{} has been aged!!!".format(img_path))


    plt.savefig('./CelebA_cDCGAN_results/images/AE_diff_identity12.png')

def diff_alpha(img_path):

    isCrop = False
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    img_PIL = Image.open(img_path).convert('RGB')
    ori = transform(img_PIL)


    # fixed noise & label    
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
    
    count = 0
    
    G_list = ['elevenAEgenerator_e5_param.pkl','twelveAEgenerator_e4_param.pkl','FifthAEgenerator_e5_param.pkl']
    size_figure_grid = len(G_list) + 1
    fig, ax = plt.subplots(1, size_figure_grid,figsize=(20, 5))
    fig.tight_layout()
    plt.subplots_adjust(wspace =0.2, hspace =0.2)

    for i in range(size_figure_grid):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    
    ori = torch.unsqueeze(ori, 0)

    ori = Variable(ori.cuda())
    ax[0].imshow((ori[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    for k in range(size_figure_grid-1):
        print("epochs: ",k)
        G = generator()
        G.load_state_dict(torch.load('./CelebA_cDCGAN_results/'+G_list[k]))        
        G.cuda()    
        G.eval()
        G.train()

        y_ = np.array([0])
        y_label_ = onehot[y_]
        y_label_ = Variable(y_label_.cuda())
        test_images = G(ori, y_label_)

        ax[k+1].imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)


    plt.savefig('./CelebA_cDCGAN_results/images/diff_alpha.png')



def ok(dir_path):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    # fixed noise & label    
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)

    # model
    G = generator()
    G.load_state_dict(torch.load('./CelebA_cDCGAN_results/twelveAEgenerator_e4_param.pkl'))
    G.cuda()    
    G.eval()
    G.train()

    # load images
    image_paths = [os.path.join(dir_path,img) for img in os.listdir(dir_path)]    
    # size_figure_grid = len(image_paths)
    
    img_PIL = Image.open(image_paths[0]).convert('RGB')
    ori = transform(img_PIL)
    ori = torch.unsqueeze(ori, 0)
    ori = Variable(ori.cuda())

    y_ = np.array([0])
    y_label_ = onehot[y_]
    y_label_ = Variable(y_label_.cuda())
    test_images = G(ori, y_label_)

    plt.imshow((test_images[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    plt.axis('off')
    plt.savefig('./test.jpg')


if __name__=="__main__":
    # AE_diff_identity("./test2")
    
    diff_alpha("./test_image/000010.jpg")












