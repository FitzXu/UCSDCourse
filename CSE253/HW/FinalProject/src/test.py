import os, time, sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
from model import *
from dataloader import *
import matplotlib.pyplot as plt
from  torchvision.utils import *
import argparse

def test(batchsize, Gmodel_name, Dmodel_name):
	# hyper parameter
	data_dir = './data/'
	batch_size = batchsize
	img_size = 64
	model_path = "./model"
	Gmodel_name = Gmodel_name
	Dmodel_name = Dmodel_name
	result_path = "./result"
	class_num = 2
	# load images
	transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	with open(os.path.join(data_dir,"partition.pickle"), "rb") as f:
		partition = pickle.load(f)
	with open(os.path.join(data_dir,"ageLabel.pickle"), "rb") as f:
		agelabels = pickle.load(f)

	# fixed noise
	temp_z_ = torch.randn(class_num, 100)
	fixed_z_ = temp_z_
	fixed_y_ = torch.zeros(class_num, 1)
	for i in range(2):
		fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
		temp = torch.ones(class_num, 1) + i
		fixed_y_ = torch.cat([fixed_y_, temp], 0)

	onehot = torch.zeros(class_num, class_num)
	onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(class_num,1), 1).view(class_num, class_num, 1, 1)

	# fill stands for the "label image"
	fill = torch.zeros([class_num, class_num, img_size, img_size])
	for i in range(class_num):
		fill[i, i, :, :] = 1


	# network
	G = generator(128)
	D = discriminator(128)
	G.load_state_dict(torch.load(os.path.join(model_path, Gmodel_name)))
	D.load_state_dict(torch.load(os.path.join(model_path, Dmodel_name)))
	G.cuda()
	D.cuda()
	# Test Phase
	print('testing start!')
	for index in range(class_num):
		z_ = torch.randn((2, 100)).view(-1, 100, 1, 1)
		y_ = (torch.rand(2, 1)).type(torch.LongTensor).squeeze()
		y_[0] = index
		y_[1] = index   
		y_label_ = onehot[y_]
		z_, y_label_= Variable(z_.cuda()), Variable(y_label_.cuda())
		G_result = G(z_, y_label_)
		save_image(G_result,"./result/"+ str(index)+".png")
        
if __name__=="__main__":
	#parse the arguments
	parser = argparse.ArgumentParser()
	parser.description='Face Aging Testing Stage'
	parser.add_argument('--batchsize',help="batchsize used to train",default= 16)
	parser.add_argument('--Gmodel',help="GModel name to load",default= "MyModel")
	parser.add_argument('--Dmodel',help="DModel name to load",default= "MyModel")
	allPara = parser.parse_args()
	Gmodel_name = allPara.Gmodel
	Dmodel_name = allPara.Dmodel
	batchsize = int(allPara.batchsize)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
	test(batchsize, Gmodel_name, Dmodel_name)
	print("Testing Finish!!!")