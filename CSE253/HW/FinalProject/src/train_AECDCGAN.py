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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from model_modi import *
# label preprocess
img_size = 64
onehot = torch.zeros(2, 2)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
fill = torch.zeros([2, 2, img_size, img_size])
for i in range(2):
    fill[i, i, :, :] = 1

with open('./data/ageBalance.pickle', 'rb') as fp:
    y_gender_ = pickle.load(fp)

y_gender_ = torch.LongTensor(y_gender_).squeeze()

with open('./data/ageBalance2.pickle', 'rb') as fp:
    y_gender_2 = pickle.load(fp)

y_gender_2 = torch.LongTensor(y_gender_2).squeeze()

with open('./data/ageBalance3.pickle', 'rb') as fp:
    y_gender_3= pickle.load(fp)

y_gender_3 = torch.LongTensor(y_gender_3).squeeze()

print(y_gender_.size())
print(y_gender_2.size())
print(y_gender_3.size())
# fixed noise & label
temp_z0_ = torch.randn(4, 100)
temp_z0_ = torch.cat([temp_z0_, temp_z0_], 0)
temp_z1_ = torch.randn(4, 100)
temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

fixed_z_ = torch.cat([temp_z0_, temp_z1_], 0)
fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()

fixed_z_ = fixed_z_.view(-1, 100, 1, 1) #generate a fixed noise with 16 x 100 x 1 x 1
fixed_y_label_ = onehot[fixed_y_]
fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_label_.cuda(), volatile=True)

# training parameters
batch_size = 128
lr = 0.0001
train_epoch = 10

# data_loader
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
data_dir  = './celebAWrapper_1/'          # this path depends on your computer
data_dir2 = './celebAWrapper_2/'          # this path depends on your computer
data_dir3 = './celebAWrapper_3/'          # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
dset2 = datasets.ImageFolder(data_dir2, transform)
dset3 = datasets.ImageFolder(data_dir3, transform)
dset.imgs.sort()
dset2.imgs.sort()
dset3.imgs.sort()
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)
train_loader2 = torch.utils.data.DataLoader(dset2, batch_size=128, shuffle=False)
train_loader3 = torch.utils.data.DataLoader(dset3, batch_size=128, shuffle=False)

temp = plt.imread(train_loader2.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
autoEncoderLoss = nn.MSELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# results save folder
root = 'CelebA_cDCGAN_results/'
model = 'EnlargeAE'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['auto_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    auto_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()

    for loader, gender in [(train_loader,y_gender_), (train_loader2,y_gender_2),(train_loader3,y_gender_3)]:
        num_iter = 0
        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        for x_, _ in loader:
            x_ = Variable(x_.cuda())
            # train discriminator D
            D.zero_grad()   

            mini_batch = x_.size()[0]

            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                y_ = gender[batch_size*num_iter:]
            else:
                y_ = gender[batch_size*num_iter:batch_size*(num_iter+1)]

            y_fill_ = fill[y_]
            x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())

            D_result = D(x_, y_fill_).squeeze()
            
            D_real_loss = BCE_loss(D_result, y_real_)
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_] 
            y_fill_ = fill[y_] 
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())


            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()
            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.item())

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_] 
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

            G_result = G(x_, y_label_)
            D_result = D(G_result, y_fill_).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)
            autoL = autoEncoderLoss(G_result,x_)
            all_loss = G_train_loss
            all_loss.backward()
            G_optimizer.step()             
            G_losses.append(G_train_loss.item())
            auto_losses.append(all_loss.item())

            num_iter += 1

            if (num_iter % 100) == 0:
                print('%d - %d complete!' % ((epoch+1), num_iter))

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, auto: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(auto_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    # show_result((epoch+1), save=True, path=fixed_p)
    torch.save(G.state_dict(), root + model + 'generator_e'+str(epoch)+'_param.pkl')
    torch.save(D.state_dict(), root + model + 'discriminator_e'+str(epoch)+'_param.pkl')
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['auto_losses'].append(torch.mean(torch.FloatTensor(auto_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")

with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)