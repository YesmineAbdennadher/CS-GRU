import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch.optim as optim
from snntorch import spikegen

import os
import os.path
import numpy as np
import time


from cifar10 import DVSCifar10
from ntidigits import NTIDIGITS
from model import Net
import config
from shd import my_Dataset, generate_dataset

import matplotlib.pyplot as plt

def main():

    args = config.get_args()
    if args.dataset == 'cifar10DVS':
        train_filename = "./dvs-cifar10/train/"
        test_filename = "./dvs-cifar10/test/"

        train_loader = DataLoader(DVSCifar10(train_filename),
                              batch_size=arg.batch_size,
                              shuffle=False)
    
        test_loader = DataLoader(DVSCifar10(test_filename),
                             batch_size=arg.batch_size,
                             shuffle=False)
    elif args.dataset == 'NTIDIGITS':
        dt= 5.0
        train_dataset = NTIDIGITS(root=os.path.join("./", "data"), download=True, train=True, dt=dt)
        test_dataset = NTIDIGITS(root=os.path.join("./", "data"), download=True, train=False, dt=dt)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        print('dataset loaded')
    elif args.dataset == 'MNIST':
        data_path='/tmp/data/mnist'
        transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)

    elif args.dataset == 'DVSGesture':
        data_dir = "./"
        trainset = DVS128Gesture(root=data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = DVS128Gesture(root=data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

        train_loader = torch.utils.data.DataLoader(dataset=trainset,batch_size=args.batch_size,shuffle=True, drop_last=True,num_workers=4,pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=4,pin_memory=True)
    elif args.dataset == 'shd':
        i = 0
        files = ['./data/shd/shd_test.h5','./data/shd/shd_train.h5']
        new_files = ['./data/shd/test_4ms/','./data/shd/train_4ms/']
        for file in new_files:
            if os.path.exists(file):
                print("Files to be generated exist")
            else:
                if i == 0:
                    generate_dataset(files[i],output_dir='./data/shd/test_4ms/',dt=4e-3)
                    i = 1
                else:
                    generate_dataset(files[i],output_dir='./data/shd/train_4ms/',dt=4e-3)
        train_dir = './data/shd/train_4ms/'
        train_files = [train_dir+i for i in os.listdir(train_dir)]

        test_dir = './data/shd/test_4ms/'
        test_files = [test_dir+i for i in os.listdir(test_dir)]
        
        train_dataset = my_Dataset(train_files)
        test_dataset = my_Dataset(test_files)
        
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset=='MNIST':
        channels = 1
        height = 28
        width = 28
        num_inputs= 28*28
        num_classes=10
    elif args.dataset=='NTIDIGITS':
        channels = 1
        num_inputs= 64
        num_classes=11
        height = 8
        width = 8
    elif args.dataset=='DVSGesture':
        channels = 2
        height = 128
        width = 128
        num_inputs= 2*128*128
        num_classes=11
    elif args.dataset == 'shd':
        channels = 7
        num_inputs = 700
        num_classes = 20
        height = 10
        width = 10


    model = Net(channels,128,num_classes,height,width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    start = time.time()

    for epoch in range(args.epochs):
        print('epoch number:',epoch)
        train_accuracy,train_loss = train(args,epoch, train_loader, model, criterion, optimizer)
        train_acc.append(train_accuracy)
        val_accuracy,val_loss = validate(args,epoch, test_loader, model, criterion)

def train(args, epoch, train_data,  model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    total_correct = 0 
    total_samples = 0
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs,targets = inputs.float(),targets.long()
        
        if args.dataset == 'MNIST':
            inputs = spikegen.rate(inputs, num_steps=args.T)
            inputs = inputs.permute(1, 0, 2, 3, 4)
        elif args.dataset == 'NTIDIGITS':
            inputs = inputs.reshape(inputs.size(0), inputs.size(1), 1, 8, 8) 
        elif args.dataset == 'shd':
            inputs = inputs.reshape(inputs.size(0), inputs.size(1), 7, 10, 10)
        optimizer.zero_grad()
        outputs = model(inputs)
        max_logits_over_time = outputs.max(dim=1)[0]
        if args.dataset == 'shd' :
            loss = criterion(max_logits_over_time, targets.squeeze(1))
        else:
            loss = criterion(max_logits_over_time, targets)
       
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = torch.argmax(max_logits_over_time, dim=1)  # Predicted class indices
            if args.dataset == 'shd':
                total_correct += (predictions == targets.squeeze(1)).sum().item()
            else:
                total_correct += (predictions == targets).sum().item()  # Count correct predictions
            total_samples += targets.size(0)  # Update total sample count

        # Calculate average loss and accuracy
        train_accuracy = (total_correct / total_samples)*100
        train_loss += loss.item()
    print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % train_accuracy)
    return train_accuracy,train_loss


def validate(args,epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_total_correct = 0 
    val_total_samples = 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs,targets = inputs.float(),targets.long()
            
            if args.dataset=='MNIST':
                inputs = spikegen.rate(inputs, num_steps=args.T)
                inputs = inputs.permute(1, 0, 2, 3, 4)
            elif args.dataset=='NTIDIGITS':
                inputs = inputs.reshape(inputs.size(0), inputs.size(1), 1, 8, 8)
            elif dataset == 'shd':
                inputs = inputs.reshape(inputs.size(0), inputs.size(1), 7, 10, 10)
            
            outputs_val = model(inputs)
            max_logits_over_time_val = outputs_val.max(dim=1)[0]
            if args.dataset == 'shd':
                loss = criterion(max_logits_over_time_val, targets.squeeze(1))
            else:
                loss = criterion(max_logits_over_time_val, targets)
         
            val_loss += loss.item()
            with torch.no_grad():
                predictions_val = torch.argmax(max_logits_over_time_val, dim=1)  # Predicted class indices
                if args.dataset == 'shd':
                    val_total_correct += (predictions_val == targets.squeeze(1)).sum().item()
                else:
                    val_total_correct += (predictions_val == targets).sum().item()  # Count correct predictions
                val_total_samples += targets.size(0)

            val_accuracy = (val_total_correct / val_total_samples)*100
    print('val_loss: %.6f' % (val_loss / len(val_data)), 'val_acc: %.6f' % val_accuracy)
    return val_accuracy,val_loss



if __name__ == '__main__':
	main()
