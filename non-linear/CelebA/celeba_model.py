# Training Celeba concept models
# --label: choose one concept to train
# --train: skip training if 'no'

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import argparse
import sys
import time
import copy
#sys.path.insert(0, 'lib')
from pathlib import Path


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='train celeba models')
    parser.add_argument('--label', default='', type=str, help='concepts to learn: gender, attractive, glasses, age')
    parser.add_argument('--train', default='no', type=str, help='include training or not: yes, no')
    configs = parser.parse_args()
    
    # directory to save trained models
    model_dir = './models_celeba'
    Path(model_dir).mkdir(exist_ok=True)

    # Training four different classifiers
    if configs.label == 'gender':
        #Male
        label_index = 20
        output_file = model_dir+'/pytorch-rn50-gender.pth'
    elif configs.label == 'attractive':
        #Attactiveness
        label_index = 2
        output_file = model_dir+'/pytorch-rn50-attr.pth'
    elif configs.label == 'glasses':
        # Eyeglasses
        label_index = 15
        output_file = model_dir+'/pytorch-rn50-eyeglasses.pth'
    else: #if configs.label == 'young':
        # Young
        label_index = 39
        output_file = model_dir+'/pytorch-rn50-young.pth'

    # train model or not
    if configs.train == 'yes':
        TRAIN = True 
    else:
        TRAIN = False 

    return label_index, output_file, TRAIN 


# Modify pretrained resent50 model
def model_setup():
    model = models.__dict__['resnet50'](pretrained=True) 
    modules = list(model.children())
    modules.pop(-1)
    modules.pop(-1)
    temp = nn.Sequential(nn.Sequential(*modules))
    tempchildren = list(temp.children())
    tempchildren.append(
        nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.375),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.75),
            nn.Linear(512, 2),
        )
    )
    model = nn.Sequential(*tempchildren)

    for param in model.parameters():
        param.requires_grad = True
    return model


# get data and preprocess
def data_setup():
    train_dir = 'data/celeba'
    val_dir = 'data/celeba'
    data_root = 'data'

    # training data transformer
    train_tfms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomAffine(degrees=10.0, scale=(1, 1.2)),
                                     transforms.ColorJitter(brightness=0.2), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    train_dataset = datasets.CelebA(data_root, split="train", target_type=["attr"], transform=train_tfms)

    # test data transformer
    test_tfms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])


    valid_dataset = datasets.CelebA(data_root, split="valid", target_type=["attr"], transform=test_tfms)
    test_dataset = datasets.CelebA(data_root, split="test", target_type=["attr"], transform=test_tfms)

    # data loader for training
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    # data loader for validation 
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    # data loader for testing 
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, test_loader, val_loader


# training procedure  
def train(model, label_index, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dl_dict  ={'train': train_loader, 'val': val_loader}

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            iter_loss = 0.0
            iter_corrects = 0

            for inputs, labels in dl_dict[phase]:
                inputs = inputs.to(device)#cuda(non_blocking=True)
                labels = labels[:,label_index].to(device)#cuda(non_blocking=True)

                # Zero out the optimizer
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                iter_loss += loss.item() * inputs.size(0)
                iter_corrects += torch.sum(preds == labels.data)

            epoch_loss = iter_loss / len(dl_dict[phase].dataset)
            epoch_acc = iter_corrects.double() / len(dl_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history


# train and save the best model 
def train_save_model(label_index, output_file, train_loader, val_loader):
    # model and data setup
    model = model_setup()
    model = nn.DataParallel(model).to(device)#cuda()        
    criterion = nn.CrossEntropyLoss().to(device)#cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5, weight_decay=0.001)
    num_epochs = 1

    # training 
    model_ft, acc_hist = train(model, label_index, train_loader, val_loader, criterion, optimizer, num_epochs)

    # saving the best model
    torch.save(model_ft.state_dict(), output_file)


# test the trained model
def test_trained_model(label_index, output_file, train_loader, test_loader):
    model = model_setup()
    model = nn.DataParallel(model).to(device)#cuda()        
    model.eval()
    model.load_state_dict(torch.load(output_file))
    dl_dict  ={'train': train_loader, 'test': test_loader}

    for phase in ['train', 'test']:
        iter_corrects = 0
        for inputs, labels in dl_dict[phase]:
            inputs = inputs.to(device)#cuda(non_blocking=True)
            labels = labels[:,label_index].to(device)#cuda(non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # statistics
            iter_corrects += torch.sum(preds == labels.data)
        acc = iter_corrects.double() / len(dl_dict[phase].dataset)
        print(phase+' accuracy: {:.4f}'.format(acc))

    
# main
if __name__ == '__main__':
    # label_index: class label 
    # output_file: trained model weights file
    # TRAIN: boolean to trigger training
    label_index, output_file, TRAIN = parse_args()
    train_loader, test_loader, val_loader = data_setup()

    if TRAIN:
        train_save_model(label_index, output_file, train_loader, val_loader)

    test_trained_model(label_index, output_file, train_loader, test_loader)






