import os
import PIL.Image as I
import copy
import pickle
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_EVEN
import eagerpy as ep
import sys
sys.path.insert(0,'../../lib')
import foolbox
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torchvision.models as models
from torchvision import transforms


## CustomData for adversarial examples
class CustomData(Dataset):

    def __init__(self, pd_data, root_dir='data/celeba/img_align_celeba', all_concepts=False, img_pkl_filename=None, is_train=False):

        self.root_dir = root_dir
        self.data_frame = pd_data
        self.all = all_concepts
        if img_pkl_filename is None:
            self.img_pkl_file = None
        else:
            self.img_pkl_file = pickle.load(open(img_pkl_filename, 'rb'))
        self.is_train = is_train

        if self.is_train:
            self.transform_norm = transforms.Compose([transforms.RandomResizedCrop(224),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomAffine(degrees=10.0, scale=(1, 1.2)),
                                                      transforms.ColorJitter(brightness=0.2), transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])

        else:
            self.transform_norm = transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])

        self.transform_raw = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                 transforms.ToTensor()])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.data_frame.iloc[index, 0]

        if self.all:
            img_label = list(self.data_frame.iloc[index, 1:])
        else:
            img_label = self.data_frame.iloc[index, 1]

        if self.img_pkl_file is not None:
            if img_name not in self.img_pkl_file.keys():
                raise FileNotFoundError

            image = self.img_pkl_file[img_name]
        else:
            img_path = os.path.join(self.root_dir, img_name)

            image = I.open(img_path)

        image_raw = self.transform_raw(image)
        image_norm = self.transform_norm(image)

        # the sample contains image tensor, label of the concept, image name
        sample = {'image': image_raw, 'image_norm': image_norm, 'labels': img_label, 'img_name': img_name}

        return sample


# #### Normalize the image separately
# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class Normalize(nn.Module):
    def __init__(self, mean=None, std=None):
        super(Normalize, self).__init__()
        if mean is None:
            self.mean = torch.Tensor([0.485, 0.456, 0.406])
        else:
            self.mean = torch.Tensor(mean)

        if std is None:
            self.std = torch.Tensor([0.229, 0.224, 0.225])
        else:
            self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


## Flatten Class to create pytorch resnet-50 model
class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        if self.full:
            return x.view(-1)
        else:
            return x.view(x.size(0), -1)


## AdaptiveConcatPool2d Class to create pytorch resnet-50 model
class AdaptiveConcatPool2d(nn.Module):
    # from pytorch
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    "Output will be 2*sz or 2 if sz is None"

    def __init__(self, sz=None):
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


## my_head helper function to create pytorch resnet-50 model
def myhead(nf, nc):
    # nf = number of features
    # nc = number of classes
    # the dropout is needed otherwise you cannot load the weights

    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(nf),
        nn.Dropout(p=0.375),
        nn.Linear(nf, 512),
        nn.ReLU(True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.75),
        nn.Linear(512, nc),
    )

    return head


## get pytorch resnet-50 model from the model weights
def get_pytorch_model(weights_dir='celeb_individual/saved_weights/original_data/models',
                      weights_file='pytorch-rn50-gender.pth', is_torch=True, num_classes=2,
                      cuda_flag=True):
    model = models.resnet50()

    modules = list(model.children())
    modules.pop(-1)
    modules.pop(-1)
    # print(modules)

    temp = nn.Sequential(nn.Sequential(*modules))
    tempchildren = list(temp.children())
    tempchildren.append(myhead(4096, num_classes))
    # print(tempchildren)

    model = nn.Sequential(*tempchildren)

    """
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
    """

    if weights_dir is None or weights_file is None:
        if str(next(model.parameters()).device) == 'cpu' and torch.cuda.is_available():
            model.cuda()
        return model

    weights = torch.load(os.path.join(weights_dir, weights_file))

    if is_torch:
        # using pytorch weights:
        model.load_state_dict(weights)
    else:
        model.load_state_dict(weights['model'])

    # shift the model to gpu if cuda is available and flag is true
    if str(next(model.parameters()).device) == 'cpu' and torch.cuda.is_available():
        if cuda_flag:
            model.cuda()

    return model

## get pytorch mobilenet model from the model weights
def get_pytorch_mobilenet_model(weights_dir=None,
                      weights_file=None, is_torch=True, num_classes=2,
                      cuda_flag=True):

    from torchvision.models import mobilenet_v2
    from collections import OrderedDict

    num_class=2
    model = mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1]=torch.nn.Sequential(OrderedDict([('fc1',torch.nn.Linear(in_features, num_class)),('activation1', torch.nn.Softmax())]))
    #model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model = torch.nn.DataParallel(model).cuda()


    #model = models.mobilenet_v2()

    #modules = list(model.children())
    #modules.pop(-1)
    ## print(modules)

    #temp = nn.Sequential(nn.Sequential(*modules))
    #tempchildren = list(temp.children())
    #tempchildren.append(myhead(4096, num_classes))
    ## print(tempchildren)

    #model = nn.Sequential(*tempchildren)

    #if weights_dir is None or weights_file is None:
    #    if str(next(model.parameters()).device) == 'cpu' and torch.cuda.is_available():
    #        model.cuda()
    #    return model

    #weights = torch.load(os.path.join(weights_dir, weights_file))
    weights = torch.load(weights_file)
    model.load_state_dict(weights['state_dict'])

    #if is_torch:
    #    # using pytorch weights:
    #    model.load_state_dict(weights['state_dict'])
    #else:
    #    model.load_state_dict(weights['model'])

    # shift the model to gpu if cuda is available and flag is true
    if str(next(model.parameters()).device) == 'cpu' and torch.cuda.is_available():
        if cuda_flag:
            model.cuda()

    return model

## get pytorch resnet model from the model weights
def get_pytorch_resnet_model(weights_dir=None,
                      weights_file=None, is_torch=True, num_classes=2,
                      cuda_flag=True):

    from torchvision.models import mobilenet_v2
    from collections import OrderedDict

    model = models.resnet50()

    num_class=2
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

    #for param in model.fc.parameters():
    for param in model.parameters():
        param.requires_grad = True


    # Wrap the model into DataParallel
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(weights_file))

    # Criterion:
    #criterion = nn.CrossEntropyLoss().cuda()

    if str(next(model.parameters()).device) == 'cpu' and torch.cuda.is_available():
        if cuda_flag:
            model.cuda()

    return model


## get foolbox resnet-50 model from the model weights
def get_foolbox_model(modeltype='resnet50', weights_file='pytorch-r50-young-new.pth', dev='cuda:1',
                      num_classes=2, is_torch=True, preprocess=True, models_dir=None):
    model = get_pytorch_model(weights_dir=models_dir, weights_file=weights_file,
                           is_torch=is_torch, num_classes=num_classes)
    model.eval()
    model = model.to(dev)
    if preprocess:
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    else:
        preprocessing = None
    fmodel = foolbox.PyTorchModel(model, device=dev, bounds=(0, 1), preprocessing=preprocessing)

    return model, fmodel


## Get dictionary tuple for CelebA data with labels and corresponding img
def get_pdadvx_dict(df_advx):
    dict_advx = (df_advx.set_index('image_id')).to_dict(orient='index')
    for img in dict_advx:
        dict_tuple = dict_advx[img]
        dict_tuple['pretty'] = dict_tuple.pop('Attractive')
        dict_tuple['glasses'] = dict_tuple.pop('Eyeglasses')
        dict_tuple['gender'] = dict_tuple.pop('Male')
        dict_tuple['age'] = dict_tuple.pop('Young')
        dict_advx[img] = dict_tuple

    return copy.deepcopy(dict_advx)


## Get dictionary tuple for UTKFace data with labels and corresponding img
def get_pdadvx_dict_utkface(df_advx):
    dict_advx = (df_advx.set_index('name')).to_dict(orient='index')
    for img in dict_advx:
        dict_tuple = dict_advx[img]
        dict_tuple['age'] = dict_tuple.pop('age')
        dict_tuple['gender'] = dict_tuple.pop('gender')
        dict_tuple['ethnicity'] = dict_tuple.pop('ethnicity')
        dict_advx[img] = dict_tuple

    return copy.deepcopy(dict_advx)


## Set GPU device for the current session
def set_device(device_name="cuda:0", num_workers=1):
    dev = device_name if torch.cuda.is_available() else "cpu"
    kwgs = {'num_workers': num_workers, 'pin_memory': True} if dev == device_name else {}

    return dev, kwgs


## get dataloader dictionary
def get_dataloader_dict(data_loader):
    data_loader_dict = {}
    for batch_id, b in enumerate(data_loader):
        x, img_names, labels, x_raw = b['image_norm'], b['img_name'], b['labels'], b['image']
        for img_index, img_name in enumerate(img_names):
            if img_name not in data_loader_dict.keys():
                data_loader_dict[img_name] = {'X': x[img_index], 'X_raw': x_raw[img_index],
                                              'labels': [labels[i][img_index] for i in range(len(labels))]}

    return data_loader_dict


## given dataframe of adversarial candidate images, get dataloader
def get_dataloader(kwargs_func, batch_size, pd_advx=None, root_dir=None, img_pkl_file=None, is_for_train=False):
    dataset_all = CustomData(pd_data=pd_advx, root_dir=root_dir, all_concepts=True,
                             img_pkl_filename=img_pkl_file, is_train=is_for_train)

    g = torch.Generator()
    g.manual_seed(0)

    data_loader = DataLoader(dataset_all, batch_size=batch_size, **kwargs_func)
    #data_loader = DataLoader(dataset_all, batch_size=batch_size, sampler=RandomSampler(dataset_all), **kwargs_func)

    return data_loader


## given the adversarial batch list, dictionary (for labels), model, concept, get the accuracy for that concept
def get_gen_acc(advx_batch_list, advx_dict, model, concept='age', img_key='advx'):
    acc_after = 0

    for batch_index in advx_batch_list:
        img_names = advx_batch_list[batch_index]['img_names']
        labels = []

        for _, img in enumerate(img_names):
            labels.append(advx_dict[img][concept])
        labels = torch.tensor(labels).cpu()
        advs_ = advx_batch_list[batch_index][img_key]
        if type(advs_) == list or len(advs_) == 1:
            advs_ = advs_[0]

        yp_after = model(advs_.cuda()).cpu()
        acc_after += (yp_after.max(dim=1)[1] == labels).sum().item()

    return acc_after


## given the regular dataloader, model and concept_index, get the regular accuracy for that concept
def get_gen_acc_metrics(dl, model, concept_index):
    acc_after = 0

    for batch_index, batch in enumerate(dl):
        concept_label = batch['labels'][concept_index]
        imgs = batch['image_norm']
        yp_after = model(imgs.cuda()).cpu()

        acc_after += (yp_after.max(dim=1)[1] == concept_label).sum().item()

    return acc_after


## helper function to assert that the adversarial image is lp bound wrt orig_img
def helper_lp_bound(advx, x_orig, norms, eps):
    # get l2/linf norm of perturbation
    pert = ep.astensor(advx - x_orig)
    if norms == 'l2':
        norms = pert.norms.l2(axis=(1, 2, 3)).numpy()
    if norms == 'l1':
        norms = pert.norms.l1(axis=(1, 2, 3)).numpy()
    else:
        norms = pert.norms.linf(axis=(1, 2, 3)).numpy()

    eps = Decimal(eps).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)

    # assert that l2/linf norm is less than scaled_l2_eps
    for n in norms:
        n = Decimal(n.item()).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
        if eps.compare(n) < 0:  # -1 if eps<n
            raise AssertionError


## assert that the adversarial image is lp bound wrt orig_img
# https://github.com/bethgelab/foolbox/issues/388
def assert_lp_bound(advx_batches, dict_x_to_label, eps=0.4, norms='l2'):
    # scaled_l2_eps = np.sqrt(eps*eps*224*224*3)
    # print(scaled_l2_eps)

    for index in range(len(advx_batches)):
        # assert that img_names match
        assert (dict_x_to_label[index]['img_names'] == advx_batches[index]['img_names'])

        # get X and advx
        X = dict_x_to_label[index]['X_raw']
        if isinstance(advx_batches[index]['advx'], list):
            for i in range(len(advx_batches[index]['advx'])):
                advx = advx_batches[index]['advx'][i]
                helper_lp_bound(advx, X, norms, eps)
        else:
            advx = advx_batches[index]['advx']
            helper_lp_bound(advx, X, norms, eps)


## given the dataloader and the adversarial batches, map the original image to the corresponding advx counterpart
def get_x_to_label_mapping(advx_batches, dataloader_dict):
    dict_x_to_label_mapping = {}
    for batch_index in advx_batches:
        if batch_index not in dict_x_to_label_mapping:
            dict_x_to_label_mapping[batch_index] = {'X': None,
                                                    'X_raw': None,
                                                    'img_names': [],
                                                    'advx': None}
        batch = advx_batches[batch_index]
        dict_x_to_label_mapping[batch_index]['img_names'] = batch['img_names']
        dict_x_to_label_mapping[batch_index]['advx'] = batch['advx']

        list_x, list_x_raw = [], []
        for img_index, img_name in enumerate(batch['img_names']):
            list_x.append(dataloader_dict[img_name]['X'])
            list_x_raw.append(dataloader_dict[img_name]['X_raw'])
        dict_x_to_label_mapping[batch_index]['X'] = torch.stack(list_x)
        dict_x_to_label_mapping[batch_index]['X_raw'] = torch.stack(list_x_raw)
    return dict_x_to_label_mapping


## if running without any iterations, get the adversarial accuracy
def get_advx_acc(advx_batch_list, advx_dict, model, concept='age', img_key='advx', iterations=None):
    acc_after = 0

    for batch_index in advx_batch_list:
        #print(batch_index)
        img_names = advx_batch_list[batch_index]['img_names']
        #print(advx_batch_list[batch_index]['img_names'])
        labels = []

        for _, img in enumerate(img_names):
            labels.append(advx_dict[img][concept])
        #print('\nadv label: ', labels)
        labels = torch.tensor(labels).cpu()
        if iterations is not None:
            advs_ = advx_batch_list[batch_index][img_key][iterations]
        else:
            advs_ = advx_batch_list[batch_index][img_key]

        #yp_after = model(ep.astensor(advs_.cuda())).raw.cpu()
        yp_after = model(advs_.cuda()).cpu()
        #print('adv pred: ', yp_after.max(dim=1)[1])
        acc_after += (yp_after.max(dim=1)[1] == labels).sum().item()

    return acc_after
