import json
import os
print(os.getcwd())
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from PIL import Image
import pickle
import pandas as pd
import argparse

from numpy.linalg import norm
from decimal import Decimal, ROUND_HALF_EVEN

from fb_utils import *

def get_gen_acc(dl, model, concept_index, foolbox_model=False):
    acc_after = 0

    #default_batch_sampler = dl.batch_sampler
    for batch_index, batch in enumerate(dl):
        concept_label = batch['labels'][concept_index]
        #print(concept_label)
        if not foolbox_model:
            imgs = batch['image_norm']
            yp_after = model(imgs.cuda()).cpu()
        else:
            imgs = batch['image']
            yp_after = model(ep.astensor(imgs.cuda())).raw.cpu()
        acc_after += (yp_after.max(dim=1)[1] == concept_label).sum().item()

    return acc_after

parser = argparse.ArgumentParser(description='Adversarial Training Flag')
parser.add_argument('--traintype', default='std', help='Specify model to attack: std, ad_train')
parser.add_argument('--modeltype', default ='res', help='model type: res, mobile')
args = parser.parse_args()

torch.cuda.set_device(0)
device, kwargs = set_device(device_name="cuda:0")
print(device, kwargs)
torch.cuda.empty_cache()

df_advx = pickle.load(open('pickle_files/df_glasses_gender_CelebA.pkl', 'rb'))
df_advx_1000_dict = get_pdadvx_dict(df_advx)

dataloader = get_dataloader(kwargs_func=kwargs, root_dir='data/celeba/img_align_celeba', batch_size=20, pd_advx=df_advx)
dataloader_dict_1000 = get_dataloader_dict(data_loader=dataloader)

# get the conceptDict lst and model_list
concept_dict = {'pretty': 0, 'glasses': 1, 'male': 2, 'age': 3}

#standard training
if args.traintype == 'std':
    # resnet50
    if args.modeltype =='res':
        model_glasses, fmodel_glasses = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-eyeglasses.pth', dev=device, num_classes=2,
                                                   is_torch=True)
        model_gender, fmodel_male = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-gender.pth', dev=device, num_classes=2, is_torch=True)
        model_age, fmodel_age = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-young.pth', dev=device, num_classes=2, is_torch=True)
        model_pretty, fmodel_pretty = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-attr.pth', dev=device, num_classes=2, is_torch=True)
        data_dir = 'final_exp_celeba/'
    # mobile net
    else:
        model_glasses, fmodel_glasses = get_foolbox_model(modeltype='mobilenet',weights_file='Mobilenet_models/celeba_models/model_best_eyeglasses.pth.tar', dev=device, num_classes=2,
                                                    is_torch=True)
        model_gender, fmodel_male = get_foolbox_model(modeltype='mobilenet',weights_file='Mobilenet_models/celeba_models/model_best_gender.pth.tar', dev=device, num_classes=2, is_torch=True)
        model_age, fmodel_age = get_foolbox_model(modeltype='mobilenet',weights_file='Mobilenet_models/celeba_models/model_best_young.pth.tar', dev=device, num_classes=2, is_torch=True)
        model_pretty, fmodel_pretty = get_foolbox_model(modeltype='mobilenet',weights_file='Mobilenet_models/celeba_models/model_best_attractive.pth.tar', dev=device, num_classes=2, is_torch=True)
        data_dir = 'Mobilenet_models/Mobile_net_results/'
#adversarial training
else:
    model_glasses, fmodel_glasses = get_foolbox_model(modeltype='mobilenet', weights_file='Mobilenet_models/ad_celeba_models/model_best_eyeglasses.pth.tar', dev=device, num_classes=2,
                                               is_torch=True)
    model_gender, fmodel_male = get_foolbox_model(modeltype='mobilenet', weights_file='Mobilenet_models/ad_celeba_models/model_best_gender.pth.tar', dev=device, num_classes=2, is_torch=True)
    model_age, fmodel_age = get_foolbox_model(modeltype='mobilenet', weights_file='Mobilenet_models/ad_celeba_models/model_best_young.pth.tar', dev=device, num_classes=2, is_torch=True)
    model_pretty, fmodel_pretty = get_foolbox_model(modeltype='mobilenet', weights_file='Mobilenet_models/ad_celeba_models/model_best_attractive.pth.tar', dev=device, num_classes=2, is_torch=True)
    data_dir = 'Mobilenet_models/Mobile_net_results/adTrain_'


#attack one, defend None
f_glass_10 = data_dir+'pgd_custom_linf_glasses_eps_0.3_step_40_iter_200.pkl'
f_gender_10 = data_dir+'pgd_custom_linf_gender_eps_0.3_step_40_iter_200.pkl'
f_pretty_10 = data_dir+'pgd_custom_linf_pretty_eps_0.3_step_40_iter_200.pkl'
f_age_10 = data_dir+'pgd_custom_linf_age_eps_0.3_step_40_iter_200.pkl'

#attack one, defend one
f_glass_11 = data_dir+'pgd_custom_linf_glasses_pretty_eps_0.3_step_40_iter_200.pkl'
f_gender_11 = data_dir+'pgd_custom_linf_gender_pretty_eps_0.3_step_40_iter_200.pkl'
f_pretty_11 = data_dir+'pgd_custom_linf_pretty_glasses_eps_0.3_step_40_iter_200.pkl'
f_age_11 = data_dir+'pgd_custom_linf_age_gender_eps_0.3_step_40_iter_200.pkl'

#attack one, defend two
f_glass_12 = data_dir+'pgd_custom_linf_glasses_prettyGender_eps_0.3_step_40_iter_200.pkl'
f_gender_12 = data_dir+'pgd_custom_linf_gender_prettyGlasses_eps_0.3_step_40_iter_200.pkl'
f_pretty_12 = data_dir+'pgd_custom_linf_pretty_genderGlasses_eps_0.3_step_40_iter_200.pkl'
f_age_12 = data_dir+'pgd_custom_linf_age_genderGlasses_eps_0.3_step_40_iter_200.pkl'

#attack one, defend three
f_glass_13 = data_dir+'pgd_custom_linf_glasses_prettyGenderAge_eps_0.3_step_40_iter_200.pkl'
f_gender_13 = data_dir+'pgd_custom_linf_gender_prettyGlassesAge_eps_0.3_step_40_iter_200.pkl'
f_pretty_13 = data_dir+'pgd_custom_linf_pretty_genderGlassesAge_eps_0.3_step_40_iter_200.pkl'
f_age_13 = data_dir+'pgd_custom_linf_age_genderGlassesPretty_eps_0.3_step_40_iter_200.pkl'

all_fs = [f_glass_10, f_gender_10, f_pretty_10, f_age_10, f_glass_11, f_gender_11, f_pretty_11, f_age_11, f_glass_12, f_gender_12, f_pretty_12, f_age_12, f_glass_13, f_gender_13, f_pretty_13, f_age_13]

acc_before = get_gen_acc(dataloader, model_pretty, concept_index=concept_dict['pretty'])/len(df_advx)*100
print(f"Accuracy of pretty before attack: {np.round(acc_before, 2)}%")
acc_before = get_gen_acc(dataloader, model_gender, concept_index=concept_dict['male'])/len(df_advx)*100
print(f"Accuracy of gender before attack: {np.round(acc_before, 2)}%")
acc_before = get_gen_acc(dataloader, model_glasses, concept_index=concept_dict['glasses'])/len(df_advx)*100
print(f"Accuracy of glasses before attack: {np.round(acc_before, 2)}%")
acc_before = get_gen_acc(dataloader, model_age, concept_index=concept_dict['age'])/len(df_advx)*100
print(f"Accuracy of age before attack: {np.round(acc_before, 2)}%")

for f in all_fs:
    advx_batches = pickle.load(open(f, 'rb'))
    acc_gender = np.round(get_advx_acc(advx_batch_list=advx_batches, advx_dict=df_advx_1000_dict, model=model_gender, concept='gender', img_key='advx', iterations=None)/len(df_advx)*100, 2)
    acc_pretty = np.round(get_advx_acc(advx_batch_list=advx_batches, advx_dict=df_advx_1000_dict, model=model_pretty, concept='pretty', img_key='advx', iterations=None)/len(df_advx)*100, 2)
    acc_glasses = np.round(get_advx_acc(advx_batch_list=advx_batches, advx_dict=df_advx_1000_dict, model=model_glasses, concept='glasses', img_key='advx', iterations=None)/len(df_advx)*100, 2)
    acc_age = np.round(get_advx_acc(advx_batch_list=advx_batches, advx_dict=df_advx_1000_dict, model=model_age, concept='age', img_key='advx', iterations=None)/len(df_advx)*100, 2)

    print(f)
    print(f"Acc of pretty concept: {acc_pretty}")
    print(f"Acc of gender concept: {acc_gender}")
    print(f"Acc of glasses concept: {acc_glasses}")
    print(f"Acc of age concept: {acc_age}")

    del acc_pretty, acc_glasses, acc_gender, advx_batches 
    torch.cuda.empty_cache()

