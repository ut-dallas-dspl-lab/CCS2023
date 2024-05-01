# Settings: File -> Settings -> Project Interpreter option (select python 3.7 from custom foolbox)
# Edit configuration: custom_foolbox_config same as above: /foolbox/venv/bin/python

import pickle
import time
import warnings
import pandas as pd
import torch

from fb_utils import *

import eagerpy as ep
import sys
sys.path.insert(0, '../../lib')
import foolbox

from foolbox import Misclassification
import foolbox.attacks as fa


def custom_foolbox_attack(fmodel1, fmodel2, images, labels1, labels2, eps_val=0.4, num_steps=50, alpha=0.025,
                          lnorm='linf', att_def_avg_third=None, att_def_avg_fourth=None,
                          weight1=1.0, weight2=1.0, fmodel3=None, labels3=None, fmodel4=None, labels4=None):
    if lnorm == 'linf':
        attack = fa.LinfProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None, steps=num_steps,
                                                       weight1=weight1, weight2=weight2,
                                                       att_def_avg_third=att_def_avg_third,
                                                       att_def_avg_fourth=att_def_avg_fourth)
    elif lnorm == 'l2':
        attack = fa.L2ProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None, steps=num_steps,
                                                     weight1=weight1, weight2=weight2,
                                                     att_def_avg_third=att_def_avg_third,
                                                     att_def_avg_fourth=att_def_avg_fourth)
    elif lnorm == 'l1':
        attack = fa.L1ProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None, steps=num_steps,
                                                     weight1=weight1, weight2=weight2,
                                                     att_def_avg_third=att_def_avg_third,
                                                     att_def_avg_fourth=att_def_avg_fourth)
    else:
        raise Exception("Enter a valid lnorm value: linf, l1 or l2")

    _, xp_, _ = attack(model1=fmodel1, model2=fmodel2, inputs=images, epsilons=eps_val, model3=fmodel3, model4=fmodel4,
                       criterion=Misclassification(labels1=labels1, labels2=labels2, labels3=labels3, labels4=labels4))

    xp = xp_.raw.to('cpu')
    del xp_
    torch.cuda.empty_cache()

    return xp

"""
# NOTE:
- max_opp_dir = None: when we simply add the losses of our models
"""
def custom_fb_advx(data_loader, fmodel1, fmodel2, eps_val=0.4, alpha=0.01 / 0.3, iters=200, device_name='cuda:1',
                   concept_attacked=0, concept_preserved=0, attack_type='linf', weight1=1.0, weight2=1.0,
                   fmodel3=None, concept_third=None, att_def_avg_third=None, fmodel4=None, concept_fourth=None,
                   att_def_avg_fourth=None, do_label_flip=True, is_multi_concept=False, multi_concept_maxlabel_dict=None,
                   multi_concept_attacked=False, multi_concept_pgd=False, multi_concept_truth_dict=None):
    advx_batches_result = {}

    # enumerate dataloader
    for batch_id, dl_batch in enumerate(data_loader):
        before_time = time.time()

        print(type(dl_batch))
        img_batch, img_names = dl_batch['image'], dl_batch['img_name']
        img_batch = img_batch.to(device_name)

        if not multi_concept_attacked:
            # for regular attacked scenario, where model1 is binary-concept model and is in the 'attack' set
            labels1 = dl_batch['labels'][concept_attacked]
        else:
            if not multi_concept_pgd:
                # for binary-concept attacked scenario where model1 is binary-concept model and is in the 'protect' set
                labels1 = 1 - dl_batch['labels'][concept_attacked]
            else:
                # for multi-concept attacked scenario where model1 is multi-concept and is in the 'attack' set
                # here multi_concept_truth_dict contains labels with maximum loss difference wrt ground truth label
                labels1 = torch.LongTensor(multi_concept_truth_dict[batch_id])

        if do_label_flip:
            # when model2 is in 'protect' set
            # applies to both binary-concept and multi-concept scenario
            labels2 = 1 - dl_batch['labels'][concept_preserved]

        elif not do_label_flip and not is_multi_concept:
            # for running 1) regular pgd an 2) simple averaging out losses of all models scenario
            labels2 = dl_batch['labels'][concept_preserved]

        else:
            if not multi_concept_attacked:
                # for multi-concept attacked scenario where model2 is multi-concept and is in the 'protect' set
                # here multi_concept_maxlabel_dict contains labels with maximum loss difference wrt ground truth label
                labels2 = torch.LongTensor(multi_concept_maxlabel_dict[batch_id])
            else:
                # here multi_concept_truth_dict contains labels with maximum loss difference wrt ground truth label
                labels2 = torch.LongTensor(multi_concept_truth_dict[batch_id])

        labels1, labels2 = labels1.type(torch.LongTensor), labels2.type(torch.LongTensor)
        labels1, labels2 = labels1.to(device_name), labels2.to(device_name)
        imgs, img_labels1, img_labels2 = ep.astensors(img_batch, labels1, labels2)
        del img_batch, labels1, labels2

        img_labels3 = None
        if concept_third is not None:
            if att_def_avg_third == 'att':
                labels3 = dl_batch['labels'][concept_third]
            elif att_def_avg_third == 'def':
                if do_label_flip or is_multi_concept:
                    labels3 = 1 - dl_batch['labels'][concept_third]
                else:
                    labels3 = dl_batch['labels'][concept_third]
            else:
                raise NotImplementedError
            labels3 = labels3.type(torch.LongTensor)
            labels3 = labels3.to(device_name)
            img_labels3 = ep.astensor(labels3)
            del labels3

        img_labels4 = None
        if concept_fourth is not None:
            if att_def_avg_fourth == 'att':
                labels4 = dl_batch['labels'][concept_fourth]
            elif att_def_avg_fourth == 'def' or is_multi_concept:
                if do_label_flip:
                    labels4 = 1 - dl_batch['labels'][concept_fourth]
                else:
                    labels4 = dl_batch['labels'][concept_fourth]
            else:
                raise NotImplementedError
            labels4 = labels4.type(torch.LongTensor)
            labels4 = labels4.to(device_name)
            img_labels4 = ep.astensor(labels4)
            del labels4

        torch.cuda.empty_cache()

        batch_dict = {'img_names': img_names}

        if attack_type == 'linf':
            advx_imgs = custom_foolbox_attack(fmodel1=fmodel1, fmodel2=fmodel2, images=imgs,
                                              labels1=img_labels1, labels2=img_labels2,
                                              eps_val=eps_val, num_steps=iters,
                                              alpha=alpha, lnorm='linf',
                                              weight1=weight1, weight2=weight2,
                                              fmodel3=fmodel3, labels3=img_labels3,
                                              att_def_avg_third=att_def_avg_third,
                                              fmodel4=fmodel4, labels4=img_labels4,
                                              att_def_avg_fourth=att_def_avg_fourth)

        elif attack_type == 'l1':
            advx_imgs = custom_foolbox_attack(fmodel1=fmodel1, fmodel2=fmodel2, images=imgs,
                                              labels1=img_labels1, labels2=img_labels2,
                                              eps_val=eps_val, num_steps=iters,
                                              alpha=alpha, lnorm='l1',
                                              weight1=weight1, weight2=weight2,
                                              fmodel3=fmodel3, labels3=img_labels3,
                                              att_def_avg_third=att_def_avg_third,
                                              fmodel4=fmodel4, labels4=img_labels4,
                                              att_def_avg_fourth=att_def_avg_fourth)

        elif attack_type == 'l2':
            advx_imgs = custom_foolbox_attack(fmodel1=fmodel1, fmodel2=fmodel2, images=imgs,
                                              labels1=img_labels1, labels2=img_labels2,
                                              eps_val=eps_val, num_steps=iters,
                                              alpha=alpha, lnorm='l2',
                                              weight1=weight1, weight2=weight2,
                                              fmodel3=fmodel3, labels3=img_labels3,
                                              att_def_avg_third=att_def_avg_third,
                                              fmodel4=fmodel4, labels4=img_labels4,
                                              att_def_avg_fourth=att_def_avg_fourth)

        else:
            raise NotImplementedError

        batch_dict['advx'] = advx_imgs
        advx_batches_result[batch_id] = batch_dict

        del advx_imgs, img_names, img_labels1, img_labels2, img_labels3, img_labels4, batch_dict, imgs
        after_time = time.time()
        print(round((after_time - before_time), 3), ' sec for batch:', batch_id)
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return advx_batches_result


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    torch.cuda.set_device(0)
    device, kwargs = set_device(device_name="cuda:0")
    print(device, kwargs)
    torch.cuda.empty_cache()

    # Fill in the variables
    batch_size_val = 20

    #"""
    # CelebA
    #pd_advx = pickle.load(open('pickle_files/pd_advx_age_pretty_1000.pkl', 'rb'))
    pd_advx = pickle.load(open('pickle_files/df_glasses_gender_CelebA.pkl', 'rb'))
    #pd_advx_dict = get_pdadvx_dict(pd_advx)
    
    dataloader = get_dataloader(kwargs_func=kwargs, root_dir='data/celeba/img_align_celeba', batch_size=batch_size_val, pd_advx=pd_advx)
    
    #dataloader_dict_1000 = get_dataloader_dict(data_loader=dataloader)
    
    concept_dict = {'pretty': 0, 'glasses': 1, 'male': 2, 'age': 3}
    
    _, fmodel_glasses = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-eyeglasses.pth', dev=device, num_classes=2,
                                              is_torch=True)

    _, fmodel_male = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-gender.pth', dev=device, num_classes=2, is_torch=True)

    _, fmodel_age = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-young.pth', dev=device, num_classes=2, is_torch=True)

    _, fmodel_pretty = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-attr.pth', dev=device, num_classes=2, is_torch=True)
    #_, fmodel_glasses = get_foolbox_model(weights_file='ad_celeba_models/model_best_eyeglasses.pth.tar', dev=device, num_classes=2,
    #                                          is_torch=True)

    #_, fmodel_male = get_foolbox_model(weights_file='ad_celeba_models/model_best_gender.pth.tar', dev=device, num_classes=2, is_torch=True)
    #_, fmodel_age = get_foolbox_model(weights_file='ad_celeba_models/model_best_young.pth.tar', dev=device, num_classes=2, is_torch=True)

    #_, fmodel_pretty = get_foolbox_model(weights_file='ad_celeba_models/model_best_attractive.pth.tar', dev=device, num_classes=2, is_torch=True)
    #"""

    # UTKFace
    """
    df_advx = pd.read_csv('df_utkface_1000.csv')
    df_advx_1000_dict = get_pdadvx_dict_utkface(df_advx)
    
    # get the conceptDict lst and model_list
    concept_dict = {'age': 0, 'gender': 1, 'ethnicity': 2}
    
    # get the data loader and data_loader_dictionary
    dataloader = get_dataloader(kwargs_func=kwargs, batch_size=batch_size_val, pd_advx=df_advx, root_dir='',
                                img_pkl_file=None, is_for_train=False)
    dataloader_dict_1000 = get_dataloader_dict(data_loader=dataloader)
    """

    #df_advx_multi = pd.read_csv('df_utkface_1000_multi.csv')
    #df_advx_1000_dict_multi = get_pdadvx_dict_utkface(df_advx_multi)
    #dataloader_multi = get_dataloader(kwargs_func=kwargs, batch_size=20, pd_advx=df_advx_multi, root_dir='',
    #                                  img_pkl_file=None, is_for_train=False)
    #multi_concept_label_dict = pickle.load(open("maxlabel_multi_age.p", "rb"))
    #multi_concept_groundtruth_dict = pickle.load(open("groundtruthlabel_multi_age.p", "rb"))
    
    #concept_dict = {'age': 0, 'gender': 1, 'ethnicity': 2}

    """
    #UTKFace
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_age, eps_val=0.3,
                                  alpha=0.06, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['gender'], attack_type='linf',
                                  concept_preserved=concept_dict['age'], weight1=1, weight2=1,
                                  fmodel3=None, concept_third=None, att_def_avg_third='att',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False)

    pickle.dump(advx_batches,
                open('pgd_avgloss_linf_genderAge_fb_label_flip_sum_None_abs_eps_0.3_sum_utk.pkl',
                     'wb'))

    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_age, eps_val=0.3,
                                  alpha=0.06, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['gender'], attack_type='linf',
                                  concept_preserved=concept_dict['age'], weight1=1, weight2=1,
                                  fmodel3=fmodel_ethnicity, concept_third=concept_dict['ethnicity'],
                                  att_def_avg_third='att',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False)

    pickle.dump(advx_batches,
                open('pgd_avgloss_linf_genderAgeEthnicity_fb_label_flip_sum_None_abs_eps_0.3_sum_utk.pkl',
                     'wb'))
    """

    #"""
    # CelebA
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_male, fmodel2=fmodel_glasses, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  #alpha=40 * 0.3 / 200, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['male'], attack_type='linf',
                                  concept_preserved=concept_dict['glasses'], weight1=1, weight2=1,
                                  fmodel3=None, concept_third=None, att_def_avg_third='att',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False)

    pickle.dump(advx_batches,
                open('pgd_avgloss_linf_maleGlasses_fb_label_flip_sum_None_abs_eps_0.3_sum.pkl',
                     'wb'))

    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_male, fmodel2=fmodel_glasses, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  #alpha=40 * 0.3 / 200, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['male'], attack_type='linf',
                                  concept_preserved=concept_dict['glasses'], weight1=1, weight2=1,
                                  fmodel3=fmodel_age, concept_third=concept_dict['age'], att_def_avg_third='att',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False)

    pickle.dump(advx_batches,
                open('pgd_avgloss_linf_maleGlassesAge_fb_label_flip_sum_None_abs_eps_0.3_sum.pkl',
                     'wb'))

    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_male, fmodel2=fmodel_glasses, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  #alpha=40 * 0.3 / 200, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['male'], attack_type='linf',
                                  concept_preserved=concept_dict['glasses'], weight1=1, weight2=1,
                                  fmodel3=fmodel_age, concept_third=concept_dict['age'], att_def_avg_third='att',
                                  fmodel4=fmodel_pretty, concept_fourth=concept_dict['pretty'],
                                  att_def_avg_fourth='att', do_label_flip=False)

    pickle.dump(advx_batches,
                open('pgd_avgloss_linf_maleGlassesAgePretty_fb_label_flip_sum_None_abs_eps_0.3_sum.pkl',
                     'wb'))
    #"""



