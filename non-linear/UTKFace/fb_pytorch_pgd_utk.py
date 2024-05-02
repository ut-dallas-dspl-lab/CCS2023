# Settings: File -> Settings -> Project Interpreter option (select python 3.7 from custom foolbox)
# Edit configuration: custom_foolbox_config same as above: /foolbox/venv/bin/python
# in project structures, project dependencies, I unmarked foolbox library as a dependency

import pickle
import time
import warnings
import pandas as pd
import torch

from fb_utils import *
from pathlib import Path

import eagerpy as ep
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

    #create output folder
    output_dir = './final_exp_utkface'
    Path(output_dir).mkdir(exist_ok=True)

    batch_size_val = 25

    ##### UTKFace dataset
    
    # get the data loader and data_loader_dictionary for UTKFace dataset
    df_advx = pd.read_csv('df_utkface_1000.csv')
    dataloader = get_dataloader(kwargs_func=kwargs, root_dir=os.getcwd(), batch_size=batch_size_val, pd_advx=df_advx)
    dataloader_dict_1000 = get_dataloader_dict(data_loader=dataloader)

    # get the UTKFace models
    _, fmodel_gender = get_foolbox_model(weights_file='stage-2-rn50-gender-utk.pth', dev=device, num_classes=2, is_torch=False, preprocess=True, models_dir='models_utkface')

    _, fmodel_age = get_foolbox_model(weights_file='stage-2-rn50-age-utk.pth', dev=device, num_classes=2,
                                      is_torch=False, preprocess=True, models_dir='models_utkface')

    _, fmodel_ethnicity = get_foolbox_model(weights_file='stage-2-rn50-ethnicity-utk.pth', dev=device, num_classes=2, is_torch=False, preprocess=True, models_dir='models_utkface')
    
    # get the concept dictionary list
    concept_dict = {'age': 0, 'gender': 1, 'ethnicity': 2}

    # attack "age", defend "gender"
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_age,
                                  fmodel2=fmodel_gender, eps_val=4, alpha=40 * 4 / 200,
                                  iters=201, device_name=device,
                                  concept_attacked=concept_dict['age'],
                                  attack_type='l2', #att_def_avg='att',
                                  concept_preserved=concept_dict['gender'], weight1=1,
                                  weight2=1, #att_or_def_third='att', 
                                  fmodel3=None,
                                  concept_third=None)

    pickle.dump(advx_batches,
    open('final_exp_utkface/pgd_custom_l2_age_gender_eps_4_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()

    # attack "age", defend "gender" and "ethenicity"
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_age,
                                  fmodel2=fmodel_gender, eps_val=4, alpha=40 * 4 / 200,
                                  iters=201, device_name=device,
                                  concept_attacked=concept_dict['age'],
                                  attack_type='l2', #att_def_avg='att',
                                  concept_preserved=concept_dict['gender'], weight1=1,
                                  weight2=1,  att_def_avg_third='def', 
                                  fmodel3=fmodel_ethnicity,
                                  concept_third=concept_dict['ethnicity'])

    pickle.dump(advx_batches,
    open('final_exp_utkface/pgd_custom_l2_age_genderEthnicity_eps_4_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()

