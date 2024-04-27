# Settings: File -> Settings -> Project Interpreter option (select python 3.7 from custom foolbox)
# Edit configuration: custom_foolbox_config same as above: /foolbox/venv/bin/python
# in project structures, project dependencies, I unmarked foolbox library as a dependency
import time
import warnings
import pandas as pd
from fb_utils import *
from fb_pgd import custom_fb_advx

from foolbox import Misclassification
import foolbox.attacks as fa

import sys
sys.path.insert(0, '../')


def orig_foolbox_attack(fmodel, images, labels, eps_val=0.4, num_steps=50, alpha=0.025, lnorm='linf'):
    if lnorm == 'linf':
        attack = foolbox.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None,
                                                                    steps=num_steps)
    elif lnorm == 'l2':
        attack = foolbox.attacks.L2ProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None,
                                                                  steps=num_steps)
    elif lnorm == 'l1':
        attack = foolbox.attacks.L1ProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None,
                                                                  steps=num_steps)
    else:
        raise Exception("Enter a valid lnorm value: linf, l1 or l2")

    # _, xp_, _ = attack(fmodel, images, labels, epsilons=eps_val)
    # xp_ = attack(fmodel, inputs=images, criterion=labels, epsilons=eps_val)
    xp_ = attack(model1=fmodel, inputs=images, epsilons=eps_val, criterion=Misclassification(labels1=labels))

    # """
    if isinstance(xp_, list):
        for index, elem in enumerate(xp_):
            xp_[index] = elem.raw.to('cpu')

        torch.cuda.empty_cache()
        return xp_

    xp = xp_.raw.to('cpu')
    del xp_
    torch.cuda.empty_cache()

    return xp


def orig_fb_advx(data_loader, fmodel, eps_val, iters, alpha=0.01 / 0.3, device_name='cuda:1', concept_index=0,
                 attack_type='linf', logging=False):
    advx_batches_result = {}

    # enumerate dataloader
    for batch_id, dl_batch in enumerate(data_loader):
        before_time = time.time()

        img_batch, img_names, labels = dl_batch['image'], dl_batch['img_name'], dl_batch['labels'][concept_index]
        labels = labels.type(torch.LongTensor)
        img_batch, labels = img_batch.to(device_name), labels.to(device_name)
        imgs, img_labels = ep.astensors(img_batch, labels)
        del img_batch, labels
        torch.cuda.empty_cache()

        if logging:
            batch_dict = {'X': imgs, 'img_names': img_names, 'labels': img_labels}
        else:
            batch_dict = {'img_names': img_names}

        if attack_type == 'linf':
            advx_imgs = orig_foolbox_attack(fmodel=fmodel, images=imgs, labels=img_labels, eps_val=eps_val,
                                            num_steps=iters, alpha=alpha, lnorm='linf')

        elif attack_type == 'l1':
            advx_imgs = orig_foolbox_attack(fmodel=fmodel, images=imgs, labels=img_labels, eps_val=eps_val,
                                            num_steps=iters, alpha=alpha, lnorm='l1')

        elif attack_type == 'l2':
            advx_imgs = orig_foolbox_attack(fmodel=fmodel, images=imgs, labels=img_labels, eps_val=eps_val,
                                            num_steps=iters, alpha=alpha, lnorm='l2')

        else:
            raise NotImplementedError

        batch_dict['advx'] = advx_imgs
        advx_batches_result[batch_id] = batch_dict

        del advx_imgs, img_names, img_labels, batch_dict
        after_time = time.time()
        print(round((after_time - before_time), 3), ' sec for batch:', batch_id)
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return advx_batches_result


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    torch.cuda.set_device(2)
    device, kwargs = set_device(device_name="cuda:2")
    print(device, kwargs)
    torch.cuda.empty_cache()

    # Fill in the variables
    batch_size_val = 20

    df_advx = pd.read_csv('df_utkface_1000.csv')
    df_advx_1000_dict = get_pdadvx_dict_utkface(df_advx)

    # get the data loader and data_loader_dictionary
    dataloader = get_dataloader(kwargs_func=kwargs, batch_size=batch_size_val, pd_advx=df_advx, root_dir='',
                                img_pkl_file=None, is_for_train=False)

    dataloader_dict_1000 = get_dataloader_dict(data_loader=dataloader)

    concept_dict = {'age': 0, 'gender': 1, 'ethnicity': 2}

    # CelebA dataset
    # in this case, male = 1, female = 0

    # UTKFace dataset
    model_dir = 'models_utkface/'
    _, fmodel_gender = get_foolbox_model(weights_file=model_dir+'stage-2-rn50-gender-utk.pth', dev=device, num_classes=2,
                                         is_torch=False, preprocess=True, models_dir='models')

    _, fmodel_age = get_foolbox_model(weights_file=model_dir+'stage-2-rn50-age-utk.pth', dev=device, num_classes=2,
                                      is_torch=False, preprocess=True, models_dir='models')

    _, fmodel_ethnicity = get_foolbox_model(weights_file=model_dir+'stage-2-rn50-ethnicity-utk.pth', dev=device, num_classes=2,
                                            is_torch=False, preprocess=True, models_dir='models')
    """
    _, fmodel_age_multi = get_foolbox_model(weights_file='stage-2-rn50-age-multi-utk-final.pth',
                                            dev=device,
                                            num_classes=3, is_torch=False, preprocess=True,
                                            models_dir='models')
    """
    advx_batches = custom_fb_advx(data_loader=dataloader_multi, fmodel2=fmodel_age_multi, fmodel1=fmodel_gender,
                                  eps_val=4, alpha=0.8, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['gender'], attack_type='l2',
                                  concept_preserved=concept_dict['age'], weight1=1, weight2=1,
                                  fmodel3=fmodel_ethnicity, concept_third=concept_dict['ethnicity'], att_def_avg_third='att',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False,
                                  is_multi_concept=True, multi_concept_maxlabel_dict=multi_concept_label_dict)

    pickle.dump(advx_batches, open('pgd_multi_maxloss_l2_genderEthnicity_age_eps_4_utk.pkl', 'wb'))

    advx_batches = custom_fb_advx(data_loader=dataloader_multi, fmodel2=fmodel_age_multi, fmodel1=fmodel_gender,
                                  eps_val=4, alpha=0.8, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['gender'], attack_type='l2',
                                  concept_preserved=concept_dict['age'], weight1=1, weight2=1,
                                  fmodel3=fmodel_ethnicity, concept_third=concept_dict['ethnicity'],
                                  att_def_avg_third='def',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False,
                                  is_multi_concept=True, multi_concept_maxlabel_dict=multi_concept_label_dict)

    pickle.dump(advx_batches, open('pgd_multi_maxloss_l2_gender_ageEthnicity_eps_4_utk.pkl', 'wb'))

    advx_batches = custom_fb_advx(data_loader=dataloader_multi, fmodel2=fmodel_age_multi, fmodel1=fmodel_gender,
                                  eps_val=0.3, alpha=0.06, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['gender'], attack_type='linf',
                                  concept_preserved=concept_dict['age'], weight1=1, weight2=1,
                                  fmodel3=fmodel_ethnicity, concept_third=concept_dict['ethnicity'],
                                  att_def_avg_third='def',
                                  fmodel4=None, concept_fourth=None,
                                  att_def_avg_fourth='att', do_label_flip=False,
                                  is_multi_concept=True, multi_concept_maxlabel_dict=multi_concept_label_dict)

    pickle.dump(advx_batches, open('pgd_multi_maxloss_linf_gender_ageEthnicity_eps_0.3_utk.pkl', 'wb'))
