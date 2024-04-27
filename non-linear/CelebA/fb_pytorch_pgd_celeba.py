# Settings: File -> Settings -> Project Interpreter option (select python 3.7 from anaconda3/bin/python)
# Edit configuration: custom_foolbox_config same as above: python3.7 of anaconda3/bin/python

from fb_utils import *
from fb_pgd_UTKFace import *
from fb_pgd import custom_fb_advx


def orig_foolbox_attack(fmodel, images, labels, eps_val=0.4, num_steps=50, alpha=0.025, lnorm='linf'):
    if lnorm == 'linf':
        attack = foolbox.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None,
    elif lnorm == 'l2':
        attack = foolbox.attacks.L2ProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None,
    elif lnorm == 'l1':
        attack = foolbox.attacks.L1ProjectedGradientDescentAttack(rel_stepsize=alpha, abs_stepsize=None,
    else:
        raise Exception("Enter a valid lnorm value: linf, l1 or l2")

    xp_ = attack(fmodel, images, labels, epsilons=eps_val) 

    if isinstance(xp_, list):
        for index, elem in enumerate(xp_):
            xp_[index] = elem.raw.to('cpu')

        torch.cuda.empty_cache()
        return xp_
    else:
        xp = xp_.raw.to('cpu')
        del xp_
        return xp

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='std', help='Specify model to attack: std, ad_train')
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    torch.cuda.set_device(0)
    device, kwargs = set_device(device_name="cuda:0")
    print(device, kwargs)

    batch_size_val = 25
    df_advx = pickle.load(open('pickle_files/df_glasses_gender_CelebA.pkl', 'rb'))
    pd_advx_1000_dict = get_pdadvx_dict(df_advx)
    print(df_advx.describe())
    print(df_advx.head())

    # get the data loader and data_loader_dictionary
    dataloader = get_dataloader(kwargs_func=kwargs, root_dir='celeba-dataset/img_align_celeba', batch_size=batch_size_val, pd_advx=df_advx)

    # CelebA dataset
    # standard training
    if args.model == 'std':
        _, fmodel_glasses = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-eyeglasses.pth', dev=device, num_classes=2,
                                            is_torch=True)
        _, fmodel_gender = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-gender.pth', dev=device, num_classes=2,
                                             is_torch=True)
        _, fmodel_age = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-young.pth', dev=device, num_classes=2, is_torch=True)
        _, fmodel_pretty = get_foolbox_model(weights_file='models_celeba/pytorch-rn50-attr.pth', dev=device, num_classes=2, is_torch=True)
        f_name = 'pgd_custom_linf'
    # adversaril training
    else:
        _, fmodel_glasses = get_foolbox_model(weights_file='ad_celeba_models/model_best_eyeglasses.pth.tar', dev=device, num_classes=2,
                                                  is_torch=True)

        _, fmodel_gender = get_foolbox_model(weights_file='ad_celeba_models/model_best_gender.pth.tar', dev=device, num_classes=2, is_torch=True)
        _, fmodel_age = get_foolbox_model(weights_file='ad_celeba_models/model_best_young.pth.tar', dev=device, num_classes=2, is_torch=True)

        _, fmodel_pretty = get_foolbox_model(weights_file='ad_celeba_models/model_best_attractive.pth.tar', dev=device, num_classes=2, is_torch=True)
        f_name = 'adTrain_pgd_custom_linf'


    # get the conceptDict lst and model_list
    concept_dict = {'pretty': 0, 'glasses': 1, 'male': 2, 'age': 3}

    ######################
    # attack GENDER linf #
    ######################

    # atttack one, defend None
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['male'], attack_type='linf', 
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=0)


    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_gender_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()

    # atttack one, defend one
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['male'], attack_type='linf', 
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1)

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_gender_pretty_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()

    # atttack one, defend two
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['male'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_gender_prettyGlasses_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()

    # atttack one, defend three 
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['male'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'],
                                  att_def_avg_fourth='def', fmodel4=fmodel_age, concept_fourth=concept_dict['age'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_gender_prettyGlassesAge_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()


    """
    # attack l2
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_gender, fmodel2=fmodel_pretty, eps_val=4,
                                  alpha=40 * 4 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['male'], attack_type='l2', att_def_avg_third='def',
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'])

    pickle.dump(advx_batches, open('final_exp_celeba/pgd_custom_l2_gender_prettyGlasses_eps_4_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()
    """


    ##########################
    # attack EYEGLASSES linf #
    ##########################

    # atttack one, defend None
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_glasses, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['glasses'], attack_type='linf', 
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=0)


    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_glasses_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend one
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_glasses, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['glasses'], attack_type='linf', 
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1)

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_glasses_pretty_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend two
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_glasses, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['glasses'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1,
                                  fmodel3=fmodel_gender, concept_third=concept_dict['male'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_glasses_prettyGender_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend three
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_glasses, fmodel2=fmodel_pretty, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device, 
                                  concept_attacked=concept_dict['glasses'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1,
                                  fmodel3=fmodel_gender, concept_third=concept_dict['male'],
                                  att_def_avg_fourth='def', fmodel4=fmodel_age, concept_fourth=concept_dict['age'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_glasses_prettyGenderAge_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    """
    # attack l2
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_glasses, fmodel2=fmodel_pretty, eps_val=4,
                                  alpha=40 * 4 / 200, iters=201, device_name=device, 
                                  #alpha=40 * 4 / 200, iters=201, device_name=device, variant='v1',
                                  concept_attacked=concept_dict['glasses'], attack_type='l2', att_def_avg_third='def',
                                  concept_preserved=concept_dict['pretty'], weight1=1, weight2=1,
                                  fmodel3=fmodel_gender, concept_third=concept_dict['male'])
                                  #att_or_def_third='def', fmodel3=fmodel_gender, concept_third=concept_dict['male'])

    pickle.dump(advx_batches, open('final_exp_celeba/pgd_custom_l2_glasses_prettyGender_eps_4_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()
    """


    ##########################    
    # attack ATTRACTIVE linf #
    ##########################    

    # atttack one, defend None
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_pretty, fmodel2=fmodel_glasses, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['pretty'], attack_type='linf',
                                  concept_preserved=concept_dict['glasses'], weight1=1, weight2=0)


    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_pretty_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()


    #attack one, defend one
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_pretty, fmodel2=fmodel_glasses, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['pretty'], attack_type='linf',
                                  concept_preserved=concept_dict['glasses'], weight1=1, weight2=1)

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_pretty_glasses_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend two
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_pretty, fmodel2=fmodel_gender, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['pretty'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['male'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_pretty_genderGlasses_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend three
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_pretty, fmodel2=fmodel_gender, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['pretty'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['male'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'],#)
                                  att_def_avg_fourth='def', fmodel4=fmodel_age, concept_fourth=concept_dict['age'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_pretty_genderGlassesAge_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()



    ###################    
    # attack Age linf #
    ###################    

    # atttack one, defend None
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_age, fmodel2=fmodel_gender, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['age'], attack_type='linf',
                                  concept_preserved=concept_dict['male'], weight1=1, weight2=0)


    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_age_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))

    del advx_batches
    torch.cuda.empty_cache()


    #attack one, defend one
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_age, fmodel2=fmodel_gender, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['age'], attack_type='linf',
                                  concept_preserved=concept_dict['male'], weight1=1, weight2=1)

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_age_gender_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend two
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_age, fmodel2=fmodel_gender, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['age'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['male'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_age_genderGlasses_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

    #attack one, defend three
    advx_batches = custom_fb_advx(data_loader=dataloader, fmodel1=fmodel_age, fmodel2=fmodel_gender, eps_val=0.3,
                                  alpha=40 * 0.3 / 200, iters=201, device_name=device,
                                  concept_attacked=concept_dict['age'], attack_type='linf', att_def_avg_third='def',
                                  concept_preserved=concept_dict['male'], weight1=1, weight2=1,
                                  fmodel3=fmodel_glasses, concept_third=concept_dict['glasses'],#)
                                  att_def_avg_fourth='def', fmodel4=fmodel_pretty, concept_fourth=concept_dict['pretty'])

    pickle.dump(advx_batches, open('final_exp_celeba/'+f_name+'_age_genderGlassesPretty_eps_0.3_step_40_iter_200.pkl',
                                   'wb'))
    del advx_batches
    torch.cuda.empty_cache()

