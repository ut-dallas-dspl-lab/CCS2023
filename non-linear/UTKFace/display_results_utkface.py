import os
print(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import pandas as pd
from numpy.linalg import norm
from decimal import Decimal, ROUND_HALF_EVEN
from fb_utils import *
#from UTKFace_train_utils import *


def get_gen_acc(dl, model, concept_index, foolbox_model=False):
    acc_after = 0

    for batch_index, batch in enumerate(dl):
        concept_label = batch['labels'][concept_index]
        if not foolbox_model:
            imgs = batch['image_norm']
            yp_after = model(imgs.cuda()).cpu()
        else:
            imgs = batch['image']
            yp_after = model(ep.astensor(imgs.cuda())).raw.cpu()

        acc_after += (yp_after.max(dim=1)[1] == concept_label).sum().item()

    return acc_after



def get_gen_loss(dl, model, concept_index, foolbox_model=False, avg_overall_loss=True):
    #acc_after = 0
    avg_loss_list = []

    for batch_index, batch in enumerate(dl):
        concept_label = batch['labels'][concept_index]
        if not foolbox_model:
            imgs = batch['image_norm']
            loss_tensor = model(imgs.cuda()).cpu()
        else:
            imgs = batch['image']
            loss_tensor = model(ep.astensor(imgs.cuda())).raw.cpu()

        if avg_overall_loss:
            avg_loss = torch.mean(loss_tensor)
            avg_loss_list.append(avg_loss.item())
        else:
             for i in range(len(concept_label)):
                curr_label = concept_label[i]
                curr_loss = loss_tensor[i][curr_label]
                avg_loss_list.append(curr_loss.item())
        
    avg_loss = sum(avg_loss_list)/len(avg_loss_list)
    print(avg_loss)
    return avg_loss


# In[12]:


def get_advx_acc_across_iters(advx_batches, df_advx_dict, model, concept, img_key, num_iters=11, len_df=1000):
    acc_after_dict = {i: None for i in range(num_iters)}

    for i in range(num_iters):
        acc_after = np.round(get_advx_acc(advx_batch_list=advx_batches, advx_dict=df_advx_dict,
                                          model=model, concept=concept, img_key=img_key, iterations=i) / len_df * 100,
                             2)
        acc_after_dict[i] = acc_after

    return acc_after_dict


# In[13]:


#given advx_batches
def get_advx_acc(advx_batch_list, advx_dict, model, concept='age', img_key='advx'):
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


# In[14]:


def get_minmax_loss_diff_label(dataloader, model, label_idx, is_max=True):
    label_dict = {}
    label_dict_groundtruth = {}
    
    for batch_idx, batch in enumerate(dataloader):
        
        minmax_label = []
        groundtruth_label = []
        
        concept_label = batch['labels'][label_idx]
        imgs = batch['image_norm']
        loss_val = model(imgs.cuda()).cpu()
        
        for i in range(len(concept_label)):
            curr_label = concept_label[i]
            
            label1 = abs(curr_label-2)
            loss1 = abs(loss_val[i][curr_label] - loss_val[i][label1])
            
            label2 = abs(curr_label-1)
            loss2 = abs(loss_val[i][curr_label] - loss_val[i][label2])
            
            #print(f"label1:{label1}, label2:{label2}, curr_label:{curr_label}")
            #print(f"loss1:{loss1}")
            #print(f"loss2:{loss2}")
            
            if is_max:
                if loss1 > loss2:
                    minmax_label.append(label1.item())
                else:
                    minmax_label.append(label2.item())
            else:
                if loss1 > loss2:
                    minmax_label.append(label2.item())
                else:
                    minmax_label.append(label1.item())
            groundtruth_label.append(curr_label.item())
            
            #print(f"groundtruth: {groundtruth_label}")
            #print(f"max_label:{minmax_label}")
            #sys.exit()

        #print(minmax_label)
        label_dict[batch_idx] = minmax_label
        label_dict_groundtruth[batch_idx] = groundtruth_label

    return label_dict, label_dict_groundtruth


# In[15]:


def custom_advx_wrapper(parent_dir, advx_dict, df_advx, dl_dict, norm='linf', concept_list=None):
    if concept_list is None:
        concept_list = [('gender', model_gender), ('age', model_age), ('glasses', model_glasses), ('pretty', model_pretty)]
    
    file_list = sorted(os.listdir(parent_dir))
    for curr_file in file_list:
        print(f"File_name: {curr_file}")
        attack_concepts = curr_file.split("_"+norm+"_")[-1].split("_")[0]
        defend_concepts = curr_file.split("_"+norm+"_")[-1].split("_")[1]
        print(f"Custom {norm} norm advx with Attack concepts: {attack_concepts} and Defend concepts: {defend_concepts}")
        
        advx_batches = pickle.load(open(os.path.join(parent_dir, curr_file), 'rb'))
        dict_X_to_label_mapping = get_x_to_label_mapping(advx_batches, dl_dict)
        if norm=='linf':
            assert_lp_bound(advx_batches, dict_X_to_label_mapping, eps=0.3, norms='linf')
        elif norm=='l2':
            assert_lp_bound(advx_batches, dict_X_to_label_mapping, eps=4.0, norms='l2')
        else:
            raise NotImplementedError
        
        for concept_tuple in concept_list:
            concept = concept_tuple[0]
            model = concept_tuple[1]
            acc_ = np.round(get_advx_acc(advx_batch_list=advx_batches, advx_dict=advx_dict, model=model, concept=concept, img_key='advx', iterations=None)/len(df_advx)*100, 2)
            print(f"Acc of {concept} concept: {acc_}")
            del acc_ 
        del advx_batches
    
        print()


# ## Main

# In[16]:


#df_advx = pd.read_csv('df_gender_age_UTKFace.csv') # not relevant
#df_advx = pd.read_csv('df_utkface_4000.csv') # not relevant
#df_advx_1000_dict = get_pdadvx_dict_utkface(df_advx) # not relevent
df_advx = pd.read_csv('df_utkface_1000.csv')
df_advx_1000_dict = get_pdadvx_dict_utkface(df_advx)


df_advx_multi = pd.read_csv('df_utkface_1000_multi.csv')
df_advx_1000_dict_multi = get_pdadvx_dict_utkface(df_advx_multi)


torch.cuda.set_device(0)
device, kwargs = set_device(device_name="cuda:0")
print(device, kwargs)
torch.cuda.empty_cache()

#model_age_multi, fmodel_age_multi = get_foolbox_model(weights_file='stage-2-rn50-age-multi-utk-final.pth', dev=device, 
#                                          num_classes=3, is_torch=False, preprocess=True, models_dir='models_utkface')

#dataloader_multi = get_dataloader(kwargs_func=kwargs, batch_size=20, pd_advx=df_advx_multi, root_dir='', img_pkl_file=None, is_for_train=False)
#dataloader_dict_multi_1000 = get_dataloader_dict(data_loader=dataloader_multi)

model_age, fmodel_age = get_foolbox_model(weights_file='stage-2-rn50-age-utk.pth', dev=device, 
                                          num_classes=2, is_torch=False, preprocess=True, models_dir='models_utkface')


model_gender, fmodel_gender = get_foolbox_model(weights_file='stage-2-rn50-gender-utk.pth', dev=device, 
                                                num_classes=2, is_torch=False, preprocess=True, models_dir='models_utkface')

model_ethnicity, fmodel_ethnicity = get_foolbox_model(weights_file='stage-2-rn50-ethnicity-utk.pth', dev=device, 
                                          num_classes=2, is_torch=False, preprocess=True, models_dir='models_utkface')


dataloader = get_dataloader(kwargs_func=kwargs, batch_size=20, pd_advx=df_advx, root_dir='', img_pkl_file=None, is_for_train=False)
dataloader_dict_1000 = get_dataloader_dict(data_loader=dataloader)


# get the conceptDict lst and model_list
concept_dict = {'age': 0, 'gender': 1, 'ethnicity': 2}

#dataloader_dict_multi_1000['crop_part1/18_1_0_20170109214216731.jpg.chip.jpg']['labels']dataloader_dict_1000['crop_part1/18_1_0_20170109214216731.jpg.chip.jpg']['labels']
#dataloader_multi = get_dataloader(kwargs_func=kwargs, batch_size=20, pd_advx=df_advx_multi, root_dir='', img_pkl_file=None, is_for_train=False)
#dataloader_dict_multi_1000= get_dataloader_dict(data_loader=dataloader_multi)
# ## Accuracy of original models on sample dataset


acc_before = get_gen_acc(dataloader, model_age, concept_dict['age'])/len(df_advx)*100
print(f"Accuracy of age before attack: {np.round(acc_before, 2)}%")
acc_before = get_gen_acc(dataloader, model_gender, concept_index=concept_dict['gender'])/len(df_advx)*100
print(f"Accuracy of gender before attack: {np.round(acc_before, 2)}%")
acc_before = get_gen_acc(dataloader, model_ethnicity, concept_index=concept_dict['ethnicity'])/len(df_advx)*100
print(f"Accuracy of ethnicity before attack: {np.round(acc_before, 2)}%")


os.listdir('final_exp_utkface')


concept_dict = {'age': 0, 'gender': 1, 'ethnicity': 2}
concept_list = [('gender', model_gender), ('age', model_age), ('ethnicity', model_ethnicity)]


advx_batches = pickle.load(open('final_exp_utkface/pgd_custom_l2_age_gender_eps_4_step_40_iter_200.pkl', 'rb'))
dict_X_to_label_mapping = get_x_to_label_mapping(advx_batches, dataloader_dict_1000)


#acc_gender = get_advx_acc_across_iters(advx_batches=advx_batches, df_advx_dict=df_advx_1000_dict, model=model_gender, concept='gender', 
#                                       img_key='advx', num_iters=2, len_df=len(df_advx))
print("FOOLBOX LINF ADVX EXAMPLES OF ATTACK AGE:")
acc_gender = get_advx_acc(advx_batches, df_advx_1000_dict, model_gender, concept='gender', 
                                       img_key='advx')/len(df_advx)*100
print(f"Acc of gender concept: {np.round(acc_gender, 2)}%")
#acc_age = get_advx_acc_across_iters(advx_batches=advx_batches, df_advx_dict=df_advx_1000_dict, model=model_age, concept='age', 
#                                    img_key='advx', num_iters=2, len_df=len(df_advx))
acc_age = get_advx_acc(advx_batches, df_advx_1000_dict, model_age, concept='age', 
                                    img_key='advx')/len(df_advx)*100
print(f"Acc of age concept: {np.round(acc_age, 2)}%")
#acc_ethnicity = get_advx_acc_across_iters(advx_batches=advx_batches, df_advx_dict=df_advx_1000_dict, model=model_ethnicity, concept='ethnicity', 
#                                        img_key='advx', num_iters=2, len_df=len(df_advx))
acc_ethnicity = get_advx_acc(advx_batches, df_advx_1000_dict, model_ethnicity, concept='ethnicity', 
                                        img_key='advx')/len(df_advx)*100

print(f"Acc of ethnicity concept: {np.round(acc_ethnicity,2)}%")

del acc_age, acc_ethnicity, acc_gender, advx_batches
torch.cuda.empty_cache()



advx_batches = pickle.load(open('final_exp_utkface/pgd_custom_l2_age_genderEthnicity_eps_4_step_40_iter_200.pkl', 'rb'))
dict_X_to_label_mapping = get_x_to_label_mapping(advx_batches, dataloader_dict_1000)

print("FOOLBOX LINF ADVX EXAMPLES OF ATTACK AGE protecting gender and ethnicity:")
acc_gender = get_advx_acc(advx_batches, df_advx_1000_dict, model_gender, concept='gender', 
                                       img_key='advx')/len(df_advx)*100
acc_age = get_advx_acc(advx_batches, df_advx_1000_dict, model_age, concept='age', 
                                    img_key='advx')/len(df_advx)*100
acc_ethnicity = get_advx_acc(advx_batches, df_advx_1000_dict, model_ethnicity, concept='ethnicity', 
                                        img_key='advx')/len(df_advx)*100

print(f"Acc of age concept: {np.round(acc_age, 2)}%")
print(f"Acc of gender concept: {np.round(acc_gender, 2)}%")
print(f"Acc of ethnicity concept: {np.round(acc_ethnicity,2)}%")


# In[68]:


del acc_age, acc_ethnicity, acc_gender, advx_batches
torch.cuda.empty_cache()


