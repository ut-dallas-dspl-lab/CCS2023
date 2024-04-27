import sys
import dlib
import torch
from fastai import *
from fastai.vision import *
from fastai.layers import MSELossFlat, CrossEntropyFlat
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

import pandas as pd


def get_faces(files_train, files_valid):
    detector = dlib.get_frontal_face_detector()

    count_train = 0
    for f in files_train:
        # print(f"File being processed:{f}")
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        if len(dets) < 0:
            # print(f"No image found in {f}")
            count_train += 1

    count_valid = 0
    for f in files_valid:
        # print(f"File being processed:{f}")
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        if len(dets) == 0:
            # print(f"No image found in {f}")
            count_valid += 1


def set_device(device_name="cuda:0", num_workers=1):
    dev = device_name if torch.cuda.is_available() else "cpu"
    kwgs = {'num_workers': num_workers, 'pin_memory': True} if dev == device_name else {}

    return dev, kwgs


def remove_incorrect_files(filename="testing_incorrect_class"):
    f = open(filename, "r")
    lines = f.readlines()

    # get list of filenames
    list_files = []
    for line in lines:
        list_files.append(line)

    # remove newline chars and , delimiters from filenames
    list_files = [elem.split(",\n")[0] for elem in list_files]

    # remove additional .jpg extension from Sopam's list of files
    list_files = [".".join(s.split(".")[:-1]) for s in list_files]

    return list_files

    # verify that list of files exists in the folder
    # len(set(filename_list).intersection(set(df.index))) == 342


def get_train_valid_df(files_list):
    df = pd.DataFrame(files_list, columns=["name"])

    df.name = df.name.apply(str)
    df["label"] = df.name.apply(lambda x: re.findall(r"\d{1,3}_\d_\d", x)[0])
    df["label"] = df.label.apply(lambda x: re.sub("_", " ", x))
    df["age"] = df.label.apply(lambda x: int(x.split(" ")[0]))
    df["gender"] = df.label.apply(lambda x: int(x.split(" ")[1]))
    df["ethnicity"] = df.label.apply(lambda x: int(x.split(" ")[2]))

    return df


def get_df_of_concept(df_train, df_valid, concept='age'):
    df_valid['is_valid'] = True
    df_train['is_valid'] = False

    df_concept = pd.concat([df_train, df_valid])
    df_concept.rename(columns={'name': 'image_id'}, inplace=True)
    df_concept = df_concept[['image_id', concept, 'is_valid']]

    return df_concept


def get_df_files(concept='age', df_all_concepts=False, get_all_ages=False):
    list_files = remove_incorrect_files()
    files_train = [str(x) for x in get_image_files("UTKFace") if str(x).split("/")[-1] not in set(list_files)]
    files_valid = [str(x) for x in get_image_files("crop_part1") if str(x).split("/")[-1] not in set(list_files)]

    df_train = get_train_valid_df(files_train)
    df_valid = get_train_valid_df(files_valid)
    
    # age concept
    if not get_all_ages:
        df_train.age = df_train.age.apply(lambda x: False if x <= 30 else True).astype(int)
        df_valid.age = df_valid.age.apply(lambda x: False if x <= 30 else True).astype(int)
    
    # ethnicity concept
    df_train.ethnicity = df_train.ethnicity.apply(lambda x: False if x == 0 else True).astype(int)
    df_valid.ethnicity = df_valid.ethnicity.apply(lambda x: False if x == 0 else True).astype(int)    
    
    # gender concept
    idx = df_valid[df_valid.gender == '3'].index
    df_valid.loc[idx, "gender"] = '1'  # 1 means woman
        
    if df_all_concepts:
        return df_train, df_valid

    df_concept = get_df_of_concept(df_train, df_valid, concept=concept)

    return df_concept


def get_image_databunch(df):
    data_ItemList = (ImageList.from_df(df, '').split_from_df())

    # get the label list
    data_LabelList = data_ItemList.label_from_df()
    
    ds_tfms = get_transforms()
    
    data_bunch = ImageDataBunch.create_from_ll(lls = data_LabelList, bs=64,
                                             size=224, ds_tfms=ds_tfms).normalize(imagenet_stats)
    
    return data_bunch


def verify_dl_index(dl, df_valid):
    img_name_list = None
    for b_id, b in enumerate(dl):
        img_name_list = b['img_name']
        print(b.keys())
        print(b['labels'][0]) #age
        print(b['labels'][1]) #gender
        print(b['labels'][2]) #ethnicity
        break
    
    #verify the order of concepts as follows:
    #compare above to these values
    print(list(df_valid.loc[df_valid['name'].isin(img_name_list)]['age']))
    print(list(df_valid.loc[df_valid['name'].isin(img_name_list)]['gender']))
    print(list(df_valid.loc[df_valid['name'].isin(img_name_list)]['ethnicity']))

    
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


def get_advx_acc(advx_batch_list, advx_dict, model, concept='age', img_key='advx', iterations=None):
    acc_after = 0

    for batch_index in advx_batch_list:
        img_names = advx_batch_list[batch_index]['img_names']
        labels = []

        for _, img in enumerate(img_names):
            labels.append(advx_dict[img][concept])
        labels = torch.tensor(labels).cpu()
        if iterations is not None:
            advs_ = advx_batch_list[batch_index][img_key][iterations]
        else:
            advs_ = advx_batch_list[batch_index][img_key]

        yp_after = model(advs_.cuda()).cpu()
        acc_after += (yp_after.max(dim=1)[1] == labels).sum().item()

    return acc_after

def get_advx_acc_across_iters(advx_batches, df_advx_dict, model, concept, img_key, num_iters=11, len_df=1000):
    acc_after_dict = {i: None for i in range(num_iters)}

    for i in range(num_iters):
        
        acc_after = np.round(get_gen_acc(advx_batch_list=advx_batches, advx_dict=df_advx_dict, 
                                         model=model, concept=concept, img_key=img_key, iterations=i)/len_df*100, 2)
        acc_after_dict[i] = acc_after
        
    return acc_after_dict


if __name__ == '__main__':
    pass
