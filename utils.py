import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import random
import itertools
import shutil

from PIL import Image


def prepare_image(image_path, image_size):
    
    image = Image.open(image_path).resize((image_size, image_size))
    image = np.array(image) / 255
    return image

def prepare_dataset(path, l_train=60, l_dev=20, l_test=20):
    dataset = {}
    classes = os.listdir(path)
    for ccls in classes:
        dataset[ccls] = {}
        dataset[ccls]['images'] = glob.glob(os.path.join(path, ccls, '*'))
        total = len(dataset[ccls]['images'])
        img_dev     = np.floor(total * l_dev/100).astype(int)
        img_test    = np.floor(total * l_test/100).astype(int)
        img_train   = total - (img_dev + img_test)

        dataset[ccls]['l_train'] = img_train
        dataset[ccls]['l_dev']   = img_dev
        dataset[ccls]['l_test']  = img_test

        print('_'*10)
        print('##Class    :', ccls)
        print('Total Set  :', total)
        print('Train      :', img_train)
        print('Dev        :', img_dev)
        print('Test       :', img_test)
        print('Using      :', img_train + img_dev + img_test)

    for ttype in ['train', 'dev', 'test']:
        dataset[ttype] = []
        for ccls in classes:
            random.shuffle(dataset[ccls]['images'])
            im_path_im_class = list(zip( dataset[ccls]['images'][ :dataset[ccls][F'l_{ttype}' ]], itertools.repeat(classes.index(ccls))))
            dataset[ttype].extend( im_path_im_class )
            del dataset[ccls]['images'][ :dataset[ccls][F'l_{ttype}' ]]

    print('_'*10)
    print('##Class    :', ccls)
    print('Total Set  :', total)
    print('Train      :', img_train)
    print('Dev        :', img_dev)
    print('Test       :', img_test)
    print('Using      :', img_train + img_dev + img_test)

    save_path = os.path.join(path, "prepared")
    
    for ttype in ['train', 'dev', 'test']:
        os.makedirs(os.path.join(save_path, ttype), exist_ok=True)

        for index, path_class in enumerate(dataset[ttype]):
            dest_filename = os.path.join(save_path, ttype, f'{path_class[1]}-{str(index).rjust(4, "0")}.png')
            shutil.copy(path_class[0], dest_filename)
            print("[{:0>3}] {:20} => {}".format(index, path_class[0], dest_filename))

def load(path='dataset/prepared/dev', img_size=64):
    paths = glob.glob(os.path.join(path, '*'))
    random.shuffle(paths)
    
    X = [prepare_image(x, img_size) for x in paths]
    X = np.array(X)

    Y = list(map(lambda x: os.path.basename(x).split('-')[0], paths))
    Y = np.array(Y).astype(int).reshape(1, -1)

    return X, Y







    
    