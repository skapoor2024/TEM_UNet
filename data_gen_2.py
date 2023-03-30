"""
Generating dataset for segmenation.
"""


import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tiler import Tiler
import os

def data_gen():

    pth = os.getcwd()
    #input the hdf5 file data directory
    data_dir = "pure_iron_grain_data_sets.hdf5"

    file_h5 = h5py.File(data_dir, "r")

    real_image = np.array(file_h5['image'])/255
    real_label = np.array(file_h5["label"])
    #real_boundary = np.array(file_h5['boundary'])/255

    del(file_h5)

    #applying normalization on the complete dataset

    x,y,z = real_image.T.shape
    # normalization of the input dataset image
    real_image_mean = np.sum(real_image)/(x*y*z)
    real_image_std = np.sqrt(np.sum(np.square(real_image - real_image_mean))/(x*y*z))

    real_image = (real_image - real_image_mean)/real_image_std

    img_train,img_test,bound_train,bound_test = train_test_split(real_image.T,
                                                                 real_label.T,
                                                                 test_size=0.2,
                                                                 random_state=42)
    del(real_image)
    del(real_label)

    tiler = Tiler(data_shape = (y,z),
                  tile_shape = (256,256),
                  overlap = 0.51,
                  mode = 'reflect')
    
    print(img_train.shape)
    print(bound_train.shape)
    print(img_test.shape)
    print(bound_test.shape)

    #train = []
    train_path = os.path.join(pth,'train_2')
    if os.path.exists(train_path) == False:
        os.mkdir(train_path)

    for i in range(img_train.shape[0]):

        for t1,t2 in zip(tiler(img_train[i]),tiler(bound_train[i])):

            ip = np.expand_dims(t1[1],axis = 0)
            op = np.expand_dims(t2[1],axis = 0)
            #train.append(np.concatenate((ip,op),axis = 0))
            di = os.path.join(pth,'train_2',str(i)+'_'+str(t1[0]))
            arr = np.concatenate((ip,op),axis = 0)
            np.save(di,arr)

    #train = np.array(train)
    #print(train.shape)

    #np.save('train',train)

    #del(train)

    ##tes = []

    label_path = os.path.join(pth,'test_2')
    if os.path.exists(label_path) == False:
        os.mkdir(label_path)

    for k in range(img_test.shape[0]):

        for t3,t4 in zip(tiler(img_test[k]),tiler(bound_test[k])):

            ip = np.expand_dims(t3[1],axis = 0)
            op = np.expand_dims(t4[1],axis = 0)
            di = os.path.join(pth,'test_2',str(k)+'_'+str(t3[0]))
            arr = np.concatenate((ip,op),axis = 0)
            np.save(di,arr)
            
            #tes.append(np.concatenate((ip,op),axis = 0))

    # tes = np.array(tes)
    # print(tes.shape)

    # np.save('test',tes)
    
    # del(tes)

if __name__ == '__main__':
    
    data_gen()


            
