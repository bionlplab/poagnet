import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from models import vgg16, ensemble_vgg
from modelnew import Res, ensemble_res, Den, ensemble_model, ensemble_resden,ensemble_resden1, multiscale_Net, Multiscale_multimodel,triplescale_Net, Res1,Den1,mobv2, vgg_16, nas,naslarge, xception,ensemble_resden_double,ensemble_resden2,Res_double,Den_double,ensemble_resden_siamese,ensemble_resden_siamese1,ensemble_resden_siamese2
from modelnew import ensemble_resden_siamese3,ensemble_resden_siamese4
from data_load_cv import load_data
from data_load_cv_vf import load_data_vf
# from data_load_cv_double import load_data_double
from data_load_cv_siamese import load_data_siamese
import numpy as np
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import copy
from skimage.transform import resize
from ImageGenerator_cv import DataGenerator
# from ImageGenerator_cv_double import DataGenerator_double
from ImageGenerator_cv_siamese import DataGenerator_siamese
path = '/prj0129/mil4012/glaucoma' 


#def weighted_binary_crossentropy(y_true, y_pred) :
#    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
#    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
#    logloss = -(y_true * K.log(y_pred) * weight[0] + weight[1] * (1 - y_true) * K.log(1 - y_pred))
#    return K.mean(logloss, axis=-1)

def weighted_binary_crossentropy(y_true, y_pred) :
    weight = 1 - K.sum(y_true) /(K.sum(y_true) + K.sum(1 - y_true))
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight +  (1 - y_true) * K.log(1 - y_pred) * (1-weight))
    return K.mean(logloss, axis=-1)


def get_train_test_p_id(glaucoma_list,normal_list, fold, total_num_fold):
    num_glaucoma = len(glaucoma_list) // 2
    test_num_glaucoma = num_glaucoma // total_num_fold * 2
    
    num_normal = len(normal_list) // 2
    test_num_normal = num_normal // total_num_fold * 2

    if fold == total_num_fold:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):,:]
        test_normal = normal_list[((fold-1) * test_num_normal):,:]
        train_glaucoma = glaucoma_list[0:((fold-1) * test_num_glaucoma),:]
        train_normal = normal_list[0:((fold-1) * test_num_normal),:]
    else:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):fold * test_num_glaucoma,:]
        test_normal = normal_list[((fold-1) * test_num_normal):fold * test_num_normal,:]
        train_glaucoma = np.concatenate((glaucoma_list[0:((fold-1) * test_num_glaucoma),:], glaucoma_list[(fold * test_num_glaucoma):,:]), axis=0)
        train_normal = np.concatenate((normal_list[0:((fold-1) * test_num_normal),:], normal_list[(fold * test_num_normal):,:]), axis=0)
    

    valiation_glaucoma = train_glaucoma[int(0.8*len(train_glaucoma) // 2) * 2:,:]
    validation_normal = train_normal[(len(train_normal) - len(valiation_glaucoma)):,:] 
    train_glaucoma = train_glaucoma[0:(len(train_glaucoma)-len(valiation_glaucoma)) :]
    train_normal = train_normal[0:(len(train_normal) - len(validation_normal)),:]
    le_train_glaucoma = len(train_glaucoma)
    le_train_normal = len(train_normal)
    le_validation_glaucoma = len(valiation_glaucoma)
    le_validation_normal = len(validation_normal)
    
    le_test_glaucoma = len(test_glaucoma)
    le_test_normal = len(test_normal)
    
    
    train_name = np.concatenate((train_normal, train_glaucoma), axis=0)
    validation_name = np.concatenate((validation_normal, valiation_glaucoma), axis=0)
    test_name = np.concatenate((test_normal, test_glaucoma), axis=0)
    return train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal



def train_simense(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    model.fit(x_train,y_train, validation_data=(x_val, y_val), batch_size= 24, epochs=epochs
              ,shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')


def train(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    datagen.fit(x_train)
#    print('data tpye of x_train is', type(x_train), type(y_train))
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    print('the program start to fit')
    model.fit_generator(datagen.flow(x_train, y_train, batch_size= 32), validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // 32, epochs=epochs
                        , shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')


# for double input
def get_test(x_test, y_test, model, weights):
#def test(x_test, y_test, model, weights):
    p_test = np.zeros(len(y_test))
    model.load_weights(weights)
    x_test1 = x_test[0]
    x_test2 = x_test[1]
    for i in range(len(y_test)):
        
        x1 = x_test1[i, :, :, :]
        x1 = x1.reshape(1, 224, 224, 3)
        x2 = x_test2[i, :, :, :]
        x2 = x2.reshape(1, 224, 224, 3)
        p_test1 = model.predict([x1,x2])
        
        x1_vertical_flip = copy.deepcopy(x1)
        x1_vertical_flip = np.squeeze(x1_vertical_flip)
        x1_vertical_flip = np.flipud(x1_vertical_flip)
        x1_vertical_flip = x1_vertical_flip.reshape(1, 224, 224, 3)
        
        x2_vertical_flip = copy.deepcopy(x2)
        x2_vertical_flip = np.squeeze(x2_vertical_flip)
        x2_vertical_flip = np.flipud(x2_vertical_flip)
        x2_vertical_flip = x2_vertical_flip.reshape(1, 224, 224, 3)
        
        #p_test_vertical_flip = model.predict([x1_vertical_flip,x2_vertical_flip])
        p_test_vertical_flip = model.predict([x1,x2_vertical_flip])
        
        x1_horizontal_flip = copy.deepcopy(x1)
        x1_horizontal_flip = np.squeeze(x1_horizontal_flip)
        x1_horizontal_flip = np.flipud(x1_horizontal_flip)
        x1_horizontal_flip = x1_horizontal_flip.reshape(1, 224, 224, 3)
        
        x2_horizontal_flip = copy.deepcopy(x2)
        x2_horizontal_flip = np.squeeze(x2_horizontal_flip)
        x2_horizontal_flip = np.flipud(x2_horizontal_flip)
        x2_horizontal_flip = x2_horizontal_flip.reshape(1, 224, 224, 3)
        
        
       # p_test_horizontal_flip = model.predict([x1_horizontal_flip,x2_horizontal_flip])
        p_test_horizontal_flip = model.predict([x1,x2_horizontal_flip])
        
#         p_test[i] = p_test1*2/3 + ((p_test_vertical_flip + p_test_horizontal_flip))/3
        p_test[i] = (p_test1 + p_test_vertical_flip + p_test_horizontal_flip)/3
    
    return p_test

# for single input
def get_test1(x_test, y_test, model, weights):
#def test(x_test, y_test, model, weights):
    p_test = np.zeros(len(y_test))
    model.load_weights(weights)
    for i in range(len(y_test)):
        
        x = x_test[i, :, :, :]
        x = x.reshape(1, 224, 224, 3)
        p_test1 = model.predict(x)
        
        x_vertical_flip = copy.deepcopy(x)
        x_vertical_flip = np.squeeze(x_vertical_flip)
        x_vertical_flip = np.flipud(x_vertical_flip)
        x_vertical_flip = x_vertical_flip.reshape(1, 224, 224, 3)
        

        
        p_test_vertical_flip = model.predict(x_vertical_flip)
        
        x_horizontal_flip = copy.deepcopy(x)
        x_horizontal_flip = np.squeeze(x_horizontal_flip)
        x_horizontal_flip = np.flipud(x_horizontal_flip)
        x_horizontal_flip = x_horizontal_flip.reshape(1, 224, 224, 3)
        
        
        p_test_horizontal_flip = model.predict(x_horizontal_flip)
        
        p_test[i] = p_test1*2/3 + ((p_test_vertical_flip + p_test_horizontal_flip))/3
    
    return p_test

def test(x_test, y_test, model, weights):
#def test(x_test, y_test, model, weights):
    model.load_weights(weights)
    p_test = model.predict(x_test)
    
#     p_test = 1 - p_test
#     y_test = 1 - y_test
#    np.savetxt(weights[i][:-3]+'.txt', np.reshape(p_test,(len(p_test),)))
#     p_test = get_test(x_test, y_test, model, weights)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]
#    print(p_test)
#    print(p_classes)
    print('the shape of test is', p_test.shape)
    accuracy = accuracy_score(y_test, p_classes)
    print('classification accuracy: ', accuracy)
    precision = precision_score(y_test, p_classes)
    print('precision: ', precision)
    recall = recall_score(y_test, p_classes)
    print('recall: ', recall)
    f1 = f1_score(y_test, p_classes)
    print('F1 score: ', f1)
    auc = roc_auc_score(y_test, p_test)
    print('AUC: ', auc)
    matrix = confusion_matrix(y_test, p_classes)
    print(matrix)
    result_den = np.concatenate((y_test, p_classes,p_test), axis=-1)
    np.savetxt('predict.txt', result_den)
    return





if __name__ == '__main__':
    w_path2 = '/prj0129/mil4012/glaucoma/weights/glaucoma_DenseNet201.h5'
    w_path1 = '/prj0129/mil4012/glaucoma/weights/glaucoma_ResNet152.h5'
    w_path22 = '/prj0129/mil4012/glaucoma/weights/glaucoma_DenseNet201double3_ohtsnew.h5'
    w_path11 = '/prj0129/mil4012/glaucoma/weights/glaucoma_ResNet152double3_ohts.h5'
    model_path = '/prj0129/mil4012/glaucoma/weights/glaucoma_MultiNet1sp_5.h5'
#     w_path2 = 'glaucoma_DenseNet201LAG_5.h5'
#     w_path1 = 'glaucoma_ResNet152LAG_5.h5'
    #model = vgg16(img_size=(224, 224, 3), scale=1,dropout=False)
    #model.load_weights('vgg16_glaucoma.h5')
    #model.summary()
   # model = vgg_16(vgg_en='vgg_16',img_size=(224, 224, 3), dropout=False)
   # model = nas(nas_en ='nasmobile',img_size=(224, 224, 3), dropout=False)
  #  model = naslarge(naslarge_en = 'naslarge',img_size=(331, 331, 3), dropout=False)
   # model = xception(xcep_en = 'xception',img_size=(299, 299, 3), dropout=False)
   # model = mobv2(mob_en='mobv2',img_size=(224, 224, 3), dropout=False)
   # model = ensemble_vgg(img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
   # model = ensemble_res(res_en=['res50','res101','res152'],img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
   # model = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
   # model = Den_double(w_path2,den_en='den201',img_size=(224, 224, 6), dropout=False,flag = 1)
   # model = Res1(res_en='res152',img_size=(224, 224, 6), dropout=False)
   # model = Den1(den_en='den201',img_size=(224, 224, 6), dropout=False)
   # model = Res(res_en='res152',img_size=(224, 224, 3), dropout=False)
#     model = Res_double(w_path1,res_en='res152',img_size=(224, 224, 6), dropout=False,flag = 1)
  #  model = ensemble_model(model_en=['res152','den201'],img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
  #  model = ensemble_resden(img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
   # model = ensemble_resden2(model_path,w_path1,w_path2,w_path11,w_path22,img_size=(224, 224, 6), model_input=Input((224, 224, 3)),dropout=False,flag = 0)
   # model = ensemble_resden_siamese(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 0) #flag=1: imagenet, flag=0: ohts
   # model = ensemble_resden_siamese1(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
   # model = ensemble_resden_siamese2(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
    # the proposed method
    model = ensemble_resden_siamese3(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
   # model = ensemble_resden_siamese4(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
   # model = ensemble_resden1(w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
   # model = ensemble_resden_double(w_path1,w_path2,img_size=(224, 224, 6), model_input=Input((224, 224, 6)),dropout=False,flag=1)
  #  model = multiscale_Net(net='res152',img_size=(224, 224, 3), dropout=False, flag=1)
  #  model = Multiscale_multimodel(img_size=(224, 224, 3), dropout=False, flag=1)
  #  model = triplescale_Net(net='den201',img_size=(224, 224, 3), dropout=False, flag=0)
#     model.load_weights('glaucoma_ResNet152AREDS.h5')
    learning_rate = 5*1e-5
    epochs = 15
    weights_path = '/prj0129/mil4012/glaucoma/weights/DenseNet201conv_ohts_sosiamese_au_w55.h5'
    model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_binary_crossentropy)
    
    label_path1 = os.path.join(path,'glaucoma_list_patient.csv')
    tmp = np.loadtxt(label_path1, dtype=np.str, delimiter=",")

    label_path2 = os.path.join(path,'normal_list_patient.csv')
    tmp_1 = np.loadtxt(label_path2, dtype=np.str, delimiter=",")

    tmp = tmp[1:,:] 
    tmp_1 = tmp_1[1:,:]
    fold = 1
    total_num_fold = 5
    x_size = 224
    y_size = 224
    train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal = get_train_test_p_id(tmp, tmp_1, fold, total_num_fold)
    
    #print(test_name)

    #/prj0129/mil4012/   /home/mil4012/glaucoma
   # val_images,val_labels,test_images,test_labels = load_data(data_path='/prj0129/mil4012/image_crop/',label_path='/prj0129/mil4012/lab_new.csv',validation_name=validation_name,test_name=test_name)
    val_images,val_labels,test_images,test_labels,ind_start = load_data_siamese(x_size,y_size, data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),
                                                                                                                          image_s_path=os.path.join(path,'patient_s.csv'), uncentain_path=os.path.join(path,'uncentain.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)

    
    test_images_vf, test_labels_vf = load_data_vf(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),vf_path=os.path.join(path,'patient_vf1.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)

    print('the shape of testing image:', np.shape(test_images))

    ind_start = ind_start.astype(np.int)
    np.savetxt('ind_start.txt', np.reshape(ind_start,(len(ind_start),)))
    
    train_generator, train_labels = DataGenerator_siamese(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),train_normal=train_normal,train_glaucoma=train_glaucoma)

    train_labels = train_labels.astype(np.float)
    val_labels = val_labels.astype(np.float)
    test_labels = test_labels.astype(np.float)
#     test_labels_s = test_labels_s.astype(np.float)
#     test_labels_un = test_labels_un.astype(np.float)
    test_labels_vf = test_labels_vf.astype(np.float)
    print('the shape of training image:', np.shape(train_generator))


    index_p = np.loadtxt('/prj0129/mil4012/glaucoma/ind_replace.txt')
    index_1=np.argwhere(train_labels==1)
    index_1 = np.reshape(index_1,(len(index_1),))
    index_2=np.argwhere(train_labels==0)
    index_2 = np.reshape(index_2,(len(index_2),))
    train_generator1 = [train_generator[0][index_1],train_generator[1][index_1]]
    train_labels1 = train_labels[index_1]
    train_generator2 = [train_generator[0][index_2],train_generator[1][index_2]]
    train_labels2 = train_labels[index_2]
    print(type(train_generator1))
    print(type(train_generator2))
    
    print('the shape of train_generator:', np.shape(train_generator))
    print('the shape of training label:', np.shape(train_labels))
    print(type(train_generator))
    print(type(train_labels))
    
#     train_generator1 = np.delete(train_generator1,index_p.astype(np.int).tolist(),axis=1)
#     train_labels1 = np.delete(train_labels1,index_p.astype(np.int).tolist(),axis=0)
    
    print('the shape of train_generator1:', np.shape(train_generator1))
    print('the shape of train_labels1:', np.shape(train_labels1))
    
    temp1 = copy.deepcopy(train_generator1)
    temp2 = copy.deepcopy(train_generator2)
    train_generator1 = np.concatenate((temp1[0],temp2[0],temp2[0]),axis=0)
    train_generator2 = np.concatenate((temp1[1],temp2[1],temp2[1]),axis=0)
    train_generator = [train_generator1,train_generator2]
    
    
#     train_generator = np.concatenate((train_generator1,train_generator2,train_generator2),axis=1)
    train_labels = np.concatenate((train_labels1,train_labels2,train_labels2),axis=0)
    
    print('the shape of train_generator:', np.shape(train_generator))
    print('the shape of train_labels:', np.shape(train_labels))
    
    print(type(train_generator))
    print(type(train_labels))
    
    train_generator1 =[]
    train_generator2 =[]  
    temp1 = []
    temp2 = []
    
   
   ## single input
#   
   train_simense(train_generator, (1-train_labels), val_images, (1-val_labels), model, epochs, weights_path)
 #   train(train_generator, train_labels, val_images, val_labels, model, epochs, weights_path)
    
#     test_images = np.concatenate((test_images, val_images, train_generator), axis=0)
#     test_labels= np.concatenate((test_labels, val_labels, train_labels), axis=0)

    test(test_images, (1-test_labels), model, weights_path)
#   # test(test_images_un, test_labels_un, model, weights_path)
   

