from keras.layers import Input, GaussianNoise, Dropout, Activation, BatchNormalization, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, average, add, merge
from keras.layers import Lambda
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import AveragePooling2D
from keras import regularizers
from keras.applications import Xception, ResNet50, InceptionV3, DenseNet121, DenseNet169, DenseNet201, VGG16, VGG19, ResNet101, ResNet152, NASNetMobile, NASNetLarge
from keras.applications import ResNet50V2, ResNet101V2, ResNet152V2,MobileNetV2
import numpy as np
import tensorflow as tf
import keras.backend as K
def conv2d_block(x, num_filters, filter_size, with_bn, activation, name=None):
    num_filters = int(num_filters)
    x = Conv2D(num_filters, filter_size, padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    if with_bn == True:
        x = BatchNormalization(axis=-1)(x)
    if name != None:
        x = Activation(activation, name=name)(x)
    else:
        x = Activation(activation)(x)
    return x


def vgg16(img_size, scale, dropout):
    input_ct = Input(img_size)

    # noise_input = GaussianNoise(0.1)(input_ct)

    x = conv2d_block(input_ct, num_filters=64 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv1_1')
    x = conv2d_block(x, num_filters=64 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv1_2')
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv2d_block(x, num_filters=128 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv2_1')
    x = conv2d_block(x, num_filters=128 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv2_2')
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv2d_block(x, num_filters=256 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv3_1')
    x = conv2d_block(x, num_filters=256 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv3_2')
    x = conv2d_block(x, num_filters=256 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv3_3')
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv2d_block(x, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv4_1')
    x = conv2d_block(x, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv4_2')
    x = conv2d_block(x, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv4_3')
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv2d_block(x, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv5_1')
    x = conv2d_block(x, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv5_2')
    x = conv2d_block(x, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv5_3')
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D(name='globalaveragepooling')(x)

    x = Dense(int(128 * scale), activation='relu', name='fc1')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(int(64 * scale), activation='relu', name='fc2')(x)

    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=[input_ct], outputs=[x])

    return model
    
    
def dense_vgg16(img_size, scale, dropout):
    input_ct = Input(img_size)

    # noise_input = GaussianNoise(0.1)(input_ct)

    conv1_1 = conv2d_block(input_ct, num_filters=64 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv1_1')
    conv1_2 = conv2d_block(conv1_1, num_filters=64 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv1_2')
    conv1_2 = add([conv1_1, conv1_2])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = conv2d_block(pool1, num_filters=128 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv2_1')
    conv2_2 = conv2d_block(conv2_1, num_filters=128 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv2_2')
    conv2_2 = add([conv2_1, conv2_2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv3_1 = conv2d_block(pool2, num_filters=256 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv3_1')
    conv3_2 = conv2d_block(conv3_1, num_filters=256 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv3_2')
    conv3_3 = conv2d_block(conv3_2, num_filters=256 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv3_3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

    conv4_1 = conv2d_block(pool3, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv4_1')
    conv4_2 = conv2d_block(conv4_1, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv4_2')
    conv4_3 = conv2d_block(conv4_2, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv4_3')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)

    conv5_1 = conv2d_block(pool4, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv5_1')
    conv5_2 = conv2d_block(conv5_1, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv5_2')
    conv5_3 = conv2d_block(conv5_2, num_filters=512 * scale, filter_size=(3,3), with_bn=True, activation='relu', name='conv5_3')
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_3)

    x = GlobalAveragePooling2D(name='globalaveragepooling')(pool5)

    x = Dense(int(128 * scale), activation='relu', name='fc1')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(int(64 * scale), activation='relu', name='fc2')(x)

    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=[input_ct], outputs=[x])

    return model

def vgg_16(vgg_en,img_size,dropout):
    input_ct = Input(img_size)
    if vgg_en == 'vgg_16':       
        model_backbone = VGG16(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model

def nas(nas_en,img_size,dropout):
    input_ct = Input(img_size)    
    if nas_en == 'nasmobile':       
        model_backbone = NASNetMobile(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model

def naslarge(naslarge_en,img_size,dropout):
    input_ct = Input(img_size)    
    if naslarge_en == 'naslarge':       
        model_backbone = NASNetLarge(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model    

def mobv2(mob_en,img_size,dropout):
    input_ct = Input(img_size)
    if mob_en == 'mobv2':       
        model_backbone = MobileNetV2(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model

def xception(xcep_en,img_size,dropout):
    input_ct = Input(img_size)
    if xcep_en == 'xception':       
        model_backbone = Xception(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model

def Res(res_en, img_size, dropout):
    input_ct = Input(img_size)
    if res_en == 'res50':       
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    if res_en == 'res101':       
        model_backbone = ResNet101(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    
    if res_en == 'res152': 
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        return model

    
# input is 6 channel, add 1*1 filter to covert it to 3 channel, then feed into resnet152.flag =0 used the imagNet as the pretrained weight, flag=1 used the pretrained weight based #on ResNet201 trained on OHTS dataset  
def Res_double(w_path1,res_en, img_size, dropout,flag):
    input_ct = Input(img_size)
    if flag == 0:
        if res_en == 'res152':
            input_3 = conv2d_block(input_ct, num_filters=3, filter_size=(1,1), with_bn=True, activation='relu', name='con_in')
            model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=(224,224,3),pooling='avg')
            output_backbone = model_backbone(input_3)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input_ct], outputs=[x])
    else:
        if res_en == 'res152':
            input_3 = conv2d_block(input_ct, num_filters=3, filter_size=(1,1), with_bn=True, activation='relu', name='con_in')
            model_backbone = Res(res_en='res152',img_size=(224, 224, 3), dropout=False)
            model_backbone.load_weights(w_path1)
            x= model_backbone(input_3)
            model = Model(inputs=[input_ct], outputs=[x])
            
        
    return model
    
def Resnew(res_en, img_size, dropout):
    input_ct = Input(img_size)
    if res_en == 'res152':     
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        output_backbone = model_backbone(input_ct)
        output_backbone = AveragePooling2D(pool_size=(7, 7),name='globalaveragepooling')(output_backbone)
      #  output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        return model

# ResNet without pretrain 
def Res1(res_en, img_size, dropout):
    input_ct = Input(img_size)
    model_backbone = ResNet152(include_top=False,weights=None,input_shape=img_size,pooling='avg')
    output_backbone = model_backbone(input_ct)
    x = Dense(1, activation='sigmoid', name='output')(output_backbone)
    model = Model(inputs=[input_ct], outputs=[x])
    return model
    
def Den(den_en, img_size, dropout):
    input_ct = Input(img_size)
    if den_en == 'den121':       
        model_backbone = DenseNet121(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    if den_en == 'den169':       
        model_backbone = DenseNet169(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    
    if den_en == 'den201':
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])

        return model

# input is 6 channel, add 1*1 filter to covert it to 3 channel, then feed into den201,flag =0 used the imagNet as the pretrained weight, flag=1 used the pretrained weight based on
#DenseNet201 trained on OHTS dataset
def Den_double(w_path2,den_en, img_size, dropout,flag):
    input_ct = Input(img_size)
    if flag == 0 :        
        if den_en == 'den201':
            input_3 = conv2d_block(input_ct, num_filters=3, filter_size=(1,1), with_bn=True, activation='relu', name='con_in')
            model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=(224,224,3),pooling='avg')
            output_backbone = model_backbone(input_3)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input_ct], outputs=[x])
    else:        
        if den_en == 'den201':
            input_3 = conv2d_block(input_ct, num_filters=3, filter_size=(1,1), with_bn=True, activation='relu', name='con_in')                
            model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
            model_backbone.load_weights(w_path2)
            x = model_backbone(input_3)
            model = Model(inputs=[input_ct], outputs=[x])
        

    return model    
    

def Dennew(den_en, img_size, dropout):
    input_ct = Input(img_size)
    if den_en == 'den201':     
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        output_backbone = model_backbone(input_ct)
        output_backbone = AveragePooling2D(pool_size=(7, 7),name='globalaveragepooling')(output_backbone)
      #  output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        return model
#        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
#        intermediate_output = model_backbone.get_layer('bn').output
#        model_backbone = Model(model_backbone.input, [model_backbone.output, intermediate_output])
#        output_backbone,output_intermediate = model_backbone(input_ct)
#        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
#        model = Model(inputs=[input_ct], outputs=[x, output_intermediate])
#        model.layers[1].get_layer('relu').output
#        model.get_layer('densenet201').get_layer('relu').output
#        return model

# DenseNet without pretrain 
def Den1(den_en, img_size, dropout):
    input_ct = Input(img_size)
    model_backbone = DenseNet201(include_top=False,weights=None,input_shape=img_size,pooling='avg')
    output_backbone = model_backbone(input_ct)
    x = Dense(1, activation='sigmoid', name='output')(output_backbone)
    model = Model(inputs=[input_ct], outputs=[x])
    return model

def Em(model_en, img_size, dropout):
    input_ct = Input(img_size)
    if model_en == 'res50v2':       
        model_backbone = ResNet50V2(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    if model_en == 'res152':       
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    if model_en == 'den121':       
        model_backbone = DenseNet121(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    if model_en == 'den201':       
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model
    
#    if model_en == 'eNetB3':       
#        model_backbone = EfficientNetB3(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
#        output_backbone = model_backbone(input_ct)
#        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
#        model = Model(inputs=[input_ct], outputs=[x])
#        
#        return model
    
    if model_en == 'nasmobile':       
        model_backbone = NASNetMobile(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone = model_backbone(input_ct)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
        return model

def ensemble_res(res_en,img_size,model_input, dropout=False):
    ensemble = []
    for i in range(len(res_en)):
        modelTemp = Res(res_en[i],img_size,dropout=False)
        modelTemp.name1 = 'res_'+str(i)
        ensemble.append(modelTemp)
    
    yModels = [model(model_input) for model in ensemble]

    yAvg = average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns

def ensemble_model(model_en,img_size,model_input, dropout=False):
    ensemble = []
    for i in range(len(model_en)):
        modelTemp = Em(model_en[i],img_size,dropout=False)
        modelTemp.name1 = 'model_'+str(i)
        ensemble.append(modelTemp)
    
    yModels = [model(model_input) for model in ensemble]

    yAvg = average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns

def ensemble_res1(img_size,model_input, dropout=False):
    input_ct = Input(img_size)
    model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
    output_backbone1 = model_backbone(input_ct)
    model_backbone = ResNet101(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
    output_backbone2 = model_backbone(input_ct)
    model_backbone = ResNet50(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
    output_backbone3 = model_backbone(input_ct)
#    print('the shape of 1:', np.shape(output_backbone1))
#    print('the shape of 2:', np.shape(output_backbone2))
#    print('the shape of 3:', np.shape(output_backbone3))
    #output_backbone = np.concatenate((output_backbone1, output_backbone2,output_backbone3), axis=-1)
   # output_backbone = merge([output_backbone1, output_backbone2,output_backbone3], mode='concat', concat_axis=-1)

    output_backbone = concatenate([output_backbone1, output_backbone2,output_backbone3], axis=3)
    output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
    output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
    output_backbone = Flatten()(output_backbone)
    x = Dense(1, activation='sigmoid', name='output')(output_backbone)
    model = Model(inputs=[input_ct], outputs=[x])
        
    return model

def ensemble_resden(img_size,model_input, dropout=False, flag = 1):
    input_ct = Input(img_size)
    if flag == 1 :
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        output_backbone1 = model_backbone(input_ct)
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        output_backbone2 = model_backbone(input_ct)
    #    print('the shape of 1:', np.shape(output_backbone1))
    #    print('the shape of 2:', np.shape(output_backbone2))
    #    print('the shape of 3:', np.shape(output_backbone3))
        #output_backbone = np.concatenate((output_backbone1, output_backbone2,output_backbone3), axis=-1)
       # output_backbone = merge([output_backbone1, output_backbone2,output_backbone3], mode='concat', concat_axis=-1)
    
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
    elif flag == 0:
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone1 = model_backbone(input_ct)
        
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone2 = model_backbone(input_ct)
        
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=-1)
        
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
        
               
    return model
 
def ensemble_resden1(w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_ct = Input(img_size)
    if flag == 1 :
        model_backbone = Res(res_en='res152',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path1)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('resnet152').get_layer('conv5_block3_out').output)
        output_backbone1 = model_backbone(input_ct)
        model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
        output_backbone2 = model_backbone(input_ct)
    #    print('the shape of 1:', np.shape(output_backbone1))
    #    print('the shape of 2:', np.shape(output_backbone2))
    #    print('the shape of 3:', np.shape(output_backbone3))
        #output_backbone = np.concatenate((output_backbone1, output_backbone2,output_backbone3), axis=-1)
       # output_backbone = merge([output_backbone1, output_backbone2,output_backbone3], mode='concat', concat_axis=-1)
    
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
    elif flag == 0: # should be change in the future
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone1 = model_backbone(input_ct)
        
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone2 = model_backbone(input_ct)
        
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=-1)
        
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
    return model

def ensemble_resden_double(w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_ct = Input(img_size)
    if flag == 1 :
        model_backbone = Res1(res_en='res152',img_size=(224, 224, 6), dropout=False)
        model_backbone.load_weights(w_path1)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('resnet152').get_layer('conv5_block3_out').output)
        output_backbone1 = model_backbone(input_ct)
        model_backbone = Den1(den_en='den201',img_size=(224, 224, 6), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
        output_backbone2 = model_backbone(input_ct)
    #    print('the shape of 1:', np.shape(output_backbone1))
    #    print('the shape of 2:', np.shape(output_backbone2))
    #    print('the shape of 3:', np.shape(output_backbone3))
        #output_backbone = np.concatenate((output_backbone1, output_backbone2,output_backbone3), axis=-1)
       # output_backbone = merge([output_backbone1, output_backbone2,output_backbone3], mode='concat', concat_axis=-1)
    
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
    elif flag == 0: # should be change in the future
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone1 = model_backbone(input_ct)
        
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone2 = model_backbone(input_ct)
        
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=-1)
        
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
               
    return model

def ensemble_resden_siamese(model_path,w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_l = Input(img_size)
    input_r = Input(img_size)
    if flag == 1 :
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        
#         model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
#         model_backbone.load_weights(w_path2)
#         model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('avg_pool').output)
        
#         model_backbone = ensemble_resden1(w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
#         model_backbone.load_weights(model_path)
#         model_backbone = Model(inputs=model_backbone.input, outputs=model_backbone.layers[-2].output)
        # encode each of the two inputs into a vector with the convnet
        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        # merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

#         L1_distance = lambda x: tf.abs(x[0] - x[1])
#         both = merge([encoded_l,encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1, activation='sigmoid', name='output')(L1_distance)
        model = Model(inputs=[input_l, input_r], outputs=prediction)

    elif flag == 0: # use the the pretrained model based on ohts
        model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('avg_pool').output)
        
        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

#         L1_distance = lambda x: tf.abs(x[0] - x[1])
#         both = merge([encoded_l,encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1, activation='sigmoid', name='output')(L1_distance)
        model = Model(inputs=[input_l, input_r], outputs=prediction)
        
    return model

# siamese network, concatenate the output of encoder 
def ensemble_resden_siamese1(model_path,w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_l = Input(img_size)
    input_r = Input(img_size)
    if flag == 1 :        
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        
        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        
        output_backbone = concatenate([encoded_l, encoded_r], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_l, input_r], outputs=[x])
    return model

def ensemble_resden_siamese2(model_path,w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_l = Input(img_size)
    input_r = Input(img_size)
    if flag == 1 :
        model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
        
        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        
        output_backbone = concatenate([encoded_l, encoded_r], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_l, input_r], outputs=[x])
    return model

def siamse_opreation(x1,x2,output_n,pool_size,name1,name2):
    
    encoded_l = output_n(x1)
    encoded_r = output_n(x2)
        
    output_backbone = concatenate([encoded_l, encoded_r], axis=3)
    #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
    output_backbone = conv2d_block(output_backbone, num_filters=512, filter_size=(1,1), with_bn=True, activation='relu', name=name1)
    output_backbone = AveragePooling2D(pool_size= pool_size)(output_backbone)
    output_backbone = Flatten()(output_backbone)
    x = Dense(1, activation='sigmoid', name=name2)(output_backbone)
    
    return x
    

def ensemble_resden_siamese3(model_path,w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_l = Input(img_size)
    input_r = Input(img_size)
    if flag == 1 :
        model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)

#         output2 = model_backbone.get_layer('pool2_relu')
#        output3 = model_backbone.get_layer('pool3_relu')
#        output4 = model_backbone.get_layer('pool4_relu')
        output4 = Model(inputs=model_backbone.layers[0].output, outputs=model_backbone.get_layer('pool4_relu').output)
        
#         x2 = siamse_opreation(input_l,input_r,output_n = output2,pool_size=(56,46),name1='con_f2',name2='output2')
#        x3 = siamse_opreation(input_l,input_r,output_n = output3,pool_size=(28,28),name1='con_f3',name2='output3')
        x4 = siamse_opreation(input_l,input_r,output_n = output4,pool_size=(14,14),name1='con_f4',name2='output4')
        
        
        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        
        output_backbone = concatenate([encoded_l, encoded_r], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        
       # x = average([x4, x])
        x = average([x4, x, x, x, x])
        model = Model(inputs=[input_l, input_r], outputs=[x])
    return model


def siamse_opreation1(x1,x2,pool_size,name1,name2):
        
    output_backbone = concatenate([encoded_l, encoded_r], axis=3)
    #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
    output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name=name1)
    output_backbone = AveragePooling2D(pool_size= pool_size)(output_backbone)
    output_backbone = Flatten()(output_backbone)
    x = Dense(1, activation='sigmoid', name=name2)(output_backbone)

def ensemble_resden_siamese33(model_path,w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_l = Input(img_size)
    input_r = Input(img_size)
    if flag == 1 :
        model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('pool4_relu').output)
        model_backbone1 = Model(inputs=model_backbone.get_layer('densenet201').get_layer('pool4_conv').output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
#        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)

        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        
        x4 = siamse_opreation(encoded_l,encoded_r,pool_size=(14,14),name1='con_f4',name2='output4')
                
        encoded_l = model_backbone1(encoded_l)
        encoded_r = model_backbone1(encoded_r)

        
        output_backbone = concatenate([encoded_l, encoded_r], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        
        x = average([x4, x])
        model = Model(inputs=[input_l, input_r], outputs=[x])
    return model



# get two feature embeddings from two side out, then concatenate
def ensemble_resden_siamese4(model_path,w_path1,w_path2,img_size,model_input, dropout=False, flag = 1):
    input_l = Input(img_size)
    input_r = Input(img_size)
    if flag == 1: # use the the pretrained model based on ohts
        model_backbone = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
        model_backbone.load_weights(w_path2)
        model_backbone = Model(inputs=model_backbone.layers[1].layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('avg_pool').output)
        
        output4 = Model(inputs=model_backbone.layers[0].output, outputs=model_backbone.get_layer('pool4_relu').output)
        
        encoded_l4 = output4(input_l)
        encoded_r4 = output4(input_r)
        output_backbone_l4 = AveragePooling2D(pool_size= (14,14))(encoded_l4)
        output_backbone_l4 = Flatten()(output_backbone_l4)
        output_backbone_r4 = AveragePooling2D(pool_size= (14,14))(encoded_r4)
        output_backbone_r4 = Flatten()(output_backbone_r4)

        
        encoded_l = model_backbone(input_l)
        encoded_r = model_backbone(input_r)
        
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        
        L2_distance = L1_layer([output_backbone_l4, output_backbone_r4])
        
        L12_distance = concatenate([L1_distance, L2_distance], axis=-1)

#         L1_distance = lambda x: tf.abs(x[0] - x[1])
#         both = merge([encoded_l,encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1, activation='sigmoid', name='output')(L12_distance)
        model = Model(inputs=[input_l, input_r], outputs=prediction)

    return model

def ensemble_resden2(model_path,w_path1,w_path2,w_path11,w_path22,img_size,model_input, dropout=False, flag = 1):
    input_ct = Input(img_size)
    if flag == 1 : # covert 6 channels into 3 three channels as input,feed into the GluacomaNet pretrained on OHTS
        input_3 = conv2d_block(input_ct, num_filters=3, filter_size=(1,1), with_bn=True, activation='relu', name='con_in')
        model_backbone = ensemble_resden1(w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
        model_backbone.load_weights(model_path)
        x = model_backbone(input_3)
        
        model = Model(inputs=[input_ct], outputs=[x])
    elif flag == 0: # six channels version of GlaucomaNet, and Resnet152 and DenseNet201 pretrained on OHTS (6 channels)        
        res_en='res152'
        den_en='den201'
        model_backbone =  Res_double(w_path1,res_en, img_size, dropout,flag=1)
        model_backbone.load_weights(w_path11)
        
        
                # for Res_double and Den_double, it's still need to chosse the flag value to determine the pretrained weight,
        # flag = 0: used the pretrained weight based on ImageNet, flag =1: used the pretrained based on ohts
        
        # resnet152 flag = 1
        #model_backbone = model_backbone.layers[-1].get_layer('resnet152').get_layer('conv5_block3_out').output
        model_backbone3 = Model(inputs=model_backbone.input, outputs=model_backbone.get_layer('con_in').output)
        model_backbone33 = model_backbone33 = Model(inputs=model_backbone.layers[-1].get_layer('resnet152').input, outputs=model_backbone.layers[-1].get_layer('resnet152').get_layer('conv5_block3_out').output)
        output_backbone1 = model_backbone3(input_ct)
        output_backbone1 = model_backbone33(output_backbone1)
        
        
        # resnet 152 flag = 0
#         model_backbone3 = Model(inputs=model_backbone.input, outputs=model_backbone.get_layer('con_in').output)
#         model_backbone33 = Model(inputs=model_backbone.layers[-2].input, outputs=model_backbone.get_layer('resnet152').get_layer('conv5_block3_out').output)
#         output_backbone1 = model_backbone3(input_ct)
#         output_backbone1 = model_backbone33(output_backbone1)
        
#         model_backbone = Model(inputs=model_backbone.layers[0].output, outputs=model_backbone.get_layer('resnet152').get_layer('conv5_block3_out').output)
#         output_backbone1 = model_backbone(input_ct)


        model_backbone = Den_double(w_path2,den_en, img_size, dropout,flag=1)
        model_backbone.load_weights(w_path22)
        
        # densenet201 flag = 1, the double densenet pretrained on densenet based on ohts
        #model_backbone1.layers[-1].get_layer('densenet201').get_layer('relu').output
        model_backbone3 = Model(inputs=model_backbone.input, outputs=model_backbone.get_layer('con_in').output)
        model_backbone33 = Model(inputs=model_backbone.layers[-1].get_layer('densenet201').input, outputs=model_backbone.layers[-1].get_layer('densenet201').get_layer('relu').output)
        #model_backbone = Model(inputs=model_backbone.layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
        output_backbone2 = model_backbone3(input_ct)
        output_backbone2 = model_backbone33(output_backbone2)
        
        
        
        # densenet 201 flag = 0, the double densenet pretrained on densenet based on ImageNet
           
#         model_backbone3 = Model(inputs=model_backbone.input, outputs=model_backbone.get_layer('con_in').output)
#         model_backbone33 = Model(inputs=model_backbone.layers[-2].input, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
#         #model_backbone = Model(inputs=model_backbone.layers[0].output, outputs=model_backbone.get_layer('densenet201').get_layer('relu').output)
#         output_backbone2 = model_backbone3(input_ct)
#         output_backbone2 = model_backbone33(output_backbone2)
        
        
#         output_backbone2 = model_backbone(input_ct)
        #output_backbone = np.concatenate((output_backbone1, output_backbone2,output_backbone3), axis=-1)
       # output_backbone = merge([output_backbone1, output_backbone2,output_backbone3], mode='concat', concat_axis=-1)
        output_backbone = concatenate([output_backbone1, output_backbone2], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input_ct], outputs=[x])
    return model

def multiscale_Net(net,img_size, dropout=False, flag = 1):
    input1 = Input(img_size, name='input1')
    input2 = Input(img_size, name='input2')
    if flag == 0 :
        if net == 'res152': 
            model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone = concatenate([output_backbone1, output_backbone2], axis=-1)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2], outputs=[x])
        if net == 'den201': 
            model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone = concatenate([output_backbone1, output_backbone2], axis=-1)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2], outputs=[x])
    if flag == 1: 
        if net == 'res152': 
            model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone = concatenate([output_backbone1, output_backbone2], axis=3)
            #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
            output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
            output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
            output_backbone = Flatten()(output_backbone)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2], outputs=[x])
        if net == 'den201': 
            model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone = concatenate([output_backbone1, output_backbone2], axis=3)
            #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
            output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
            output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
            output_backbone = Flatten()(output_backbone)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2], outputs=[x])
       
    return model

def triplescale_Net(net,img_size, dropout=False, flag = 1):
    input1 = Input(img_size, name='input1')
    input2 = Input(img_size, name='input2')
    input3 = Input(img_size, name='input3')
    if flag == 0 :
        if net == 'res152': 
            model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone3 = model_backbone(input3)
            output_backbone = concatenate([output_backbone1, output_backbone2, output_backbone3], axis=-1)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2, input3], outputs=[x])
        if net == 'den201': 
            model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone3 = model_backbone(input3)
            output_backbone = concatenate([output_backbone1, output_backbone2, output_backbone3], axis=-1)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2, input3], outputs=[x])
    if flag == 1: 
        if net == 'res152': 
            model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone3 = model_backbone(input3)
            output_backbone = concatenate([output_backbone1, output_backbone2, output_backbone3], axis=-1)
            #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
            output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
            output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
            output_backbone = Flatten()(output_backbone)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2, input3], outputs=[x])
        if net == 'den201': 
            model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
            output_backbone1 = model_backbone(input1)
            output_backbone2 = model_backbone(input2)
            output_backbone3 = model_backbone(input3)
            output_backbone = concatenate([output_backbone1, output_backbone2, output_backbone3], axis=-1)
            #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
            output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
            output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
            output_backbone = Flatten()(output_backbone)
            x = Dense(1, activation='sigmoid', name='output')(output_backbone)
            model = Model(inputs=[input1,input2, input3], outputs=[x])
       
    return model

def Multiscale_multimodel(img_size, dropout=False, flag = 1):
    input1 = Input(img_size, name='input1')
    input2 = Input(img_size, name='input2')
    if flag == 1 :
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        output_backbone1 = model_backbone(input1)
        output_backbone2 = model_backbone(input2)
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)
        output_backbone3 = model_backbone(input1)
        output_backbone4 = model_backbone(input2)
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling=None)

    #    print('the shape of 1:', np.shape(output_backbone1))
    #    print('the shape of 2:', np.shape(output_backbone2))
    #    print('the shape of 3:', np.shape(output_backbone3))
        #output_backbone = np.concatenate((output_backbone1, output_backbone2,output_backbone3), axis=-1)
       # output_backbone = merge([output_backbone1, output_backbone2,output_backbone3], mode='concat', concat_axis=-1)
    
        output_backbone = concatenate([output_backbone1, output_backbone2, output_backbone3, output_backbone4], axis=3)
        #output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(3,3), with_bn=True, activation='relu', name='con_f')
        output_backbone = conv2d_block(output_backbone, num_filters=1024, filter_size=(1,1), with_bn=True, activation='relu', name='con_f')
        output_backbone = AveragePooling2D(pool_size=(7, 7))(output_backbone)
        output_backbone = Flatten()(output_backbone)
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input1,input2], outputs=[x])
    elif flag == 0:
        model_backbone = ResNet152(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone1 = model_backbone(input1)
        output_backbone2 = model_backbone(input2)
        
        model_backbone = DenseNet201(include_top=False,weights="imagenet",input_shape=img_size,pooling='avg')
        output_backbone3 = model_backbone(input1)
        output_backbone4 = model_backbone(input2)
        
        output_backbone = concatenate([output_backbone1, output_backbone2, output_backbone3, output_backbone4], axis=-1)
        
        x = Dense(1, activation='sigmoid', name='output')(output_backbone)
        model = Model(inputs=[input1,input2], outputs=[x])
        
               
    return model

def ensemble_vgg(img_size, model_input,dropout=False):
    ensemble = []
    for i in range(3):
        modelTemp = vgg16(img_size, scale=(0.5)**i,dropout=False)
        modelTemp.name1 = 'vgg_'+str((0.5)**i)
        ensemble.append(modelTemp)

    yModels = [model(model_input) for model in ensemble]

    yAvg = average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns


if __name__ == '__main__':
	model = vgg16(img_size=(224, 224, 3), scale=1)
	model.summary()

