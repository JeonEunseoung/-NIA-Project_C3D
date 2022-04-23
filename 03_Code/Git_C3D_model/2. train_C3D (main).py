import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  
# 세션 사용시.
# session = tf.Session(config=config)
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
import numpy as np
import random
import cv2
import random
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))

def onetenth_4_8_12(lr):
    steps = [4, 8,12]
    lrs = [lr, lr/10, lr/100,lr/1000]
    return Step(steps, lrs)
    
def c3d_model():
    input_shape = (112,112,16,3)
    weight_decay = 0.005
    nb_classes = 101

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_acc.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()
  
def process_batch(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        print(path)
        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            for j in range(16):
                img = imgs[symbol + j]
             
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                if is_flip == 1:
                    image = cv2.flip(image, 1)
                batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
            labels[i] = label
    return batch, labels

def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs

def generator_train_batch(train_txt,batch_size,num_classes,img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            yield x, y

def generator_val_batch(val_txt,batch_size,num_classes,img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y_labels = process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y

def main():
    data_path = input("학습데이터가 저장되어있는 경로를 입력하세요: ")
    #학습데이터가 저장되어있는 경로로 설정

    field='main' 

    if not os.path.exists(data_path+field+'_result'):
        os.mkdir(data_path+field+'_result')

    train_list=data_path+'input/'+field+'/train_list.txt'
    test_list=data_path+'input/'+field+'/test_list.txt'

    ftrain = open(train_list, 'r')
    lines = ftrain.readlines()
    ftrain.close()
    train_samples = len(lines)

    ftest = open(test_list, 'r')
    lines = ftest.readlines()
    ftest.close()
    val_samples = len(lines)

    num_classes = 101
    batch_size = 16
    epochs = 10

    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    model.summary()
    history = model.fit_generator(generator_train_batch(train_list, batch_size, num_classes,data_path),
                                steps_per_epoch=train_samples // batch_size, epochs=epochs, callbacks=[onetenth_4_8_12(lr)],
                                validation_data=generator_val_batch(test_list, batch_size,num_classes,data_path),
                                validation_steps=val_samples // batch_size, verbose=1)

    if not os.path.exists(data_path+field+'_result/'+'/epoch'+str(epochs)+'/'):
        os.mkdir(data_path+field+'_result/'+'/epoch'+str(epochs)+'/')
    plot_history(history, data_path+field+'_result/'+'/epoch'+str(epochs)+'/')
    save_history(history, data_path+field+'_result/'+'/epoch'+str(epochs)+'/')
    model.save_weights(data_path+field+'_result/'+'/epoch'+str(epochs)+'/weights_c3d.h5')

if __name__ == '__main__':
    main()