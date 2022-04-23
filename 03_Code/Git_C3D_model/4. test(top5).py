# coding=utf8
import warnings
from keras.optimizers import SGD
import numpy as np
import cv2
import os
import json
from datetime import datetime
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
from PIL import Image,ImageDraw,ImageFont

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  

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

def main():
    video_path = input("영상이 저장되어있는 경로를 입력하세요: ")
    #영상이 저장되어 있는 경로
    test_video = input("테스트 할 영상을 입력하세요: ")
    #테스트 할 영상
    data_path = input("학습 데이터가 저장되어 있는 경로를 입력하세요: ")
    #학습 데이터가 저장되어 있는 경로

    font = ImageFont.truetype("./GULIM.TTC", 40)

    test_date=str(datetime.today().month) +'.'+ str(datetime.today().day)  
    main_action=os.path.dirname(test_video)
    video_name=os.path.basename(test_video) 

    if not os.path.exists(data_path+'/test_'+test_date):
        os.mkdir(data_path+'/test_'+test_date)
    if not os.path.exists(data_path+'/test_'+test_date+'/'+main_action):
        os.mkdir(data_path+'/test_'+test_date+'/'+main_action)
    if not os.path.exists(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]):
        os.mkdir(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0])
    
    fm=open(data_path+'/input/main/index.txt', 'r')
    main_names = fm.readlines()
    fs=open(data_path+'/input/sub/index.txt', 'r', encoding='UTF-8')
    sub_names = fs.readlines()
    fw = open(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]+'_test.txt', 'w')
    json_data=[]
    json_name = video_path+'/json/'+main_action+'/'+video_name.split('.')[0]+'_blockinfo.json'
    fjson = open(json_name, 'r') 
    json_data.append(json.load(fjson))
    out_data=json_data[0]['block_information'][0]['block_detail']

    # init model
    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()      

    cap = cv2.VideoCapture(video_path+'/test_video/'+test_video)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(3)) 
    height = int(cap.get(4)) 
    fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]+'_test.mp4', fcc, 20, (width*2, height))
    clip = []

    main_count_list = [0 for i in range(len(main_names))]
    sub_count_list = [[0 for i in range(len(sub_names))] for j in range(len(json_data[0]['block_information']))]
    scene=0

    for i in range(fps):
        ret, frame = cap.read()
        black_img=np.zeros((height,width,3),dtype=np.uint8)
        img_pil = Image.fromarray(black_img)
        draw = ImageDraw.Draw(img_pil)
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))

                model.load_weights(data_path+'/main_result/epoch10/weights_c3d.h5', by_name=True)
                pred_main = model.predict(inputs)
                main_label = np.argmax(pred_main[0])
                main_count_list[main_label]=main_count_list[main_label]+1
                
                model.load_weights(data_path+'/sub_result/epoch10/weights_c3d.h5', by_name=True)
                pred_sub = model.predict(inputs)
                sub_label = np.argmax(pred_sub[0])
                top_inds = pred_sub[0].argsort()[::-1][:5]
                for j in range (len(json_data[0]['block_information'])):
                        if int(json_data[0]['block_information'][j]['start_frame_index'])==i: 
                            scene=scene+1
                sub_count_list[scene][sub_label]=sub_count_list[scene][sub_label]+1

                print(str(i+1)+'/'+str(fps))
                fw.write('#'+str(i+1)+'\n')
                draw.text((20, 20), '검출된 범죄유형', font=font, fill=(255,255,255,0))
                draw.text((20, 70), main_names[main_label].split(':')[2].strip()+'('+main_names[main_label].split(':')[1].strip()+')', font=font, fill=(255,255,255,0))
                draw.text((20, 120), "정확도: %.2f" % (pred_main[0][main_label]*100)+'%',font=font, fill=(255,255,255,0))
                fw.write('검출된 범죄유형'+'\n')
                fw.write('\t'+main_names[main_label].split(':')[2].strip()+'('+main_names[main_label].split(':')[1].strip()+')'+'\n')
                fw.write('\t'+"정확도: %.2f" % (pred_main[0][main_label]*100)+'%'+'\n')
                
                for j in range(len(main_names)):  
                    if main_names[j].split(':')[1].strip()==main_action:
                        draw.text((20, 220), "정답 범죄유형", font=font, fill=(255,255,255,0))
                        draw.text((20, 270),main_names[j].split(':')[2].strip()+'('+main_names[j].split(':')[1].strip()+')', font=font, fill=(255,255,255,0))
                        fw.write("정답 범죄유형"+'\n')
                        fw.write('\t'+main_names[j].split(':')[2].strip()+'('+main_names[j].split(':')[1].strip()+')'+'\n')
                
                draw.text((20, 370), '검출된 세부 유형', font=font, fill=(255,255,255,0))
                fw.write('검출된 세부 유형'+'\n')
                for s,k in enumerate(top_inds):
                    draw.text((20, 420+100*s), str(s+1)+'위) '+ sub_names[k].split(':')[2].strip()+'('+sub_names[k].split(':')[1].strip()+')', font=font, fill=(255,255,255,0))           
                    draw.text((20, 470+100*s),"정확도: %.2f" % (pred_sub[0][k]*100)+'%' , font=font, fill=(255,255,255,0))  
                    fw.write('\t'+str(s+1)+'위) '+ sub_names[k].split(':')[2].strip()+'('+sub_names[k].split(':')[1].strip()+')'+'\n')   
                    fw.write('\t'+"정확도: %.2f" % (pred_sub[0][k]*100)+'%'+'\n')   

                for j in range (len(json_data[0]['block_information'])):
                    if int(json_data[0]['block_information'][j]['start_frame_index'])==i:
                        out_data=json_data[0]['block_information'][j]['block_detail']  

                for j in range(len(sub_names)):  
                    if sub_names[j].split(':')[1].strip()==out_data:
                        draw.text((20, 970),"정답 세부 유형", font=font, fill=(255,255,255,0))
                        draw.text((20, 1020),sub_names[j].split(':')[2].strip()+'('+sub_names[j].split(':')[1].strip()+')', font=font, fill=(255,255,255,0))
                        fw.write("정답 세부 유형"+'\n')
                        fw.write('\t'+sub_names[j].split(':')[2].strip()+'('+sub_names[j].split(':')[1].strip()+')'+'\n\n')
                  
                black_img = np.array(img_pil)    
                clip.pop(0)
                
            add_frame_img = cv2.hconcat((frame, black_img))
            cv2.imwrite(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]+'/'+str(i+1)+'.jpg',add_frame_img)    
            out.write(add_frame_img)        
          
    ftw = open(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]+'_total.txt', 'w')
    ftw.write(video_name+'\n')

    for i in range(len(main_names)):
        if main_action==main_names[i].split(':')[1].strip():
            ftw.write(main_names[i].split(':')[2].strip())

    ftw.write(' 영상 '+str(fps-15)+' 프레임 중 ')
    main_mode_label = np.argmax(main_count_list)    
    ftw.write(main_names[main_mode_label].split(':')[2].strip()+" 검출 "+str(main_count_list[main_mode_label])+" 프레임 ")
    main_frame_prod=main_count_list[main_mode_label]/(fps-15)*100

    ftw.write(str(int(main_frame_prod))+'%\n')
    for corr_main_label in range(len(main_names)):
        if main_action==main_names[corr_main_label].split(':')[1].strip()!=main_names[main_mode_label].split(':')[1].strip():
            main_frame_prod=main_count_list[corr_main_label]/(fps-15)*100
            ftw.write('\t\t\t\t\t\t\t'+main_names[corr_main_label].split(':')[2].strip()+" 검출 "+str(main_count_list[corr_main_label])+" 프레임 "+str(int(main_frame_prod))+'%\n')

    for j in range (len(json_data[0]['block_information'])):
        sub_mode_label = np.argmax(sub_count_list[j])
        ftw.write('\t\t'+json_data[0]['block_information'][j]['block_detail']+' 파트 ')

        correct_frame=int(json_data[0]['block_information'][j]['end_frame_index'])-int(json_data[0]['block_information'][j]['start_frame_index'])+1
        if j==0:
            correct_frame=correct_frame-15
        ftw.write(str(correct_frame)+' 프레임 중 ')
        
        sub_frame_prod=sub_count_list[j][sub_mode_label]/correct_frame*100
        ftw.write(sub_names[sub_mode_label].split(':')[1].strip()+" 검출 "+str(sub_count_list[j][sub_mode_label])+" 프레임 "+str(int(sub_frame_prod))+'%\n')

        for corr_sub_label in range(len(sub_names)):
            if json_data[0]['block_information'][j]['block_detail']==sub_names[corr_sub_label].split(':')[1].strip()!=sub_names[sub_mode_label].split(':')[1].strip():
                sub_frame_prod=sub_count_list[j][corr_sub_label]/correct_frame*100
                ftw.write('\t\t\t\t\t\t\t\t'+sub_names[corr_sub_label].split(':')[1].strip()+" 검출 "+str(sub_count_list[j][corr_sub_label])+" 프레임 "+str(int(sub_frame_prod))+'%\n')
    
    ftw.write('\n\n')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
     main()