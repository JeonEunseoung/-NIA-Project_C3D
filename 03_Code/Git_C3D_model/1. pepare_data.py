import shutil
import os
import random
import cv2
import json

def move_file(path1):
    action_list = os.listdir(path1+'/video')
    action_list.sort()
    if not os.path.exists(path1+'/test_video/'):
        os.mkdir(path1+'/test_video/')
    if not os.path.exists(path1+'/train_video/'):
        os.mkdir(path1+'/train_video/')
    for action in action_list:
        video_list = os.listdir(path1+'/video/'+action)
        video_list.sort()
        if not os.path.exists(path1+'/test_video/'+action):
            os.mkdir(path1+'/test_video/'+action)
        if not os.path.exists(path1+'/train_video/'+action):
            os.mkdir(path1+'/train_video/'+action)
        for video in video_list:
            cnt=random.randint(1, 10)
            if cnt>2:
                print(str(cnt)+'1')
                shutil.move(path1+'/video/'+action+'/'+video, path1+'/train_video/'+action+'/'+video)
            else:
                print(str(cnt)+'2')
                shutil.move(path1+'/video/'+action+'/'+video, path1+'/test_video/'+action+'/'+video)

def main_video2img(path1, path2):
    if not os.path.exists(path2+'/main_train_img'):
        os.mkdir(path2+'/main_train_img')
    if not os.path.exists(path2+'/main_test_img'):
        os.mkdir(path2+'/main_test_img')

    train_action_list = os.listdir(path1+'/train_video')
    test_action_list = os.listdir(path1+'/test_video')
    train_action_list.sort()
    test_action_list.sort()
    for train_action in train_action_list[2:4]:
        if not os.path.exists(path2+'/main_train_img/'+train_action):
            os.mkdir(path2+'/main_train_img/'+train_action)
            os.mkdir(path2+'/main_test_img/'+train_action)
        video_list = os.listdir(path1+'/train_video/'+train_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(path2+'/main_train_img/'+train_action+'/'+prefix):
                os.mkdir(path2+'/main_train_img/'+train_action+'/'+prefix)
            cap = cv2.VideoCapture(path1+'/train_video/'+train_action+'/'+video)
            print(path1+'/train_video/'+train_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    cv2.imwrite(path2+'/main_train_img/' + train_action + '/' + prefix + '/'+str(i+1)+'.jpg',frame)

    for test_action in test_action_list:          
        video_list = os.listdir(path1+'/test_video/'+test_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(path2+'/main_test_img/'+test_action+'/'+prefix):
                os.mkdir(path2+'/main_test_img/'+test_action+'/'+prefix)
            cap = cv2.VideoCapture(path1+'/test_video/'+test_action+'/'+video)
            print(path1+'/test_video/'+test_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    cv2.imwrite(path2+'/main_test_img/' + test_action + '/' + prefix + '/'+str(i+1)+'.jpg',frame)

def sub_video2img(path1, path2):
    if not os.path.exists(path2+'/sub_train_img'):
        os.mkdir(path2+'/sub_train_img')
    if not os.path.exists(path2+'/sub_test_img'):
        os.mkdir(path2+'/sub_test_img')

    train_action_list = os.listdir(path1+'/train_video')
    test_action_list = os.listdir(path1+'/test_video')
    train_action_list.sort()
    test_action_list.sort()

    for train_action in train_action_list[1:]:
        video_list = os.listdir(path1+'/train_video/'+train_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]

            json_data=[]
            print(path1+'/json/' + train_action +'/'+ prefix+'_blockinfo.json')
            with open(path1+'/json/' + train_action +'/'+ prefix+'_blockinfo.json', 'r') as f:
                json_data.append(json.load(f))
            for i in range (len(json_data[0]['block_information'])):
                subaction=json_data[0]['block_information'][i]['block_detail']
                if not os.path.exists(path2+'/sub_train_img/'+subaction):
                    os.mkdir(path2+'/sub_train_img/'+subaction)
                    os.mkdir(path2+'/sub_test_img/'+subaction)
                
            cap = cv2.VideoCapture(path1+'/train_video/'+train_action+'/'+video)
            print(path1+'/train_video/'+train_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    for j in range (len(json_data[0]['block_information'])):
                        if(int(json_data[0]['block_information'][j]['start_frame_index'])==i):
                            subaction = json_data[0]['block_information'][j]['block_detail']

                    save_name = path2+'/sub_train_img/'+subaction + '/' + prefix+'_'+subaction + '/'
                    if not os.path.exists(save_name):
                        os.mkdir(save_name)
                    cv2.imwrite(save_name+str(i+1)+'.jpg',frame)

    for test_action in test_action_list:
        video_list = os.listdir(path1+'/test_video/'+test_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]

            json_data=[]
            print(path1+'/json/' + test_action +'/'+ prefix+'_blockinfo.json')
            with open(path1+'/json/' + test_action +'/'+ prefix+'_blockinfo.json', 'r') as f:
                json_data.append(json.load(f))

            cap = cv2.VideoCapture(path1+'/test_video/'+test_action+'/'+video)
            print(path1+'/test_video/'+test_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    for j in range (len(json_data[0]['block_information'])):
                        if(int(json_data[0]['block_information'][j]['start_frame_index'])==i):
                            subaction = json_data[0]['block_information'][j]['block_detail']
                    
                    save_name = path2+'/sub_test_img/' +subaction+'/'+ prefix+'_'+subaction + '/'
                    if not os.path.exists(save_name):
                        os.mkdir(save_name)
                    cv2.imwrite(save_name+str(i+1)+'.jpg',frame)               

def sub_action_search(sub_action):
    if sub_action=='A01':
            sub_action_name='아이 혼자 있음.'
    elif sub_action=='A05':
        sub_action_name='유모차 방치.'
    elif sub_action=='A06':
        sub_action_name='유모차를 세게 밀침.'
    elif sub_action=='A07':
        sub_action_name='유모차를 발로 참.'
    elif sub_action=='A08':
        sub_action_name='유모차를 세게 올려내리침.'
    elif sub_action=='A14':
        sub_action_name='성인이 아이를 던짐.'
    elif sub_action=='A16':
        sub_action_name='유모차 엎어버림.'
    elif sub_action=='A17':
        sub_action_name='서성거림.'
    elif sub_action=='A18':
        sub_action_name='도어락 오픈 시도.'
    elif sub_action=='A19':
        sub_action_name='문을 발로 참.'
    elif sub_action=='A20':
        sub_action_name='문 안쪽을 보려고 시도.'
    elif sub_action=='A21':
        sub_action_name='문을 두드림.'
    elif sub_action=='A29':
        sub_action_name='공구로 때림.'
    elif sub_action=='A30':
        sub_action_name='택배 훔쳐가기.'
    elif sub_action=='A31':
        sub_action_name='차량 앞 서성거림.'
    elif sub_action=='N0':
        sub_action_name='정상행동.'
    elif sub_action=='N1':
        sub_action_name='정상행동.'
    elif sub_action=='SY01':
        sub_action_name='아이가 혼자 서성거림.'
    elif sub_action=='SY02':
        sub_action_name='아이가 혼자 서있음.'
    elif sub_action=='SY03':
        sub_action_name='아이가 혼자 앉아있음.'  
    elif sub_action=='SY07':
        sub_action_name='성인이 손을 들어올림.' 
    elif sub_action=='SY13':
        sub_action_name='아이가 유모차 탑승.'
    elif sub_action=='SY14':
        sub_action_name='아이가 돌아다님.'
    elif sub_action=='SY15':
        sub_action_name='사람이 문 주변에서 서성거림.'
    elif sub_action=='SY16':
        sub_action_name='사람이 문 주변에 서있음.'
    elif sub_action=='SY17':
        sub_action_name='사람이 문 주변에 앉아있음.'
    elif sub_action=='SY25':
        sub_action_name='사람이 택배앞 서성거림'
    elif sub_action=='SY26':
        sub_action_name='사람이 택배 주변에 서있음.'
    elif sub_action=='SY28':
        sub_action_name='사람이 차량앞 서성거림.'
    elif sub_action=='SY29':
        sub_action_name='사람이 차량 주변에 서있음.'
    elif sub_action=='SY30':
        sub_action_name='사람이 차량 주변에 앉아있음.'
    elif sub_action=='SY31':
        sub_action_name='사람이 문에 기대어 있음.'
    elif sub_action=='SY32':
        sub_action_name='사람이 벽(또는 기둥)에 기대있음.'
    return sub_action_name

def main_action_search(main_action):
    if main_action=='C011':
            main_action_name='아동학대(방임)'
    elif main_action=='C012':
        main_action_name='아동학대(신체학대)'
    elif main_action=='C021':
        main_action_name='주거침입(문 앞)'
    elif main_action=='C041':
        main_action_name='절도(문 앞)'
    elif main_action=='C042':
        main_action_name='절도(주차장)'
    return main_action_name

def index_gen(path2):
    if not os.path.exists(path2+'/input'):
        os.mkdir(path2+'/input')
    if not os.path.exists(path2+'/input/main'):
        os.mkdir(path2+'/input/main')
    if not os.path.exists(path2+'/input/sub'):
        os.mkdir(path2+'/input/sub')

    ftrain =  open(path2+'/input/main'+'/index.txt', "w")
    ftest =  open(path2+'/input/sub'+'/index.txt', "w")
        
    main_action_list = os.listdir(path2+'/main_train_img')
    sub_action_list = os.listdir(path2+'/sub_train_img')
    main_action_list.sort()
    sub_action_list.sort()

    for i, main_action in enumerate (main_action_list):
        main_action_name=main_action_search(main_action)
        ftrain.write(str(i)+': '+main_action+ ': '+main_action_name + '\n')

    for i, sub_action in enumerate (sub_action_list):
        sub_action_name=sub_action_search(sub_action)
        ftest.write(str(i)+': '+sub_action + ': '+sub_action_name+'\n')

def makefile(path2):    
    fm_train =  open(path2+'/input/main/trainfile.txt', "w")
    fm_test =  open(path2+'/input/main/testfile.txt', "w")

    fs_train =  open(path2+'/input/sub/trainfile.txt', "w")
    fs_test =  open(path2+'/input/sub/testfile.txt', "w")

    main_action_list = os.listdir(path2+'/main_train_img')
    sub_action_list = os.listdir(path2+'/sub_train_img')

    main_action_list.sort()
    sub_action_list.sort()

    for i, main_action in enumerate(main_action_list):
        train_img_list = os.listdir(path2+'/main_train_img/'+main_action)
        test_img_list = os.listdir(path2+'/main_test_img/'+main_action)
        train_img_list.sort()
        test_img_list.sort()
        for train_img in train_img_list:
            fm_train.write('main_train_img/'+main_action+'/'+train_img + " " + str(i) + "\n")
        for test_img in test_img_list:
            fm_test.write('main_test_img/'+main_action+'/'+test_img + " " + str(i) + "\n")
    
    for i, sub_action in enumerate(sub_action_list):
        train_img_list = os.listdir(path2+'/sub_train_img/'+sub_action)
        test_img_list = os.listdir(path2+'/sub_test_img/'+sub_action)
        train_img_list.sort()
        test_img_list.sort()
        for train_img in train_img_list:
            fs_train.write('sub_train_img/'+sub_action+'/'+train_img + " " + str(i) + "\n")
        for test_img in test_img_list:
            fs_test.write('sub_test_img/'+sub_action+'/'+test_img + " " + str(i) + "\n")
        
def file2list(path2):
    fmr_train = open(path2+'/input/main/'+'trainfile.txt',mode='r')
    fmr_test  = open(path2+'/input/main/'+'testfile.txt',mode='r')
    fsr_train = open(path2+'/input/sub/'+'trainfile.txt',mode='r')
    fsr_test  = open(path2+'/input/sub/'+'testfile.txt',mode='r')

    main_train_list = fmr_train.readlines()
    main_test_list = fmr_test.readlines()
    sub_train_list = fsr_train.readlines()
    sub_test_list = fsr_test.readlines()

    fmw_train = open(path2+'/input/main/'+'train_list.txt', 'w')
    fmw_test = open(path2+'/input/main/'+'test_list.txt', 'w')
    fsw_train = open(path2+'/input/sub/'+'train_list.txt', 'w')
    fsw_test = open(path2+'/input/sub/'+'test_list.txt', 'w')

    clip_length = 16

    for line in main_train_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fmw_train.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])

    for line in main_test_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fmw_test.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])

    for line in sub_train_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fsw_train.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])

    for line in sub_test_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fsw_test.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])

def main():
    #video_path, data_path 만 수정
    video_path= input("영상이 저장되어있는 경로를 입력하세요: ") #영상이 저장되어있는 경로
    data_path = input("학습데이터가 저장될 경로를 입력하세요: ") #학습데이터가 저장될 경로

    if not os.path.exists(data_path):
            os.mkdir(data_path)

    move_file(video_path)
    main_video2img(video_path,data_path)
    sub_video2img(video_path,data_path)
    index_gen(data_path)
    makefile(data_path)
    file2list(data_path)

if __name__ == '__main__':
     main()