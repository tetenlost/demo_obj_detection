import os
import shutil as sh
from sklearn.model_selection import train_test_split
import json
import cv2
from tqdm import tqdm
def train_test_val_split(img_paths,ratio_train=0.8,ratio_test=0.0,ratio_val=0.2):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train+ratio_test+ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    ratio=ratio_val/(1-ratio_train)
    val_img, test_img  =train_test_split(middle_img,test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def yolo2coco(root_path,foruse,save_path,random_split=False):
    originLabelsDir = os.path.join(root_path, foruse+'lab').replace('\\', '/')                                        
    originImagesDir = os.path.join(root_path, foruse+'2017').replace('\\', '/')
    with open(os.path.join(root_path, 'classes.txt').replace('\\', '/')) as f:
        classes = f.read().strip().split()
    # images dir name
    indexes = os.listdir(originImagesDir)

    if random_split:
        # 用于保存所有数据的图片信息和标注信息
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        # 建立类别标签和数字id的对应关系, 类别id从0开始。
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        train_img, val_img, test_img = train_test_val_split(indexes,0.8,0.19,0.01)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('testimg','txt').replace('.jpg','.txt').replace('.png','.txt')
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(root_path, foruse+'2017\\').replace('\\', '/') + index)
        height, width, _ = im.shape
        if random_split:
            # 切换dataset的引用对象，从而划分数据集
                if index in train_img:
                    dataset = train_dataset
                elif index in val_img:
                    dataset = val_dataset
                elif index in test_img:
                    dataset = test_dataset
        # 添加图像的信息
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile).replace('\\', '/')):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(originLabelsDir, txtFile).replace('\\', '/'), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                print(label)
                if len(label)<1:
                    continue
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'annotations').replace('\\', '/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if random_split:
        for phase in ['train','val','test']:
            json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase)).replace('\\', '/')
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, 'annotations/{}'.format(save_path)).replace('\\', '/')
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

if True:
    os.mkdir('train2017\\')
    os.mkdir('trainlab\\')
    os.mkdir('vallab\\')
    os.mkdir('val2017\\')
path = "data"
pathlist = os.listdir(path)
newlist = set([x.split('.')[0] for x in pathlist])
train_X, test_X = train_test_split(list(newlist), test_size = 0.2)
print(len(train_X))
print(len(test_X))
for a in train_X:
    jpgpath = path+'\\'+a+'.jpg'
    pngpath = path+'\\'+a+'.png'
    labelpath = path+'\\'+a+'.txt'
    os.system('copy '+jpgpath+' train2017')
    os.system('copy '+pngpath+' train2017')
    os.system('copy '+labelpath+' trainlab')
print("train finish")
for b in test_X:
    jpgpath = path+'\\'+b+'.jpg'
    pngpath = path+'\\'+b+'.png'
    labelpath = path+'\\'+b+'.txt'
    os.system('copy '+jpgpath+' val2017')
    os.system('copy '+pngpath+' val2017')
    os.system('copy '+labelpath+' vallab')
print("test finish")
input("press ENTER")
yolo2coco('','train','./instances_train2017.json')
yolo2coco('','val','./instances_test2017.json')
