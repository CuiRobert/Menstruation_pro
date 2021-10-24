from torchvision import models
from torch import nn, optim
from torchvision import transforms as T
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset,random_split
import time
import torch
import cnn_finetune
from data_pre import MyDataSet,dataSplit
import os
import argparse
import numpy as np

import logging
import pandas as pd
from tqdm import tqdm
torch.manual_seed(45)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda:0'
# device = 'cpu'

parser = argparse.ArgumentParser(description='cnn_finetune 3 classify')

parser.add_argument('--test-batch-size',type=int,default=32,metavar='N',
                    help='input batch size for testing(default:64)')

parser.add_argument('--no-cuda',action='store_true',default=False,
                    help='disables CUDA training')
parser.add_argument('--seed',type=int, default=1, metavar='S',
                    help='random seed (default:1)')

parser.add_argument('--model-name',type=str, default='densenet201',metavar='M',
                    help='model name (default: densenet201)')


parser.add_argument('--label-dir',type=str,default='../Data/所有图像对应表_有标签.xlsx',metavar='Label',
                    help='label dirct')

parser.add_argument('--testimg-dir',type=str,default='../Data/test_images',metavar='IMG',
                    help='test image dirct')

parser.add_argument('--testlabel-dir1',type=str,default='../Data/测试集——FZ——纵切一一对应2.xlsx',metavar='Label1',
                    help='section ZQ test label dirct')

parser.add_argument('--testlabel-dir2',type=str,default='../Data/测试集——FZ——其他切面一一对应2.xlsx',metavar='Label2',
                    help='section QT test label dirct')

parser.add_argument('--model-label',type=str,default='',metavar='MLabel',
                    help='model label ')

parser.add_argument('--model-ZQ',type=str,default='../weight_models_ZQ/densenet201/训练集——FZ——纵切（全部数据）无病理数据2_all_images_densenet201_classfy2_epoch40.pkl',metavar='Model_name_ZQ',
                    help='model label ZQ ')
parser.add_argument('--model-QT',type=str,default='../weight_models_QT/densenet201/训练集——FZ——其他切面（全部数据）无病理数据2_all_images_densenet201_classfy2_epoch40.pkl',metavar='Model_name_QT',
                    help='model label QT')

'''需要将trainlabel-dir 转化为 testlabel-dir'''


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()





def test(model, test_loader,classes):
    model.eval()
    correct = 0
    class_correcet = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    img_list,label_list,pred_list = [], [], []
    img2pro = {}
    with torch.no_grad():
        for data, target, imgname in test_loader:
            data, target = data.to(device),target.to(device) #batch * 0
            output = model(data) # batch * 4
            pred = output.data.max(1,keepdim=True)[1] #batch * 1
            # print(imgname)
            # print(output.data)
            for i,j in zip(imgname,output.data):
                img2pro[i] = j.cpu().tolist()

            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
            # print(imgname,pred.data.view_as(target))
            img_list.extend(imgname)
            label_list.extend(target.tolist())
            pred_list.extend(pred.data.view_as(target).cpu().tolist())
            c = target.eq(pred.data.view_as(target))
            #print(res)4
            for i in range(len(target)):
                label = target[i]
                class_correcet[label] += c[i].item()
                class_total[label] += 1
    for i in range(len(classes)):
        try :
            acc = class_correcet[i]/class_total[i]
            print('\tclassID:%d\tacc:%.2f%%\t %d/%d'%(i,100*acc,class_correcet[i],class_total[i]))
        except:
            acc = 0

    ''''''
    c = {'newname':img_list,'truelabel':label_list,'predlabel':pred_list}
    # df = pd.DataFrame(c)
    # df.to_excel(save_dir+'/' + args.model_label+ save_label + args.label_dir.replace('../Data/',''))

    print('Test set:Accuracy:{}/{} ({:.2f}%)\n'.format(correct,len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    print('安人分')
    df_org = pd.read_excel('../Data/测试集所有数据.xlsx')
    df_dict3 = pd.DataFrame.from_dict(img2pro,orient='index',columns=['predprob_0','predprob_1'])
    # print(df_dict3)
    df_dict3 = df_dict3.reset_index().rename(columns = {'index':'newname'})
    df_dict3 = pd.merge(df_dict3,df_org)
    pred_max_prob(df_dict3,keyword='增')

    return img2pro


def pred_max_prob(df_result, keyword):
    df_result = df_result

    case2img = {}  # 存储的是case_id :[图片编号1,图片编号2，图片编号3]
    for i, row in df_result.iterrows():
        case_id = row['orgname'].split('/')[-2]
        if case_id in case2img:
            case2img[case_id].append(row['newname'])
        else:
            case2img[case_id] = [row['newname']]

    case2prob = {}  # 存储case,每张图片经softmax函数的值 case_id:[[img1_prob,img2_prob],[img1_prob1,img1_prob2]]
    for key, value in case2img.items():
        list_prob_temp = [[], []]
        for i in value:
            list_prob_temp[0].append(df_result[df_result['newname'] == i]['predprob_0'].values[0])
            list_prob_temp[1].append(df_result[df_result['newname'] == i]['predprob_1'].values[0])
        case2prob[key] = list_prob_temp

    case2result = {}  # 存储 case_id:预测结果（0 or 1）
    for i, value in case2prob.items():
        a = np.where(value == np.max(value))
        case2result[i] = a[0].item()

    correct_JJ = 0
    correst_QT = 0
    count_label_1 = 0
    for _, value in case2result.items():
        if keyword in _:
            count_label_1 += 1
        if keyword in _ and value == 1:
            correct_JJ += 1
        if keyword not in _ and value == 0:
            correst_QT += 1

    print('总的准确率：%.4f\t (%d + %d)\t/%d '%((correct_JJ + correst_QT) / len(case2result), correct_JJ, correst_QT, len(case2result)))
    print('标签为1的准确率(绝经期或者增生期)：%.4f\t(%d/%d)'%(correct_JJ / count_label_1,correct_JJ,count_label_1))
    print('标签为0的准确率：%.4f\t(%d/%d)'%((correst_QT / (len(case2result) - count_label_1)),correst_QT,(len(case2result) - count_label_1)))



def main():
    '''Main function to  run code in this script'''

    model_name = args.model_name
    test_img_dir = args.testimg_dir
    test_label_dir1 = args.testlabel_dir1
    test_label_dir2 = args.testlabel_dir2

    """分类类别数"""
    # classes = ('fenmiqi', 'zengshengzaoqi', 'juejingqi')
    classes = ('qita','juejingqi')



    # print(model)
    transform = T.Compose([
        T.Resize([300,400]),
        T.ToTensor(),
        T.Normalize(#mean = [0.282, 0.280, 0.282,],
                    #std  = [0.175, 0.173, 0.174]),
                    # mean = [0.280, 0.278, 0.28, ],
                    # std = [0.175, 0.173, 0.174]
        mean =  [0.250, 0.251, 0.253],
        std  =  [0.155, 0.155, 0.156])
                        ])
    ZQ_set = MyDataSet(test_img_dir,test_label_dir1,col_name='label', transform=transform)
    QT_set = MyDataSet(test_img_dir,test_label_dir2,col_name='label', transform=transform)


    ZQ_test_loader = DataLoader(ZQ_set, batch_size=args.test_batch_size, shuffle=False,num_workers=4,pin_memory=True)
    QT_test_loader  = DataLoader(QT_set, batch_size=args.test_batch_size,shuffle=False,num_workers=4,pin_memory=True)

    model = cnn_finetune.make_model(
        model_name,
        pretrained = False,
        num_classes = len(classes),
        dropout_p =  args.dropout_p
    )
    model.load_state_dict(torch.load(args.model_ZQ))
    model = model.to(device)

    ZQ_dict = test(model, ZQ_test_loader,classes)
    print('\n纵切面')
    print(ZQ_dict)

    model.load_state_dict(torch.load(args.model_QT))
    print('\n其他切面')
    QT_dict = test(model, QT_test_loader,classes)
    print(QT_dict)


    dict3 = dict(ZQ_dict, **QT_dict)
    df_org = pd.read_excel('../Data/测试集所有数据.xlsx')
    df_dict3 = pd.DataFrame.from_dict(dict3,orient='index',columns=['predprob_0','predprob_1'])
    # print(df_dict3)
    df_dict3 = df_dict3.reset_index().rename(columns = {'index':'newname'})
    df_dict3 = pd.merge(df_dict3,df_org,on=['对应编号'],how='left')
    df_dict3.to_excel('../result/测试集概率值.xlsx',index=False)
    print('\n合并纵横的结果')
    # pred_max_prob(df_dict3,keyword='增')
    # df_dict3 = pd.merge(dict3,df_org)

    # test(model,QT_test_loader, classes = classes)
    # test(model,QT_test_loader,classes = classes)
    # save(model,epoch,model_name,classes)


if __name__ == '__main__':

    main()