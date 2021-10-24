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
from imgaug import parameters as iap
from imgaug import augmenters as iaa
import logging
import pandas as pd
from tqdm import tqdm
torch.manual_seed(45)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda:0'

parser = argparse.ArgumentParser(description='cnn_finetune 3 classify')
parser.add_argument('--batch-size',type=int,default=22,metavar='N',
                    help='input batch size for training (default:32)')
parser.add_argument('--test-batch-size',type=int,default=32,metavar='N',
                    help='input batch size for testing(default:64)')
parser.add_argument('--epochs',type=int, default=40,metavar='N',
                    help='number of epochs to train(default:100)')
parser.add_argument('--lr',type=float,default=2e-4,metavar='LR', #0.0002
                    help='learning rate(default:0.001)')
parser.add_argument('--no-cuda',action='store_true',default=False,
                    help='disables CUDA training')
parser.add_argument('--seed',type=int, default=1, metavar='S',
                    help='random seed (default:1)')
parser.add_argument('--log-interval',type=int,default=20,metavar='N',
                    help='How many batches to wait before logging training status')
parser.add_argument('--model-name',type=str, default='densenet201',metavar='M',
                    help='model name (default: resnet50)')
parser.add_argument('--dropout-p',type=float,default=0.5, metavar='D',
                    help='Dropout probability (default:0.2)')
parser.add_argument('--image-dir',type=str,default='../Data/all_images',metavar='IMG',
                    help='data dirct')
parser.add_argument('--label-dir',type=str,default='../Data/所有图像对应表_有标签.xlsx',metavar='Label',
                    help='label dirct')
parser.add_argument('--train-dir',type=str,default='../Data/all_images',metavar='IMG',
                    help='train images dirct')
parser.add_argument('--trainlabel-dir',type=str,default='../Data/训练集——FZ——纵切（全部数据）.xlsx',metavar='Label',
                    help='train data label dirct')
parser.add_argument('--test-dir',type=str,default='../Data/test_images',metavar='IMG',
                    help='test image dirct')
parser.add_argument('--testlabel-dir',type=str,default='../Data/测试集——FZ——纵切(一一对应).xlsx',metavar='Label',
                    help='test label dirct')
parser.add_argument('--model-label',type=str,default='Adam',metavar='MLabel',
                    help='model label ')


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

save_dir = os.path.join('../weight_models_ZQ/',args.model_name,)
save_label = args.trainlabel_dir.replace('../Data/','').replace('.xlsx','')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def data_enhance(X):
    X1 = []
    for i in range(X.size()[0]):
        input = X[i:i+1]
        input = np.reshape(input.cuda(),(input.shape[1],input.shape[2],input.shape[3])) #将数据转换成（shape[1]*shape[2]*shape[3）
        input = np.transpose(input.cuda(),(1,2,0)) #转换对应的位置，将RGB转换为BGR
        a = np.random.choice([0,90,180,270])
        x = iaa.Affine(rotate=a).augment_image(np.array(input))
        x = iaa.Fliplr(0.5).augment_image(x)
        x = iaa.Flipud(0.5).augment_image(x)
        x = np.transpose(x,(2,0,1))
        X1.append(x[np.newaxis,:,:,:]) #在x最外层添加了一个维度
    X = np.concatenate(X1,axis=0)   #将数组拼接在一起
    """添加噪声"""
    # noise = np.random.normal(0, 0.1, X.shape)
    # X += noise

    return X


def get_logger(filename, verbosity=1, name=None):
    if not os.path.isdir('../weight_models_ZQ'):
        os.makedirs('../weight_models_ZQ')
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
                                 "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
                )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

logger = get_logger(save_dir+'/{}_{}_{}.log'.format(args.model_label,args.trainlabel_dir.replace('../Data/','').replace('.xlsx',''),  save_label))

def train(model, epoch, optimizer,train_loader, criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2]).to(device))):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target, imgname) in enumerate(train_loader):
        # data = torch.from_numpy(data_enhance(data.to(device)))
        data, target = data.to(device),target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        # print(batch_idx)
        # print(len(data))
        # print(len(train_loader.dataset))
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
    return  total_loss / total_size


def test(model, test_loader,classes, criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2]).to(device))):
    model.eval()
    test_loss = 0
    correct = 0
    class_correcet = list(0. for i in range(len(classes)))
    class_total= list(0. for i in range(len(classes)))
    img_list,label_list,pred_list = [], [],[]
    with torch.no_grad():
        for data, target, imgname in test_loader:
            data, target = data.to(device),target.to(device) #batch * 0
            output = model(data) # batch * 4
            test_loss += criterion(output, target).item()
            pred = output.data.max(1,keepdim=True)[1] #batch * 1
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
            # print(imgname,pred.data.view_as(target))
            img_list.extend(imgname)
            label_list.extend(target.tolist())
            pred_list.extend(pred.data.view_as(target).cpu().tolist())
            c = target.eq(pred.data.view_as(target))
            #print(res)
            for i in range(len(target)):
                label = target[i]
                class_correcet[label] += c[i].item()
                class_total[label] += 1
    for i in range(len(classes)):
        try :
            acc = class_correcet[i]/class_total[i]
            logger.info('\tclassID:%d\tacc:%2d%%\t %d/%d'%(i,100*acc,class_correcet[i],class_total[i]))
        except:
            acc = 0

    ''''''
    c = {'newname':img_list,'truelabel':label_list,'predlabel':pred_list}
    df = pd.DataFrame(c)
    df.to_excel(save_dir+'/' + args.model_label+ save_label + args.trainlabel_dir.replace('../Data/',''))

    test_loss /= len(test_loader.dataset)
    logger.info('Test set:Average loss:{:.4f},Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss, correct,len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def save(model,epoch, model_name,classes,data_set=args.image_dir):
    torch.save(model.state_dict(), save_dir + '/{}_{}_{}_classfy{}_epoch{}.pkl'.format(save_label,data_set.replace('../Data/',''),model_name, len(classes), epoch))


def main():
    '''Main function to  run code in this script'''

    model_name = args.model_name
    img_dir = args.image_dir
    label_dir = args.label_dir

    train_img_dir = args.train_dir
    train_label_dir = args.trainlabel_dir

    test_img_dir = args.test_dir
    test_label_dir = args.testlabel_dir



    print('保存路径：', save_dir)
    """分类类别数"""
    # classes = ('fenmiqi', 'zengshengzaoqi', 'juejingqi')
    classes = ('qita','juejingqi')

    model = cnn_finetune.make_model(
        model_name,
        pretrained = True,
        num_classes = len(classes),
        dropout_p =  args.dropout_p
    )
    model = model.to(device)
    print(model)
    transform = T.Compose([
        T.Resize([300,400]),
        T.ToTensor(),
        T.Normalize(#mean = [0.282, 0.280, 0.282,],
                    #std  = [0.175, 0.173, 0.174]),
                    # mean = [0.280, 0.278, 0.28, ],
                    # std = [0.175, 0.173, 0.174]
        mean =  [0.250, 0.251, 0.253],
        std=    [0.155, 0.155, 0.156])
                        ])
    # dataset = MyDataSet(img_dir,label_dir,col_name='label', transform=transform)
    # train_set, test_set = dataSplit(dataset)
    train_set = MyDataSet(train_img_dir,train_label_dir,col_name='label', transform=transform)
    test_set = MyDataSet(test_img_dir,test_label_dir,col_name='label', transform=transform)



    # print(train_set. __getLabel__())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=4,pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.test_batch_size,shuffle=False,num_workers=4,pin_memory=True)

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.35,patience=5,threshold=0.0001,
                                                     threshold_mode='rel',cooldown=5,min_lr=0.000001, verbose=True,

                                                     )

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    for epoch in range(1,args.epochs + 1):


        total_loss = train(model, epoch, optimizer, train_loader)
        scheduler.step(total_loss)
        print('learning rate：',optimizer.param_groups[0]['lr'])
        test(model, test_loader,classes=classes)
        if epoch %8 ==0:
            save(model,epoch,model_name,classes)


if __name__ == '__main__':

    main()
