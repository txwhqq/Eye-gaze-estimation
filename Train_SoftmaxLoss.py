import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.utils.data as data
import os
import cv2
import scipy.misc
from datetime import datetime
from logger import Logger
import sys
import argparse

device = torch.device('cuda:0')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir',type = str,
        help='Directory where to write event logs',default = 'C:\\Users\\Shinelon\\Desktop\\Eye gaze estimation\\logs\\')
    parser.add_argument('--models_dir',type=str,
        help='Directory where to write trained models and checkpoints.',default='C:\\Users\\Shinelon\\Desktop\\Eye gaze estimation\\checkpoints\\')
    parser.add_argument('--pretrained_model',type=str,help='Loading a pretrained model before training.',default='')
    parser.add_argument('--max_nrog_epochs',type = int,help='Number of epochs to run.',default = 5)
    parser.add_argument('--batch_size',type=int,help='Number of images to process in a batch.',default = 32)
    parser.add_argument('--num_classes',type=int,help='Total number of classes',default = 5)
    parser.add_argument('--learning_rate',type=float,default = 0.0001)
    parser.add_argument('--learning_rate_file',type=str,
        help='When --learing_rate is set to -1, the learning rate will be got from this file.',default='..\\learning_rate\\learning_rate.txt')
    parser.add_argument('--optimizer',type = str,choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP'],
        help='The optimization algorithm to use',default='ADAM')
    parser.add_argument('--image_dir',type = str,default='F:\\Gaze_estimator\\train_images\\train_144_52\\train\\All\\')
    parser.add_argument('--labels_file',type = str,default ='F:\\Gaze_estimator\\train_images\\train_144_52\\train.txt')
    return parser.parse_args(argv)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
                    

def make_dataset(labels_file):
    dataset = []
    flabel = labels_file
    f = open(flabel)
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(' ')
        dataset.append(line)
    f.close()
    return dataset

class EyeDirection(data.Dataset):
    def __init__(self,labels_file,image_dir,transform=None,train = False):
        self.train = train
        self.eye_data = make_dataset(labels_file)
        self.image_dir = image_dir
    
    def __getitem__(self,idx):
        img_name,label = self.eye_data[idx]
        label = int(label)-1
        img_dir = os.path.join(self.image_dir,img_name)
        img = scipy.misc.imread(img_dir, mode='RGB')
        
        #img = scipy.misc.imresize(img, (120, 120), interp='bilinear')
        img = np.atleast_3d(img).transpose(2,0,1).astype(np.float32)
        

        #数据归一化
        mean = np.mean(img)
        std = np.std(img)
        std_adj = np.maximum(std,1.0/np.sqrt(img.size))
        img = np.multiply(np.subtract(img,mean),1/std_adj)
        img = torch.from_numpy(img)
        #print(img.dtype)
        #img = img.reshape(-1,160,160,3)
        #将label ont_hot化
        
        #label1 = torch.randn(1,5)
        #label = label1==label1[0,int(label)]
        #label = label.long()
        #label = torch.reshape(label,(num_classes,))
        label1 = np.ones(1,)
        label = int(label)*label1
        label = torch.from_numpy(label)
        label = label.long()
    
        #print(label.size())
        return img,label
    
    def __len__(self):
        return len(self.eye_data)

class FeatureExract(nn.Module):
    def __init__(self,num_classes):

        super(FeatureExract,self).__init__()
        resnet = models.resnet18(pretrained=True)   #使用resnet18作为特征提取的网络
        modules = list(resnet.children())[:-1]      #删除最后的全连接层
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features,num_classes,bias=True)  #不要偏置项
        #torch.nn.Dropout(0.5)

        
    
    ##
    def forward(self,images):
        features = self.resnet(images)
        #print(features[0,1])
        features = features.reshape(features.size(0),-1)
        features = self.linear(features)
        return features
        


def main(args):

    eyedirection = EyeDirection(args.labels_file,args.image_dir)
    train_loader = torch.utils.data.DataLoader(dataset=eyedirection,batch_size=args.batch_size,shuffle=True)
    model = FeatureExract(args.num_classes).to(device)
    if args.pretrained_model!='':
        model = torch.load(args.pretrained_model)

    model = model.cuda()

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    #['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']

    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)

    elif args.optimizer == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(model.parameters(),lr = args.learning_rate)

    elif args.optimizer == 'ADADELTA':
        optimizer = torch.optim.Adadelta(model.parameters(),lr = args.learning_rate)

    elif args.optimizer == 'RMSPROP':
        optimizer = torch.optim.RMSprop(model.parameters(),lr = args.learning_rate)   
        
    #Train the model
    total_step = len(train_loader)
    print(total_step)

    print('所使用的GPU为: ',torch.cuda.current_device())

    tf_logger = Logger(args.logs_dir)
    j=0
    for epoch in range(args.max_nrog_epochs):
        lr = get_learning_rate_from_file(args.learning_rate_file,epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i,(images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            #forward pass
            outputs = model(images)
            #print(outputs)
            #
            #labels = labels.squeeze()
            loss = criterion(outputs,labels.squeeze())

            #backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j=j+1
            _,argmax = torch.max(outputs,1)
            accuracy = (labels == argmax.squeeze()).float().mean()
            if (i+1) %10 ==0:
                
                print('Epoch [ {}/{}],Step [{}/{}],Loss: {:.4f},Accuracy: {:.4f},lr: {:.8f}'.format(epoch+1,args.max_nrog_epochs,i+1,total_step,loss.item(),accuracy.item(),optimizer.param_groups[0]['lr']))
                model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
                torch.save(model,args.models_dir+model_name+'.ckpt')
            
            info = {'loss': loss.item(),'accuracy': accuracy.item()}
            for tag, value in info.items():
                tf_logger.scalar_summary(tag, value, j)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

