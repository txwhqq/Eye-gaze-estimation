import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.utils.data as data
import os
import scipy.misc
import numpy as np

device = torch.device('cuda:0')


#img = scipy.misc.imresize(img, (120, 120), interp='bilinear')
    
class FeatureExract(nn.Module):
    def __init__(self,num_classes):

        super(FeatureExract,self).__init__()
        resnet = models.resnet18(pretrained=True)  
        modules = list(resnet.children())[:-1]     
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features,num_classes,bias=True) 
        
    
    ##
    def forward(self,images):
        features = self.resnet(images)
        #print(features[0,1])
        features = features.reshape(features.size(0),-1)
        features = self.linear(features)
        return features
        

#model = FeatureExract(3).to(device)


model = torch.load('F:\\Gaze_estimator\\weights\\model_original.ckpt') ##Load your model here
model = model.to(device)
f = open('F:\\Gaze_estimator\\test_images(unknown)\\images\\test.txt')##Open the annotation file
Testset = []
lines = f.readlines()
for line in lines:
    line = line.strip('\n')
    line = line.split(' ')
    Testset.append(line)
f.close()

model.eval()
num = 0
N = 0
with torch.no_grad():
    for im,label in Testset:
        img = scipy.misc.imread('F:\\Gaze_estimator\\test_images(unknown)\\images\\Test\\All\\'+im, mode='RGB') ##Read images of test set
        img = np.atleast_3d(img).transpose(2,0,1).astype(np.float32)
    
        mean = np.mean(img)
        std = np.std(img)
        std_adj = np.maximum(std,1.0/np.sqrt(img.size))
        img = np.multiply(np.subtract(img,mean),1/std_adj)
        img = img.reshape(-1,3,52,144)
        img = torch.from_numpy(img)
        img = img.to(device)
        
        outputs = model(img)
        #outputs = outputs.squeeze()
        _,argmax = torch.max(outputs,1)
        #print(argmax+1)
        if (argmax+1)!=int(label):
            num = num+1
            #print(argmax+1)
            #print(im)
        N = N+1
    
    print('acc:',1.0-num/N)



