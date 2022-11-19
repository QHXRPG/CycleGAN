from torchvision.models import vgg19
import torch
import numpy
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


v = vgg19(pretrained=True)
t1 = ToPILImage()
t2 = ToTensor()

img_noise = cv2.imread('img_noise.jpg')
img_noise = cv2.resize(img_noise,(400,400))
img_noise = cv2.cvtColor(img_noise,cv2.COLOR_BGR2RGB)

img_con = cv2.imread('castle1.jpg')
img_con = cv2.resize(img_con,(400,400))
img_con = cv2.cvtColor(img_con,cv2.COLOR_BGR2RGB)

img_sty = cv2.imread('style.jpg')
img_sty = cv2.resize(img_sty,(400,400))
img_sty = cv2.cvtColor(img_sty,cv2.COLOR_BGR2RGB)
img_con,img_sty,img_noise = (t2(img_con),t2(img_sty),t2(img_noise))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sty = v.features
    def forward(self,x):
        k=[]
        i=0
        while i<len(self.sty):
            x = self.sty[i](x)
            if i==3:x1 = x
            elif i==9:x2 = x
            elif i==16:x3 = x
            elif i==23:x4 = x
            elif i==30:x5 = x
            elif i==31:x6 = x
            elif i==36:x7 = x
            i = i + 1
        k.append(x1)
        k.append(x2)
        k.append(x3)
        k.append(x4)
        k.append(x5)
        k.append(x6)
        k.append(x7)
        return k,x1

class SynthesizedImage(nn.Module):
    def __init__(self):
        super(SynthesizedImage, self).__init__()
        self.weight = nn.Parameter(torch.rand(1,3,400,400))
    def forward(self):
        return self.weight


net = Net()
G = SynthesizedImage()
#格拉姆矩阵
def gram(x):
    x = x.squeeze(0)
    x = x.reshape(x.shape[0],x.shape[1]**2)
    return torch.mm(x,x.T)

youhuaqi = torch.optim.Adam(G.parameters(),lr=0.2)
loss = nn.MSELoss()
img_con = img_con.unsqueeze(0)
img_sty = img_sty.unsqueeze(0)
img = img_noise
z=0
for iii in range(450):
    z = z+1
    if z==10:
        img1=img.squeeze(0)
        img1 = img1.detach().numpy()
        img1 = img1.swapaxes(0, 1)
        img1 = img1.swapaxes(1, 2)
        plt.figure()
        plt.imshow(img1)
        plt.show()
        z=0
    """计算损失函数"""
    img = G()
    _,G_con = net(img)
    _, img_con_y = net(img_con)
    (img_sty1, img_sty2, img_sty3, img_sty4, img_sty5, img_sty6, img_sty7), _ = net(img_sty)
    img_sty1, img_sty2, img_sty3, img_sty4, img_sty5, img_sty6, img_sty7 = (gram(img_sty1), gram(img_sty2), gram(img_sty3),
                                                                  gram(img_sty4), gram(img_sty5), gram(img_sty6),gram(img_sty7))
    (G_sty1, G_sty2, G_sty3, G_sty4, G_sty5, G_sty6, G_sty7),_ = net(img)
    G_sty1, G_sty2, G_sty3, G_sty4, G_sty5, G_sty6, G_sty7 = (gram(G_sty1), gram(G_sty2), gram(G_sty3),
                                                      gram(G_sty4), gram(G_sty5), gram(G_sty6),gram(G_sty7))
    l1 = loss(G_con,img_con_y)/2
    l2 = loss(G_sty1,img_sty1)/(4*64*64*400*400)+loss(G_sty2,img_sty2)/(4*128*128*200*200)+\
         loss(G_sty3,img_sty3)/(4*256*256*100*100)+loss(G_sty4,img_sty4)/(4*512*50*512*50)+\
         loss(G_sty5,img_sty5)/(4*512*25*512*25) +loss(G_sty6,img_sty6)/(4*512*12*512*12)+\
         loss(G_sty7,img_sty7)/(4*512*512*6*7)
    l = 0.05*l1+0.95*l2

    """训练"""
    youhuaqi.zero_grad()
    l.backward()
    youhuaqi.step()

