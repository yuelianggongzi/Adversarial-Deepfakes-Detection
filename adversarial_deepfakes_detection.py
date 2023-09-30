import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 预处理的设置
# 图片转化为 backbone网络规定的图片大小
# 归一化是减去均值，除以方差
# 把 numpy array 转化为 tensor 的格式
#image_size = 224
image_size = 112
batch_size = 32
r_mean, g_mean, b_mean = 0.4914, 0.4822, 0.4465
r_std, g_std, b_std = 0.247, 0.243, 0.261

my_tf = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize([r_mean,g_mean,b_mean], [r_std,g_std,b_std])])

# 读取数据集 CIFAR-10 的图，有10个标签，5万张图片，进行预处理。
#train_dataset= torchvision.datasets.CIFAR10(root='./',train=True,transform=my_tf,download=True)
#test_dataset= torchvision.datasets.CIFAR10(root='./',train=False,transform=my_tf,download=True)
train_path ="/mnt/publicStoreA/videodata/dfdc-adv-2/all-keyframes-phase/"
#val_path = "/mnt/publicStoreA/videodata/deepfakes-videos/all-keyframe-phase/"
#test_path =  "/mnt/publicStoreA/lishaomei/image_quality_evaluation/dataset/Test/"
test_path =  "/mnt/publicStoreA/videodata/dfdc-adv-2/all-keyframes-phase/"
#test_path =  "/mnt/publicStoreA/videodata/Phase-Face-Dataset-3/all/"
train_set = dataset.MyDatasetPhase("Train", image_size, train_path, transform=my_tf)
#val_set = dataset.MyDataset("Val", image_size, val_path, transform=my_tf)
test_set = dataset.MyDatasetPhase("Test", image_size, test_path,transform=my_tf)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
resnet50=torchvision.models.resnet50(pretrained=True)



# 固定网络框架全连接层之前的参数
for param in resnet50.parameters():
    param.requires_grad=False
# 将vgg最后一层输出的类别数，改为cifar-10的类别数（10）
num_ftrs = resnet50.fc.in_features
class_size = 2
resnet50.fc = nn.Sequential(nn.Linear(num_ftrs,class_size),
                            nn.LogSoftmax(dim=1))



# 超参数设置
learning_rate = 0.01
num_epoches = 100
batch_size = 32
momentum = 0.9
# 多分类损失函数，使用默认值
criterion = nn.CrossEntropyLoss()
# 梯度下降，求解模型最后一层参数

optimizer = torch.optim.SGD(resnet50.parameters(),lr=learning_rate,momentum=momentum)
# 判断使用CPU还是GPU
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 图片分批次送入内存（32张图片,batch_size），进行计算。
#train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
#test_dataloader = DataLoader(dataset=test_dataset)
interval_checkpoint=1
# 训练阶段
#resnet50.to(device)
resnet50=resnet50.cuda()
resnet50.train()
print('************')
print(num_epoches)
for epoch in range(num_epoches):
    total, correct = 0, 0
    print(f"epoch: {epoch+1}")
    for idx,(img,label)in enumerate(train_dataloader):
        # images = img.to(device)
        # labels = label.to(device)
        images = img.cuda()
        labels = label.cuda()
        output = resnet50(images)
        _, idx1 = torch.max(output.data, 1)  # 输出最大值的位置
        total += labels.size(0) # 全部图片
        correct +=(idx1==labels).sum() # 正确的图片
        loss = criterion(output,labels)
        #print(loss)
        loss.backward() # 损失反向传播
        optimizer.step() # 更新梯度
        optimizer.zero_grad() # 梯度清零
        if idx%100==0:
            print(f"current loss = {loss.item()}")
    if epoch % interval_checkpoint == 0:
        model_path = f'model-0930/resnet50-DFDC-all-epoch-{epoch}_checkpoint.pth'
        print('this is 0930 resnet50 for DFDC phase all:')
        print(f"accuracy:{100.*correct/total}\n")
        model_checkpoint = {
            'epoch_num': epoch,
            'state_dict': resnet50.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss_train': epoch_loss_train,
            #'loss_val': epoch_loss_val,
        }
        torch.save(resnet50.state_dict(), model_path)
# model_path = f'model-0926/resnet50-DFDC-all-epoch-{epoch}_checkpoint.pth'
#
# #torch.save(model_checkpoint, model_path)
# torch.save(resnet50.state_dict(), model_path)

# # 测试阶段
# #resnet50.to(device)
# resnet50=resnet50.cuda()
# resnet50.eval() # 把训练好的模型的参数冻结
# total,correct = 0 , 0
# for img,label in test_dataloader:
#     # images = img.to(device)
#     # labels = label.to(device)
#     images = img.cuda()
#     labels = label.cuda()
#     output = resnet50(images)
#     _,idx = torch.max(output.data,1) # 输出最大值的位置
#     total += labels.size(0) # 全部图片
#     correct +=(idx==labels).sum() # 正确的图片
#
# print('this is 0926 resnet50-50 for DFDC phase all:')
# print(f"accuracy:{100.*correct/total}\n")
#
# # 测试阶段
# #resnet50.to(device)
# resnet50=resnet50.cuda()
# for epoch in range(num_epoches-1):
#     if epoch % interval_checkpoint == 0:
#         model_path = f'model-0926/resnet50-DFDC-all-epoch-{epoch}_checkpoint.pth'
#         resnet50.load_state_dict(torch.load(model_path))
#         resnet50.eval() # 把训练好的模型的参数冻结
#         total,correct = 0 , 0
#         for img,label in test_dataloader:
#             # images = img.to(device)
#             # labels = label.to(device)
#             images = img.cuda()
#             labels = label.cuda()
#             output = resnet50(images)
#             _,idx = torch.max(output.data,1) # 输出最大值的位置
#             total += labels.size(0) # 全部图片
#             correct +=(idx==labels).sum() # 正确的图片
#
#         print(f'this is 0926 resnet50-epoch-{epoch} for DFDC all :')
#         print(f"accuracy:{100.*correct/total}\n")
#



