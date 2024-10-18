import os
import nibabel as nib
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

#import medmnist
#from medmnist import INFO, Evaluator

size=256
slice_num=16





# 分类函数
def get_event_classification(nii_gz_filename):
    # 去除文件名的后缀
    file_name = os.path.splitext(os.path.basename(nii_gz_filename))[0]
    file_name = os.path.splitext(os.path.basename(file_name))[0]
    # 去除数字和下划线等非字母字符
    file_name_cleaned = ''.join(char for char in file_name if char.isalpha())
    # 读取CSV文件
    csv_filename = '/home/featurize/for2d_all/metadata174_updateall0625_fitniigz176.csv'
    csv_filename ='metadata176_survive_12month.csv'
    df = pd.read_csv(csv_filename, encoding='latin1')
    
    # 查找与清理后文件名相匹配的行
    matching_row = df[df['pinyin_adjust'] == file_name_cleaned]
    
    # 如果找到了匹配的行，则返回对应的event分类，否则返回None
    if not matching_row.empty:
        return matching_row.iloc[0]['target']
    else:
        return None

class ResizeTransform:
    def __init__(self, size,slice_num=16):
        self.size = size
        self.resize = transforms.Resize((size, size))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 

    def __call__(self, img):
        # img是一个512x512x16的numpy数组，我们需要分别处理每一层
        resized_slices = []
        for i in range(slice_num):
            slice_interval=int(128/slice_num)
            slice_img = img[:, :, slice_interval*i]
            slice_img = self.to_pil(slice_img)  # 转换为PIL图像
            resized_slice = self.resize(slice_img)  # 调整大小
            resized_slice = self.to_tensor(resized_slice)  # 转换为张量
            #resized_slice = self.normalize(resized_slice)
            resized_slices.append(resized_slice)
        resized_img = torch.stack(resized_slices, dim=0)
        mean = torch.mean(resized_img)
        std = torch.std(resized_img)
        
        # 正态分布归一化处理
        normalized_img = (resized_img - mean) / std
        normalized_img = normalized_img.squeeze(1)
        #print(resized_img.shape)
        resized_img = resized_img.squeeze(1)  # 增加一个通道维度
        return normalized_img
    '''
    def __call__(self, img):
        # img是一个512x512x16的numpy数组，我们需要分别处理每一层
        resized_slices = []
        for i in range(img.shape[2]):
            slice_img = img[:, :, i]
            slice_img = torch.from_numpy(slice_img).unsqueeze(0)  # 增加一个通道维度
            resized_slice = self.resize(slice_img)  # 调整大小
            resized_slices.append(resized_slice.squeeze(0))  # 移除增加的通道维度
        resized_img = torch.stack(resized_slices, dim=2)
        return resized_img
'''



transform_niigz = transforms.Compose([
    ResizeTransform(size,slice_num),
    #transforms.ToTensor()
])

class CTDataset_niigz(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir#'/home/featuriz/dataset176'
        self.transform = transform_niigz
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        #self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        nii_gz_file = self.file_list[idx]
        file_path = os.path.join(self.data_dir, nii_gz_file)
        
        # 读取NIfTI文件
        img = nib.load(file_path).get_fdata()
        img = img.astype(np.float32)

        # 获取类别
        label = get_event_classification(nii_gz_file)
        
        # 转换
        if self.transform:
            img = self.transform(img)
        
        return img, label


transform = transforms.Compose([
    #ResizeTransform(size,slice_num),
    transforms.ToTensor()
])
# 自定义Dataset类
class CTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        #self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
            # 初始化时预先检查每个文件的label，并只保留有效的样本
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz') and self.get_valid_label(f)]
        
    def get_valid_label(self, filename):
        """ 检查文件的label是否有效，针对有数据会被删去 """
        label = get_event_classification(filename)
        return label is not None  # 如果 label 不是 None，返回 True，表示是有效的样本
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        nii_gz_file = self.file_list[idx]
        file_path = os.path.join(self.data_dir, nii_gz_file)
        
        # 读取NIfTI文件
        #img = nib.load(file_path).get_fdata()
        #img = img.astype(np.float32)
        img =  np.load(file_path)
        img = img['data']
        #print(img.shape)
        img = torch.from_numpy(img)
        #print(img.shape)
        # 获取类别
        label = get_event_classification(nii_gz_file)
        
        # 转换
        #if self.transform:
        #    img = self.transform(img)
        # 检查label的错误
        if img is None or label is None:
            print(f"Error loading data at index {idx}")
            print( self.file_list[idx])
            print(img.shape)
            print(label)
        
        return img, label




# 定义数据目录和transform



def squeeze_transform(x):
    x=x.astype(np.float32)
    return x.squeeze(0)  # 移除第一个维度

def get_loader(dataset="ipf",filetype='npz'):
    if dataset=="medicalmnist":
        # 选取了肺部的nodulemnist3d 初始是28*28*28
        data_flag = 'nodulemnist3d'
        download = True
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        BATCH_SIZE = 64
        lr = 0.001

        # 一些基础信息
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        # load the data
        train_dataset = DataClass(split='train', download=download,transform=squeeze_transform)
        test_dataset = DataClass(split='test', download=download,transform=squeeze_transform)
        # encapsulate data into dataloader form
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader, test_loader
    
    folder_path = "/home/featurize/dataset176_processed"
    print(folder_path)
    #dataset = CustomImageDataset(folder_path=folder_path)

    # 创建一个 DataLoader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    if filetype == 'niigz':
        folder_path= '/home/featurize/dataset176'
        dataset = CTDataset_niigz(folder_path, transform=transform)
    else:
        dataset = CTDataset(folder_path, transform=transform)

    # 划分训练集和验证集
    '''
    print(len(dataset))
    train_size = int(0.7 * len(dataset))  # 训练集占比 80%
    val_size = len(dataset) - train_size  # 验证集占比 20%
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False , collate_fn=custom_collate_fn)
    '''
    #labels = np.array([dataset[i][1] for i in range(len(dataset))])  # 获取数据集中所有样本的标签
    train_size = int(0.7 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



    # 获取训练集的标签
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    inbalance = 'oversample'

    if inbalance == 'oversample':
        # 过采样
        oversampler = RandomOverSampler()
        train_indices = np.arange(len(train_dataset))
        train_indices_resampled, train_labels_resampled = oversampler.fit_resample(train_indices.reshape(-1, 1), train_labels)
    
        # 使用 resampled indices 创建新的训练集
        train_dataset_resampled = Subset(train_dataset, train_indices_resampled.flatten())
    
    elif inbalance == 'downsample':
        # 欠采样
        undersampler = RandomUnderSampler()
        train_indices = np.arange(len(train_dataset))
        train_indices_resampled, train_labels_resampled = undersampler.fit_resample(train_indices.reshape(-1, 1), train_labels)
    
        # 使用 resampled indices 创建新的训练集
        train_dataset_resampled = Subset(train_dataset, train_indices_resampled.flatten())
    
    else:
        # 不进行采样处理，使用原始的训练集
        train_dataset_resampled = train_dataset



    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset_resampled, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

   

    
    # 测试遍历数据集
    #for images, labels in train_loader:
    #    print("批次图像形状:", images.shape)  # 输出应为 (batch_size, ...)
    #    print("批次标签:", labels)
    #    break  # 退出循环，只查看第一个批次
    return train_loader, test_loader

def custom_collate_fn(batch):
    # 过滤掉 label 为 None 的样本
    filtered_batch = [(image, label) for image, label in batch if label is not None]
    
    # 检查是否所有的样本都被过滤掉
    if len(filtered_batch) == 0:
        return None
    
    # 使用默认的 collate_fn 进行批次拼接
    return torch.utils.data.default_collate(filtered_batch)



if __name__=="__main__":
    
    train_loader,test_loader=get_loader()#filetype='niigz')#dataset="medicalmnist")

                                   
    '''
    data_dir ="/home/featurize/data/dataset176"
    n=64
    transform = transforms.Compose([
        ResizeTransform(n),
        #transforms.ToTensor()
    ])

    # 创建Dataset和DataLoader
    dataset = CTDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
'''
    # 示例：迭代DataLoader
    for imgs, labels in train_loader:
        print(imgs.shape)
        #print(labels)
        #summed_imgs = imgs.sum(dim=(2, 3))  # dim=(2, 3) 表示在第2和第3维度上求和
        # 打印结果
        #print("Shape after summing:", summed_imgs.shape)
        #print("Summed images:")
        #print(summed_imgs)
        #print(imgs)
        #print(labels)
        #imgs = imgs.numpy()  # 将 PyTorch 张量转换为 NumPy 数
        break


    for i, (images, labels) in enumerate(train_loader):
        try:
            # 检查每一批的数据是否为 None
            if images is None or labels is None:
                print(f"Error: NoneType found in batch {i}")
            else:
                print(f"Batch {i} loaded successfully: {images.shape}, {labels.shape}")
    
        except Exception as e:
            print(f"Error in batch {i}: {e}")


    device='cpu'
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        print(images.shape)
        break

    # 获取 train_loader 和 test_loader 中的样本总数
    train_samples = len(train_loader.dataset)
    test_samples = len(test_loader.dataset)
    
    print(f"Total samples in train_loader: {train_samples}")
    print(f"Total samples in test_loader: {test_samples}")

        