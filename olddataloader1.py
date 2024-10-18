import os
import nibabel as nib
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split
# 分类函数
def get_event_classification(nii_gz_filename):
    # 去除文件名的后缀
    file_name = os.path.splitext(os.path.basename(nii_gz_filename))[0]
    file_name = os.path.splitext(os.path.basename(file_name))[0]
    # 去除数字和下划线等非字母字符
    file_name_cleaned = ''.join(char for char in file_name if char.isalpha())
    # 读取CSV文件
    csv_filename = '/home/featurize/for2d_all/metadata174_updateall0625_fitniigz176.csv'
    df = pd.read_csv(csv_filename, encoding='latin1')
    
    # 查找与清理后文件名相匹配的行
    matching_row = df[df['pinyin_adjust'] == file_name_cleaned]
    
    # 如果找到了匹配的行，则返回对应的event分类，否则返回None
    if not matching_row.empty:
        return matching_row.iloc[0]['event']
    else:
        return None






# 自定义Dataset类
class CTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        
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

# 定义数据目录和transform
size=128
slice_num=16
transform = transforms.Compose([
    ResizeTransform(size,slice_num),
    #transforms.ToTensor()
])


def get_loader():
    folder_path = "/home/featurize/data/dataset176"
    print(folder_path)
    #dataset = CustomImageDataset(folder_path=folder_path)

    # 创建一个 DataLoader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    dataset = CTDataset(folder_path, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.7 * len(dataset))  # 训练集占比 80%
    val_size = len(dataset) - train_size  # 验证集占比 20%
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    # 测试遍历数据集
    #for images, labels in train_loader:
    #    print("批次图像形状:", images.shape)  # 输出应为 (batch_size, ...)
    #    print("批次标签:", labels)
    #    break  # 退出循环，只查看第一个批次
    return train_loader, test_loader


if __name__=="__main__":
    data_dir ="/home/featurize/data/dataset176"
    n=64
    transform = transforms.Compose([
        ResizeTransform(n),
        #transforms.ToTensor()
    ])

    # 创建Dataset和DataLoader
    dataset = CTDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 示例：迭代DataLoader
    for imgs, labels in dataloader:
        print(imgs.shape)
        summed_imgs = imgs.sum(dim=(2, 3))  # dim=(2, 3) 表示在第2和第3维度上求和
        
        # 打印结果
        print("Shape after summing:", summed_imgs.shape)
        print("Summed images:")
        print(summed_imgs)
        print(imgs)
        print(labels)
        
        
        imgs = imgs.numpy()  # 将 PyTorch 张量转换为 NumPy 数组

        # 遍历每个通道，保存为 .npy 文件
        for i in range(imgs.shape[1]):  # 遍历通道数，这里是 4
            channel_data = imgs[0, i]  # 获取第一个样本的第 i 个通道数据
            filename = f'channel_{i}.npy'
            np.save(filename, channel_data)
        break