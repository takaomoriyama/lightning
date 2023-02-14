# 包括实现UNet需要的杂项函数，主要是数据集相关的
import math
import os
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms.functional as T_F
import numpy as np
import imageio
import random
import torchvision
from torch.onnx import export
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS


# 输入图像名，获得图像的对应标签文件名
def get_label_fname(fname):
    return 'Labels_' + fname


def twelve_way_transform(img, lab, num):
    if num / 3 == 1:
        img, lab = T_F.hflip(img), T_F.hflip(lab)
    if num / 3 == 2:
        img, lab = T_F.vflip(img), T_F.vflip(lab)
    if num / 3 == 3:
        img, lab = T_F.vflip(img), T_F.vflip(lab)
        img, lab = T_F.hflip(img), T_F.hflip(lab)
    img = [T_F.gaussian_blur(i, 3) if num % 3 == 0 else
           T_F.gaussian_blur(i, 5) if num % 3 == 1 else i for i in torch.split(img, dim=0, split_size_or_sections=1)]
    img = torch.cat(img, dim=0)
    return img, lab


def gaussian_blur_3d(img, kernel):
    img = [T_F.gaussian_blur(i, kernel) for i in torch.split(img, dim=0, split_size_or_sections=1)]
    img = torch.cat(img, dim=0)
    return img

def adjust_contrast_3d(img):
    img = [T_F.adjust_contrast(i, random.uniform(0.75, 1.5)) for i in torch.split(img, dim=0, split_size_or_sections=1)]
    img = torch.cat(img, dim=0)
    return img




# 输入图像或者标签的路径，得到已标准化的图像张量或者标签的张量
def path_to_tensor(path, label=False):
    # imread()将文件读取成一个numpy array
    img = imageio.v3.imread(path)
    if label:  # ToTensor()对16位图不方便，因此才用from_numpy
        # from_numpy()则将numpy array转换成张量
        tensor = torch.from_numpy(img)
    else:
        # 将0到65535的uint16 array压缩到0到1的double array，相当于进行了标准化
        img = img / 65535
        # print(img)
        # from_numpy()则将numpy array转换成张量
        tensor = torch.from_numpy(img)
    return tensor


def four_way_transform(img, num):
    if num / 3 <= 1:
        img = F.hflip(img)
    if num / 3 <= 2:
        img = F.vflip(img)
    if num / 3 <= 3:
        img = F.hflip(img)
        img = F.vflip(img)
    return img


# 输入图像文件夹和标签文件夹的路径，在应用图像增强参数后，生成图像跟标签有一一对应关系的文件清单
def make_dataset_train(image_dir, label_dir, extensions=IMG_EXTENSIONS):
    image_label_pair = []
    image_files = os.listdir(image_dir)
    for fname in sorted(image_files):
        if has_file_allowed_extension(fname, extensions):
            path = os.path.join(image_dir, fname)
            label_path = os.path.join(label_dir, get_label_fname(fname))
            image_label_pair.append((path, label_path))
    # image_label_pair example：
    # [
    # ('datasets\\train\\img\\testimg1.tif', 'datasets\\train\\img\\Labels_testimg1.tif'),
    # ('datasets\\train\\img\\testimg2.tif', 'datasets\\train\\img\\Labels_testimg2.tif')
    # ]
    return image_label_pair


def make_dataset_val(image_dir, label_dir, extensions=IMG_EXTENSIONS):
    image_label_pair = []
    image_dir = os.path.expanduser(image_dir)
    label_dir = os.path.expanduser(label_dir)
    for root, _, fnames in sorted(os.walk(image_dir)):
        label_root = os.path.join(label_dir, os.path.relpath(root, image_dir))
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.normpath(os.path.join(root, fname))
                label_path = os.path.normpath(os.path.join(label_root, get_label_fname(fname)))
                image_label_pair.append((path, label_path))
    return image_label_pair


# 输入图像路径，生成包含图像路径的文件清单
def make_dataset_predict(image_dir, extensions=IMG_EXTENSIONS):
    path_list = []
    image_dir = os.path.expanduser(image_dir)
    for root, _, fnames in sorted(os.walk(image_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.normpath(os.path.join(root, fname))
                path_list.append(path)
    # pic_list的例子
    # ['datasets\\predict\\testpic1.tif',
    #  'datasets\\predict\\testpic2.tif']
    return path_list


# 自定义的数据集结构，用于存储训练数据
class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_train(images_dir, labels_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = []
        self.lab_tensors = []

        # Pre-allocating the memory for img_tensors and lab_tensors
        for idx in range(self.num_files):
            # Convert the image and label path to tensors
            img_tensor, lab_tensor = path_to_tensor(self.file_list[idx][0], label=False), path_to_tensor(
                self.file_list[idx][1], label=True)
            lab_tensor = lab_tensor.long()
            # Append the tensors to the list
            self.img_tensors.append(img_tensor)
            self.lab_tensors.append(lab_tensor)

        # Stack the lists of tensors to form a single tensor
        self.img_tensors = torch.stack(self.img_tensors)
        self.lab_tensors = torch.stack(self.lab_tensors)

        super().__init__()

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        img_tensor = self.img_tensors[idx]
        lab_tensor = self.lab_tensors[idx]

        # Decide whether to apply each random transformation
        random_transforms = [random.random() < 0.5 for i in range(6)]

        # Apply vertical flip
        if random_transforms[0]:
            img_tensor, lab_tensor = T_F.vflip(img_tensor), T_F.vflip(lab_tensor)
        # Apply horizontal flip
        if random_transforms[1]:
            img_tensor, lab_tensor = T_F.hflip(img_tensor), T_F.hflip(lab_tensor)
        # Apply Gaussian blur
        if random_transforms[2]:
            img_tensor = gaussian_blur_3d(img_tensor, random.randint(1, 2) * 2 + 1)
        # Apply padding and cropping
        if random_transforms[3]:
            # Randomly generate padding sizes
            left_pad_size = random.randint(2, 64)
            right_pad_size = random.randint(2, 64)
            up_pad_size = random.randint(2, 64)
            down_pad_size = random.randint(2, 64)
            img_tensor = img_tensor[:, up_pad_size:-down_pad_size, left_pad_size:-right_pad_size]
            lab_tensor = lab_tensor[:, up_pad_size:-down_pad_size, left_pad_size:-right_pad_size]
            padding = (left_pad_size, right_pad_size,
                       up_pad_size, down_pad_size,
                       0, 0)
            img_tensor = F.pad(img_tensor, padding, mode="constant")
            lab_tensor = F.pad(lab_tensor, padding, mode="constant")
        # Apply rotations
        if random_transforms[4]:
            rotation_angle = random.uniform(-90, 90)
            img_tensor = T_F.rotate(img_tensor, rotation_angle)
            lab_tensor = T_F.rotate(lab_tensor, rotation_angle)
        # Apply contrast adjustments
        if random_transforms[5]:
            img_tensor = adjust_contrast_3d(img_tensor)
        img_tensor = img_tensor[None, :].to(torch.float16)
        return img_tensor, lab_tensor

    
class Val_Dataset(torch.utils.data.Dataset):
    file_list: list

    def __init__(self, images_dir, labels_dir):
        self.file_list = make_dataset_val(images_dir, labels_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = []
        self.lab_tensors = []

        # Pre-allocating the memory for img_tensors and lab_tensors
        for idx in range(self.num_files):
            img_tensor, lab_tensor = path_to_tensor(self.file_list[idx][0], label=False), path_to_tensor(
                self.file_list[idx][1], label=True)
            lab_tensor = lab_tensor.long()
            self.img_tensors.append(img_tensor)
            self.lab_tensors.append(lab_tensor)

        self.img_tensors = torch.stack(self.img_tensors)
        self.lab_tensors = torch.stack(self.lab_tensors)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        img_tensor = self.img_tensors[idx]
        lab_tensor = self.lab_tensors[idx]
        # 手动给张量加上一个"Channel"维度，以便修复需要Channel的问题
        img_tensor = img_tensor[None, :].to(torch.float16)
        lab_tensor = lab_tensor.long()
        return img_tensor, lab_tensor


# 自定义的数据集结构，用于存储预测数据
class Predict_Dataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, vol=True):
        self.file_list = make_dataset_predict(images_dir)
        self.volume = vol
        self.hw_size = 512
        self.depth_size = 12
        self.hw_overlap = 128
        self.img_list = [path_to_tensor(file, label=False) for file in self.file_list]
        # Get the size of all tensors in img_list, assume all tensors in img_list are the same size
        if self.volume:
            self.depth = self.img_list[0].shape[0]
            self.height = self.img_list[0].shape[1]
            self.width = self.img_list[0].shape[2]
        else:
            self.height = self.img_list[0].shape[0]
            self.width = self.img_list[0].shape[1]
        # Calculate the multipliers for padding and cropping
        self.depth_multiplier = math.ceil(self.depth / self.depth_size) if self.volume else 1
        self.height_multiplier = math.ceil(self.height / self.hw_size)
        self.width_multiplier = math.ceil(self.width / self.hw_size)
        self.total_multiplier = self.depth_multiplier * self.height_multiplier * self.width_multiplier
        # Padding and cropping
        self.padded_img_list = []
        for img_tensor in self.img_list:
            if self.volume:
                paddings = (self.hw_overlap, self.width_multiplier * self.hw_size + self.hw_overlap - self.width,
                            self.hw_overlap, self.height_multiplier * self.hw_size + self.hw_overlap - self.height,
                            0, 0)
                img_tensor = img_tensor[None, :]
                img_tensor = F.pad(img_tensor, paddings, mode="constant")
                # Loop through each depth, height, and width index
                for depth_idx in range(self.depth_multiplier):
                    for height_idx in range(self.height_multiplier):
                        for width_idx in range(self.width_multiplier):
                            # Calculate the start and end indices for depth, height, and width
                            depth_start = min(depth_idx * self.depth_size, self.depth - self.depth_size)
                            depth_end = min(depth_start + self.depth_size, self.depth)
                            height_start = height_idx * self.hw_size
                            height_end = height_start + self.hw_size + self.hw_overlap * 2
                            width_start = width_idx * self.hw_size
                            width_end = width_start + self.hw_size + self.hw_overlap * 2
                            cropped_tensor = img_tensor[:, depth_start:depth_end, height_start:height_end, width_start:width_end]
                            self.padded_img_list.append(cropped_tensor)
            else:
                paddings = (self.hw_overlap, self.width_multiplier * self.hw_size + self.hw_overlap - self.width,
                            self.hw_overlap, self.height_multiplier * self.hw_size + self.hw_overlap - self.height,)
                img_tensor = img_tensor[None, :]
                img_tensor = F.pad(img_tensor, paddings, mode="constant")
                for height_idx in range(self.height_multiplier):
                    for width_idx in range(self.width_multiplier):
                        height_start = height_idx * self.hw_size
                        height_end = height_start + self.hw_size + self.hw_overlap * 2
                        width_start = width_idx * self.hw_size
                        width_end = width_start + self.hw_size + self.hw_overlap * 2
                        cropped_tensor = img_tensor[:, height_start:height_end, width_start:width_end]
                        self.padded_img_list.append(cropped_tensor)
        super().__init__()

    def __len__(self):
        return len(self.file_list) * self.total_multiplier

    def __getitem__(self, idx):
        return self.padded_img_list[idx].to(torch.float16)

    def __getoriginalvol__(self):
        return self.img_list[0].shape


def stitch_output_volumes(output_volumes, original_volume, vol=True):
    hw_size = 512
    depth_size = 12
    overlap_size = 128 # change here for the new overlapping size
    if vol:
        depth = original_volume[0]
        height = original_volume[1]
        width = original_volume[2]
        depth_multiplier = math.ceil(depth / depth_size)
        height_multiplier = math.ceil(height / hw_size)
        width_multiplier = math.ceil(width / hw_size)
        total_multiplier = depth_multiplier * height_multiplier * width_multiplier
        result_volume = torch.zeros((depth_multiplier*depth_size,
                                     height_multiplier*hw_size,
                                     width_multiplier*hw_size), dtype=torch.int8)
        for i in range(total_multiplier):
            tensors_in_1_layer = height_multiplier * width_multiplier
            depth_idx = math.floor(i / tensors_in_1_layer) % depth_multiplier
            height_idx = math.floor(i / width_multiplier) % height_multiplier
            width_idx = i % width_multiplier
            tensor_work_with = output_volumes[i][:,
                                                 overlap_size:-overlap_size,
                                                 overlap_size:-overlap_size]
            depth_start = min(depth_idx * depth_size, depth - depth_size)
            depth_end = min(depth_start + depth_size, depth)
            height_start = height_idx * hw_size
            height_end = height_start + hw_size
            width_start = width_idx * hw_size
            width_end = width_start + hw_size
            result_volume[depth_start:depth_end, height_start:height_end, width_start:width_end] = tensor_work_with
        result_volume = result_volume[0:depth, 0:height, 0:width]
    else:
        height = original_volume[0]
        width = original_volume[1]
        height_multiplier = math.ceil(height / (hw_size + overlap_size))
        width_multiplier = math.ceil(width / (hw_size + overlap_size))
        total_multiplier = height_multiplier * width_multiplier
        result_volume = torch.zeros((height, width), dtype=torch.int8)
        for i in range(total_multiplier):
            height_idx = i % height_multiplier
            width_idx = math.floor(i / height_multiplier) % width_multiplier
            height_start = height_idx * (hw_size + overlap_size) + overlap_size//2
            height_end = height_start + hw_size
            width_start = width_idx * (hw_size + overlap_size) + overlap_size//2
            width_end = width_start + hw_size
            result_volume[height_start:height_end, width_start:width_end] = output_volumes[i][overlap_size//2:-overlap_size//2, overlap_size//2:-overlap_size//2]
    return result_volume


def predictions_to_final_img(predictions, direc, original_volume, vol_out=False):
    tensor_list = []
    for prediction in predictions:  # 将这些输出张量项目从predictions里拿出来
        # 分割prediction，形成包含了多个元素（图片张量）的元组，每个元素的Batch维度都是1
        splitted = torch.split(prediction, split_size_or_sections=1, dim=0)
        for single_tensor in splitted:  # 将这个元组里每个单独元素（图片张量）拆分出来
            # It appears that if the final output is a volume, I will need to squeeze the first dimension(Batch)
            if vol_out:
                single_tensor = torch.squeeze(single_tensor, 0)
            list.append(tensor_list, single_tensor)

    full_image = stitch_output_volumes(tensor_list, original_volume, vol_out)
    array = np.asarray(full_image)
    imageio.v3.imwrite(uri=f'{direc}/full_prediction.tif', image=np.uint8(array))


# 这里的函数用于测试各个组件是否能正常运作
if __name__ == "__main__":
    #fake_predictions = [torch.randn(4, 512, 512), torch.randn(4, 512, 512)]
    #predictions_to_final_img(fake_predictions)
    test_dataset = Train_Val_Dataset('datasets/train/img',
                                     'datasets/train/lab',
                                     vol=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=8)
    test_tensor = test_dataset.__getitem__(1)
    #print(test_loader.dataset[0][0])
    #test_tensor = path_to_tensor('datasets/train/lab/Labels_Jul13ab_nntrain_1.tif', label=True)
    #print(test_tensor.shape)
    #test_tensor = composed_transform(test_tensor, 1)
    #print(test_tensor.shape)
    #print(test_tensor)
    #test_array = np.asarray(test_tensor)
    #im = imageio.volsave('Result.tif', test_array)
