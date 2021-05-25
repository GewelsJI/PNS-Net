import os
from torch.utils.data import Dataset
from utils.preprocess import *
from PIL import Image
import torch
from config import config


# pretrain dataset
class Pretrain(Dataset):
    def __init__(self, img_dataset_list, transform):
        self.file_list = []
        for dataset in img_dataset_list:
            data_dir = config.img_dataset_root + dataset
            gt_path = data_dir + '/GT/'
            img_path = data_dir + '/Frame/'
            self.file_list.extend([(img_path + name,
                                    gt_path + name.replace('jpg', 'png'))
                                   for name in os.listdir(img_path)])

        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_path, label_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self._process(img, label)
        return img, label

    def _process(self, img, label):
        img, label = self.img_label_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)


def get_pretrain_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize(config.size[0], config.size[1]),
        Random_crop_Resize(15),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    train_loader = Pretrain(config.img_dataset_list, transform=trsf_main)

    return train_loader


# finetune dataset
class VideoDataset(Dataset):
    def __init__(self, video_dataset_list, transform=None, time_interval=1):
        super(VideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        self.video_train_list = []

        for video_name in video_dataset_list:
            video_root = os.path.join(config.video_dataset_root, video_name, 'Train')
            cls_list = os.listdir(video_root)
            self.video_filelist = {}
            for cls in cls_list:
                self.video_filelist[cls] = []
                cls_path = os.path.join(video_root, cls)
                cls_img_path = os.path.join(cls_path, "Frame")
                cls_label_path = os.path.join(cls_path, "GT")
                tmp_list = os.listdir(cls_img_path)
                tmp_list.sort()
                for filename in tmp_list:
                    self.video_filelist[cls].append((
                        os.path.join(cls_img_path, filename),
                        os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                    ))

            # ensemble
            for cls in cls_list:
                li = self.video_filelist[cls]
                for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                    batch_clips = []
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + time_interval * t])
                    self.video_train_list.append(batch_clips)
            self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
        return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)


def get_video_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(config.video_dataset_list, transform=trsf_main, time_interval=1)

    return train_loader


if __name__ == "__main__":
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(config.video_dataset_list, transform=trsf_main, time_interval=1)
