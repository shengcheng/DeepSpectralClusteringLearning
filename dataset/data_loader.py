from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.datasets import CarsDataset


def get_data_loader(cfg):

    dataset_set = CarsDataset(
        data_dir = cfg.data_dir,
        ids_per_batch = cfg.ids_per_batch,
        ims_per_id = cfg.ims_per_id,
        is_train = cfg.is_train,
        resize_h_w = cfg.resize_h_w,
        crop_h_w = cfg.crop_h_w,
        mirror = cfg.mirror,
        im_mean = cfg.im_mean,
        im_std = cfg.im_std,
        scale = cfg.scale
    )

    return dataset_set
    # if cfg.is_train:
    #     transform = transforms.Compose([
    #         transforms.Resize(cfg.resize_h_w),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomResizedCrop(cfg.crop_h_w),
    #         transforms.ToTensor(),
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize(cfg.resize_h_w),
    #         transforms.CenterCrop(cfg.crop_h_w),
    #         transforms.ToTensor()
    #     ])
    # dataset = CarsDataset(root_dir=cfg.data_dir, train=cfg.is_train, transform=transform)
    # shuffle = True if cfg.is_train else False
    # return DataLoader(
    #     dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=shuffle,
    #     num_workers=cfg.num_workers,
    #     drop_last=True
    # )

