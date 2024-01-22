import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: list = [256, 256], mask_suffix: str = '', augmentations: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augmentations = augmentations

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True)])

        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True)
                ])

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        img_ndarray = np.asarray(pil_img)
        if len(img_ndarray.shape) == 2:
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if img_ndarray.max() > 1:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def preprocess_outpic(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale[0]), int(scale[1])
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        if len(img_ndarray.shape) == 2:
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if img_ndarray.max() > 1:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            print('Image.fromarray')
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            print('Image.fromarray')
            return Image.fromarray(torch.load(filename).numpy())
        else:
            # print('Image.open')
            return Image.open(filename)

    def __getitem__(self, idx):

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        np.random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            img = self.img_transform(img)

        np.random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            mask = self.gt_transform(mask)
        return {
            'image': img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')#mask_suffix='_matte'


class Dataset_Pro(Dataset):
    def __init__(self, images_dir: str, masks_dir: str,images_indices, masks_indices, scale: list = [256, 256], mask_suffix: str = '', augmentations: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augmentations = augmentations

        self.ids = [splitext(str(file))[0] for file in images_indices if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True)])

        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.scale[0], self.scale[1]), antialias=True)
                ])

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        # print('pil_img.mode=', pil_img.mode)
        if is_mask:
            if pil_img.mode == str("L"):
                pass
            else:
                pil_img = pil_img.convert("L")
        # print('pil_img.mode=', pil_img.mode)

        img_ndarray = np.asarray(pil_img)
        # print('img_ndarray', img_ndarray.shape)
        if len(img_ndarray.shape) == 2:
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if img_ndarray.max() > 1:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            print('Image.fromarray')
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            print('Image.fromarray')
            return Image.fromarray(torch.load(filename).numpy())
        else:
            # print('Image.open')
            return Image.open(filename)

    def __getitem__(self, idx):

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        name = self.ids[idx]

        if name[-1] == str(']'):
            mask_file = list(self.masks_dir.glob(name[:-1] + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))
        else:
            mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        np.random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            # print('img', img.shape)
            img = self.img_transform(img)

        np.random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            # print('mask', mask.shape)
            mask = self.gt_transform(mask)
        return {
            'image': img,
            'mask': mask
        }