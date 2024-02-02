from torch.utils.data import Dataset
import albumentations as A


from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms.functional as F

# from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 255, 'constant')

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset(Dataset):
    def __init__(self, image_dict, opt, is_validation=False):
        self.is_validation = is_validation
        self.pars = opt

        #####
        self.image_dict = image_dict

        #####
        self.init_setup()

        #####
        if "bninception" not in opt.arch:
            self.f_norm = normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
            self.f_norm = normalize = transforms.Normalize(
                mean=[0.502, 0.4588, 0.4078], std=[0.0039, 0.0039, 0.0039]
            )

        transf_list = []

        self.crop_size = crop_im_size = 224 if "googlenet" not in opt.arch else 227
        if opt.augmentation == "big":
            crop_im_size = 256

        #############
        self.normal_transform = []
        if not self.is_validation:
            if opt.augmentation == "base" or opt.augmentation == "big":
                self.normal_transform.extend(
                    [
                        transforms.RandomResizedCrop(size=crop_im_size),
                        transforms.RandomHorizontalFlip(0.5),
                    ]
                )
            elif opt.augmentation == "adv":
                self.normal_transform.extend(
                    [
                        transforms.RandomResizedCrop(size=crop_im_size),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                        transforms.RandomHorizontalFlip(0.5),
                    ]
                )
            elif opt.augmentation == "red":
                self.normal_transform.extend(
                    [
                        transforms.Resize(size=256),
                        transforms.RandomCrop(crop_im_size),
                        transforms.RandomHorizontalFlip(0.5),
                    ]
                )
            elif opt.augmentation == "rove":
                self.normal_transform.extend(
                    [
                        transforms.Resize(size=190, max_size=200),
                        transforms.Pad(padding=30, fill=255),
                        SquarePad(),
                        transforms.RandomAffine(degrees=(-5,5),translate=(0.005,0.05),scale=(0.9,1.0), fill=255),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomResizedCrop(size=crop_im_size, scale=(0.75,1)),
                        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    ]
                )
                """ self.normal_transform = Transforms(
                    A.Compose(
                        [
                            A.LongestMaxSize(max_size=224 + 10, p=1),
                            A.PadIfNeeded(
                                min_height=244,
                                min_width=244,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=1,
                            ),
                            A.Rotate(
                                limit=5,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=1,
                            ),
                            A.CenterCrop(height=224, width=224, p=1),
                            A.VerticalFlip(p=0.5),
                            A.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                            ToTensorV2(),
                        ]
                     )
                ) """
        else:
            self.normal_transform.extend(
                [transforms.Resize(220, max_size=224), SquarePad()]
            )
            """ self.normal_transform = Transforms(
                    A.Compose(
                        [
                            A.LongestMaxSize(max_size=224 + 10, p=1),
                            A.PadIfNeeded(
                                min_height=244,
                                min_width=244,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=1,
                            ),
                            A.Rotate(
                                limit=5,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=1,
                            ),
                            A.CenterCrop(height=224, width=224, p=1),
                            A.VerticalFlip(p=0.5),
                            A.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                            ToTensorV2(),
                        ]
                if opt.augmentation == "rove":
                self.normal_transform = Transforms(
                    A.Compose(
                        [
                            A.LongestMaxSize(max_size=224 + 10, p=1),
                            A.PadIfNeeded(
                                min_height=244,
                                min_width=244,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=1,
                            ),
                            A.Rotate(
                                limit=5,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=1,
                            ),
                            A.RandomCrop(height=224, width=224, p=1),
                            A.VerticalFlip(p=0.5),
                            A.GridDistortion(
                                num_steps=5,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=0.5,
                            ),
                            A.ElasticTransform(
                                alpha=15,
                                sigma=5,
                                alpha_affine=5,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=(255, 255, 255),
                                p=0.2,
                            ),
                            A.ColorJitter(
                                brightness=0.1, contrast=0.1, hue=0.025, p=0.25
                            ),
                            A.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                            ToTensorV2(),
                        ]
                    )
                ) """
            # if opt.augmentation != "rove":
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)

    def init_setup(self):
        self.n_files = np.sum(
            [len(self.image_dict[key]) for key in self.image_dict.keys()]
        )
        self.avail_classes = sorted(list(self.image_dict.keys()))

        counter = 0
        temp_image_dict = {}
        for i, key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [
            [(x[0], key) for x in self.image_dict[key]]
            for key in self.image_dict.keys()
        ]
        self.image_list = [x for y in self.image_list for x in y]

        self.image_paths = self.image_list

        self.is_init = True

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        input_image = self.ensure_3dim(Image.open(self.image_list[idx][0]))

        ### Basic preprocessing.
        im_a = self.normal_transform(input_image)
        if "bninception" in self.pars.arch:
            im_a = im_a[range(3)[::-1], :]
        return self.image_list[idx][-1], im_a, idx

    def __len__(self):
        return self.n_files
