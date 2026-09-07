import os.path
from tqdm import tqdm
from torch.utils.data import Dataset
import pathlib
from torchvision import transforms
import numpy as np
import  torch
from PIL import Image,ImageFile
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.456, 0.485, 0.406], [0.224, 0.229, 0.225])])
import pathlib
import glob
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder



def find_classes(directory):
    classes=['images10','images11','images00']
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(class_to_idx)
    return classes, class_to_idx

class ImageFolderCustom(ImageFolder):

    def find_classes(self,directory):
        return find_classes(directory)



class ImageFolderCustom_v3(ImageFolder):

    def __init__(self, root, transform=None,is_train=True, custom_class_order=['images01','images10','images11']):
        super(ImageFolderCustom, self).__init__(root, transform=transform)
        if custom_class_order:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(custom_class_order)}
            self.classes = custom_class_order
            self.samples = [(path, self.class_to_idx[self.classes[target]]) for path, target in self.samples]
            self.targets = [s[1] for s in self.samples]

    
class ImageFolderCustom2(Dataset):

    def __init__(self, targ_dir,exclude_files, transform=None):
        self.paths = [file for file in pathlib.Path(targ_dir).rglob('*/*') if
                      not any(exclude_file in str(file) for exclude_file in exclude_files)]
        self.transform = transform
        self.class_to_idx = {'images10':1,'images01':2,'images11':3,'images00':0}
    def load_image(self, index: int) :
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) :
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx





