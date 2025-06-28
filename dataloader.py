import torch
import os
from PIL import Image

def get_image_list(raw_image_path, clear_image_path=None):
    """
    Return list of [raw_image_path, clear_image_path or None, image_file_name]
    clear_image_path can be None (for test without GT)
    """
    raw_image_list = sorted(os.listdir(raw_image_path))
    if clear_image_path is not None:
        clear_image_list = sorted(os.listdir(clear_image_path))
        assert len(raw_image_list) == len(clear_image_list), "Number of raw and clear images must be same"
    else:
        clear_image_list = [None] * len(raw_image_list)

    image_list = []
    for raw_img_name, clear_img_name in zip(raw_image_list, clear_image_list):
        raw_img_path = os.path.join(raw_image_path, raw_img_name)
        clear_img_path = os.path.join(clear_image_path, clear_img_name) if clear_img_name is not None else None
        image_list.append([raw_img_path, clear_img_path, raw_img_name])
    return image_list


class myDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path=None, transform=None, is_train=True):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train = is_train
        self.transform = transform
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path)

    def __getitem__(self, index):
        raw_path, clear_path, image_name = self.image_list[index]

        raw_img = Image.open(raw_path).convert('RGB')
        if self.transform is not None:
            raw_img = self.transform(raw_img)

        if clear_path is not None:
            clear_img = Image.open(clear_path).convert('RGB')
            if self.transform is not None:
                clear_img = self.transform(clear_img)
        else:
            clear_img = None

        if self.is_train:
            # Training mode: return both input and GT images
            return raw_img, clear_img, image_name
        else:
            # Testing mode:
            # return GT if available, else None
            return raw_img, clear_img, image_name

    def __len__(self):
        return len(self.image_list)
