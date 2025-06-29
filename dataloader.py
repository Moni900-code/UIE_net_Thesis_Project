import torch
import os
from PIL import Image
from torch.utils.data import Dataset

def get_image_list(raw_image_path, clear_image_path=None):
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

def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in validation mode.
    Returns:
        - For mode='val': (raw_img_batch, None, name_batch)
        - For mode='train' or 'test': (raw_img_batch, clear_img_batch, name_batch)
    """
    raw_imgs = []
    clear_imgs = []
    names = []
    
    mode = batch[0][3]  # Get mode from the first item (added mode to __getitem__ return)
    
    for item in batch:
        raw_imgs.append(item[0])
        if mode != "val":
            clear_imgs.append(item[1])
        names.append(item[2])
    
    raw_img_batch = torch.stack(raw_imgs)
    name_batch = names
    
    if mode == "val":
        return raw_img_batch, None, name_batch
    else:
        clear_img_batch = torch.stack(clear_imgs)
        return raw_img_batch, clear_img_batch, name_batch

class myDataSet(Dataset):
    def __init__(self, raw_image_path, clear_image_path=None, transform=None, mode="train"):
        """
        mode: 'train', 'val', or 'test'
        - 'train' => input + GT required
        - 'val'   => only input image, GT = None
        - 'test'  => input + GT required
        """
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.mode = mode
        self.transform = transform
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path)

    def __getitem__(self, index):
        raw_path, clear_path, image_name = self.image_list[index]

        # Load and transform input image
        raw_img = Image.open(raw_path).convert('RGB')
        if self.transform:
            raw_img = self.transform(raw_img)

        # Handle GT image
        if self.mode == "val":
            # No GT during validation
            clear_img = None
        else:
            if clear_path is None:
                raise ValueError(f"GT image not found for mode={self.mode}")
            clear_img = Image.open(clear_path).convert('RGB')
            if self.transform:
                clear_img = self.transform(clear_img)

        return raw_img, clear_img, image_name, self.mode  # Added mode for collate_fn

    def __len__(self):
        return len(self.image_list)
