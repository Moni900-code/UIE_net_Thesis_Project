import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision
from torchvision import transforms
import numpy as np
from dataloader import myDataSet
from metrics_calculation import calculate_metrics_ssim_psnr, calculate_UIQM
from model import ProposedMynet
from argparse import Namespace

@torch.no_grad()
def eval(config, test_dataloader, test_model):
    test_model.eval()
    os.makedirs(config.output_images_path, exist_ok=True)
    
    for img_batch, _, name_batch in test_dataloader:
        img_batch = img_batch.to(config.device)
        output_batch = test_model(img_batch)
        for i in range(output_batch.size(0)):
            output_img = output_batch[i].unsqueeze(0)
            filename = name_batch[i]
            torchvision.utils.save_image(output_img, os.path.join(config.output_images_path, filename))

    SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path, config.GTr_test_images_path)
    UIQM_measures = calculate_UIQM(config.output_images_path)
    return UIQM_measures, SSIM_measures, PSNR_measures

# config setup
config = Namespace(
    test_images_path="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/",
    GTr_test_images_path="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB/",
    output_images_path="./data/output/",
    test_batch_size=1,
    resize=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_path="./snapshots/best_model.pth"
)

# model load
model = ProposedMynet().to(config.device)
try:
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    print(f"Loaded model weights from {config.model_path}")
except FileNotFoundError:
    print(f"Model file {config.model_path} not found. Trying epoch 9 weights...")
    model.load_state_dict(torch.load('./snapshots/model_epoch_litemodel9.pth', map_location=config.device))
model.eval()

# dataloader setup
transform = transforms.Compose([
    transforms.Resize((config.resize, config.resize)),
    transforms.ToTensor()
])

test_dataset = myDataSet(config.test_images_path, None, transform, is_train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
print("Test Dataset Loaded.")

# evaluation
UIQM, SSIM, PSNR = eval(config, test_dataloader, model)

# print result
print(f"[Test Result] UIQM: {np.mean(UIQM):.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")