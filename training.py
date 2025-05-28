import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.nn import Module
import torchvision
from torchvision import transforms
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
import numpy as np

from dataloader import myDataSet
from metrics_calculation import calculate_metrics_ssim_psnr, calculate_UIQM
from model import ProposedMynet

__all__ = [
    "Trainer",
    "setup",
    "training",
]

# Loss funtions (L1 + SSIM + VGG)
class CombinedLoss(nn.Module):
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg = torchvision.models.vgg16(pretrained=True).features.to(config.device).eval()
        self.vgg_layers = [3, 8, 15]
        self.ssim_weight = 0.5
        self.vgg_weight = 0.1

    def calculate_ssim(self, output, target):
        mse = torch.mean((output - target) ** 2)
        ssim = 1 - mse
        return ssim

    def get_vgg_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.vgg_layers:
                features.append(x)
        return features

    def forward(self, output, target):
        l1_loss = self.l1_loss(output, target)
        ssim_loss = self.calculate_ssim(output, target)
        output_features = self.get_vgg_features(output)
        target_features = self.get_vgg_features(target)
        vgg_loss = 0
        for out_f, tgt_f in zip(output_features, target_features):
            vgg_loss += self.l1_loss(out_f, tgt_f)
        vgg_loss /= len(output_features)
        total_loss = l1_loss + self.ssim_weight * ssim_loss + self.vgg_weight * vgg_loss
        return total_loss, l1_loss, vgg_loss

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader=None):
        device = config.device
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []
        best_uiqm = -float('inf')
        best_model_path = os.path.join(config.snapshots_folder, 'best_model.pth')

        # Start epoch from config
        start_epoch = config.start_epoch if hasattr(config, 'start_epoch') else 0

        if config.test and test_dataloader is not None:
            UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
            mean_uiqm = np.mean(UIQM)
            print(f"Epoch [{start_epoch}] - UIQM: {mean_uiqm:.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")
            if mean_uiqm > best_uiqm:
                best_uiqm = mean_uiqm
                os.makedirs(config.snapshots_folder, exist_ok=True)
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best model at epoch {start_epoch} with UIQM: {best_uiqm:.4f}")

        for epoch in trange(start_epoch, config.num_epochs, desc="Training Epochs"):
            primary_loss_tmp, vgg_loss_tmp, total_loss_tmp = 0, 0, 0

            # Learning rate decay
            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] *= 0.7

            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
            for inp, label, _ in batch_iterator:
                inp, label = inp.to(device), label.to(device)

                self.model.train()
                self.opt.zero_grad()
                out = self.model(inp)
                loss, mse_loss, vgg_loss = self.loss(out, label)

                loss.backward()
                self.opt.step()

                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()

            total_loss_lst.append(total_loss_tmp / len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp / len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp / len(train_dataloader))

            if epoch % config.print_freq == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}] - Total Loss: {total_loss_lst[-1]:.4f}, Primary Loss: {primary_loss_lst[-1]:.4f}, VGG Loss: {vgg_loss_lst[-1]:.4f}")

            if config.test and epoch % config.eval_steps == 0 and test_dataloader is not None:
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                mean_uiqm = np.mean(UIQM)
                print(f"Evaluation at Epoch [{epoch+1}] - UIQM: {mean_uiqm:.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")
                if mean_uiqm > best_uiqm:
                    best_uiqm = mean_uiqm
                    os.makedirs(config.snapshots_folder, exist_ok=True)
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Saved best model at epoch {epoch+1} with UIQM: {best_uiqm:.4f}")

            if epoch % config.snapshot_freq == 0:
                os.makedirs(config.snapshots_folder, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(config.snapshots_folder, f'model_epoch_UIE_net_UD_{epoch}.pth'))

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
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

def setup(config):
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProposedMynet().to(config.device)

    # Load pretrained weights if available
    if hasattr(config, 'pretrained_model_path') and config.pretrained_model_path and os.path.exists(config.pretrained_model_path):
        print(f"Loading pretrained model weights from {config.pretrained_model_path}")
        model.load_state_dict(torch.load(config.pretrained_model_path, map_location=config.device))

    transform = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])

    train_dataset = myDataSet(config.raw_images_path, config.clear_image_path, transform, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    print("Train Dataset Reading Completed.")
    print(model)

    loss = CombinedLoss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(model, opt, loss)

    if config.test:
        test_dataset = myDataSet(config.test_images_path, None, transform, is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer

    return train_dataloader, None, model, trainer

def training(config):
    train_loader, test_loader, model, trainer = setup(config)
    trainer.train(train_loader, config, test_loader)
    print("==================")
    print("Training complete!")
    print("==================")

def main(config):
    training(config)

if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace(
        raw_images_path="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/",
        clear_image_path="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB/",
        test_images_path="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/",
        GTr_test_images_path="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB/",
        test=False,
        lr=0.0001,
        step_size=50,
        num_epochs=200,
        train_batch_size=8,
        test_batch_size=8,
        resize=128,
        cuda_id=0,
        print_freq=1,
        snapshot_freq=1,
        snapshots_folder="./snapshots/",
        output_images_path="./data/output/",
        eval_steps=1,
        pretrained_model_path="./snapshots/model_epoch_UIE_net_UD_99.pth",  # 99th epoch weights
        start_epoch=100  # Start from 100th epoch
    )
    main(config)