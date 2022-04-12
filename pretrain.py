import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
from tqdm import tqdm

from models import layers

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 128, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 50000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = 'pretrain')
    parser.add_argument("--checkpoint", default = -1, type=int)

    args = parser.parse_args()
    
    return args

class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_list = list()
        for name in os.listdir(self.img_dir):
            self.img_list.append(os.path.join(self.img_dir, name))
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = read_image(self.img_list[idx])
        label = None
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image


if __name__ == '__main__':
    
    args = arg_parser()

    ## load datasets
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 归一化至[-1, 1]
    ])

    face_photo_dir = 'datasets/ffhq/128px/00000'
    face_photo_dataset = FaceDataset(face_photo_dir, transform=transforms)
    face_photo_loader = DataLoader(face_photo_dataset, batch_size=args.batch_size, shuffle=True)

    losses = []

    ### define model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for pretraining.")
    G = layers.UnetGenerator(channel=32, num_blocks=4).to(device)

    ### load model

    if args.checkpoint > 0:
        model_path = f"checkpoints/saved_models/pre_gen_batch_{args.checkpoint}.pth"
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        G.load_state_dict(state_dict)

    ### set mode

    G.train()

    ### loss

    L1_loss = nn.L1Loss().to(device)

    ### trainable vars

    G_vars = [{'params': G.parameters()}]

    ### optimizers

    G_optim = torch.optim.Adam(G_vars, lr=args.adv_train_lr, betas=(0.5, 0.99))

    ### train loop

    for batchs in tqdm(range(args.total_iter)):

        ## next batch

        input_photo = next(iter(face_photo_loader)).to(device)

        ## pretrain G

        output = G(input_photo)
        g_loss = L1_loss(input_photo, output)
        
        ## backpropagation
        G_optim.zero_grad()
        g_loss.backward()
        G_optim.step()

        ## print losses

        if args.checkpoint > 0:
            iters = batchs + 1 + args.checkpoint
        else:
            iters = batchs + 1

        losses.append(g_loss.data.item())
        if iters % 10 == 0:
            mean_loss = np.mean(losses)
            print(f"    batchs: {iters}     |   g_loss: {mean_loss:>5f}    ")
            if iters % 500 == 0:
                torch.save(G.state_dict(), f"./checkpoints/saved_models/pre_gen_batch_{iters}.pth")
            losses = []
            