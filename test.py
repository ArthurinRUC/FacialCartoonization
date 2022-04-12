import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image

import os
import cv2
import numpy as np
from tqdm import tqdm

from utils import functions as udf
from utils.guided_filter import GuidedFilter
from models import layers, loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for testing.")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", default = 0, type = int)
    parser.add_argument("--batch", default = 0, type = int)

    args = parser.parse_args()
    
    return args

# from options.test_options import TestOptions
# opt = TestOptions().parse()   # get testing options

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
    

def cartoonize(input_photo, input_cartoon, model_path, save_folder):

    save_image((input_photo+1)/2, save_folder + f'/in-{args.model_version}-{args.batch}.png', nrow=4)

    ## load model
    G = layers.UnetGenerator(channel=32, num_blocks=4).to(device)
    
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    G.load_state_dict(state_dict)

    G.eval()

    network_out = G(input_photo)
    final_out = GuidedFilter(r=1)(input_photo, network_out)

    save_image((final_out+1)/2, save_folder + f'/out-{args.model_version}-{args.batch}.png', nrow=4)
    # save_image(input_photo - final_out, save_folder + '/final_out--.png')

    print(f'Test loss: {torch.nn.L1Loss()(input_photo, final_out):>5f}')

    inter_out = final_out.detach().numpy()
    # superpixel = udf.selective_adacolor(inter_out, power=1.2, seg_num=1000)
    superpixel = udf.simple_superpixel(inter_out, seg_num=200)

    save_image((torch.tensor(superpixel)+1)/2, save_folder + '/superpixel.png')

    save_image((torch.tensor(superpixel) - final_out+1)/2, save_folder + '/superpixel--.png')


if __name__ == '__main__':

    args = arg_parser()

    # model_path = f'checkpoints/saved_models/pre_gen_batch_{args.batch}.pth'
    model_path = f'checkpoints/saved_models/gen{args.model_version}_batch_{args.batch}.pth'
    save_folder = 'playground'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    ## load datasets
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    face_photo_dir = 'datasets/ffhq/128px/00000'
    face_photo_dataset = FaceDataset(face_photo_dir, transform=transforms)
    face_photo_loader = DataLoader(face_photo_dataset, batch_size=1, num_workers=4, shuffle=True)

    face_cartoon_dir = 'datasets/animeGAN/Hayao/style'
    face_cartoon_dataset = FaceDataset(face_cartoon_dir, transform=transforms)
    face_cartoon_loader = DataLoader(face_cartoon_dataset, batch_size=32, num_workers=4, shuffle=True)

    cartoonize(next(iter(face_photo_loader)).to(device), next(iter(face_cartoon_loader)).to(device), model_path, save_folder)
    

    
