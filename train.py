import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.models import vgg19, vgg16
from tqdm import tqdm

from utils import functions as udf
from utils.guided_filter import GuidedFilter
from models import layers, loss

# from options.train_options import TrainOptions  ## external import, so dot(.) is not needed

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default = 128, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 100000, type = int)
    parser.add_argument("--num_workers", default = 8, type = int)
    parser.add_argument("--beta1", default = 0.5, type = float)
    parser.add_argument("--beta2", default = 0.99, type = float)
    parser.add_argument("--w0", default = 1e4, type = float)
    parser.add_argument("--w1", default = 1e-1, type = float)
    parser.add_argument("--w2", default = 1, type = float)
    parser.add_argument("--w3", default = 3e2, type = float)
    parser.add_argument("--w4", default = 3e2, type = float)
    parser.add_argument("--train_d_times", default = 1, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = 'train_cartoon', type = str)
    parser.add_argument("--use_enhance", default = False)
    parser.add_argument("--checkpoint", default = -1, type = int)
    parser.add_argument("--model_version", default = 0, type = int)

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
    
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    args = arg_parser()

    ## load datasets
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 归一化至[-1, 1]
    ])

    face_photo_dir = 'datasets/ffhq/128px/00000'
    face_photo_dataset = FaceDataset(face_photo_dir, transform=transforms)
    face_photo_loader = DataLoader(face_photo_dataset, batch_size=args.batch_size, shuffle=True)

    # face_cartoon_dir = 'datasets/animeGAN/Hayao/style'
    # face_cartoon_dir = 'datasets/animeGAN/Paprika/style'
    # face_cartoon_dir = 'datasets/animeGAN/Shinkai/style'
    face_cartoon_dir = 'datasets/animeGAN/SummerWar/style'
    face_cartoon_dataset = FaceDataset(face_cartoon_dir, transform=transforms)
    face_cartoon_loader = DataLoader(face_cartoon_dataset, batch_size=args.batch_size, shuffle=True)  # num_workers=args.num_workers

    ### define model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training.")
    G = layers.UnetGenerator(channel=32, num_blocks=4).to(device)

    ### define vgg model

    VGG = layers.Vgg19('vgg19_no_fc.npy', device)

    ### define discriminators

    Dt = layers.Discriminator(in_channel=1, channel=32, patch=True).to(device) # Textural Distriminator 
    Ds = layers.Discriminator(in_channel=3, channel=32, patch=True).to(device) # Surface Distriminator

    ## load model and discriminators

    if args.checkpoint > 0:
        model_path = f"checkpoints/saved_models/gen{args.model_version}_batch_{args.checkpoint}.pth"
        # t_disc_path = f"checkpoints/saved_models/textural{args.model_version}_disc_batch_{args.checkpoint}.pth"
        # s_disc_path = f"checkpoints/saved_models/surface{args.model_version}_disc_batch_{args.checkpoint}.pth"
        # t_disc_state = torch.load(t_disc_path, map_location=device)
        # s_disc_state = torch.load(s_disc_path, map_location=device)
        # Dt.load_state_dict(t_disc_state)
        # Ds.load_state_dict(s_disc_state)
    else:
        # load pretrained model
        model_path = f"checkpoints/saved_models/pre_gen_batch_50000.pth"
    state_dict = torch.load(model_path, map_location=device)
    G.load_state_dict(state_dict)

    ### set mode

    G.train()
    Dt.train()
    Ds.train()
    # VGG.eval()

    ### loss

    BCE_loss = nn.BCELoss().to(device)
    L1_loss = nn.L1Loss().to(device)
    MSE_loss = nn.MSELoss().to(device)

    ### trainable vars

    D_vars = [{'params': Dt.parameters()}, {'params': Ds.parameters()}]
    G_vars = [{'params': G.parameters()}]

    ### optimizers

    D_optim = torch.optim.Adam(D_vars, lr=args.adv_train_lr, betas=(args.beta1, args.beta2))
    G_optim = torch.optim.Adam(G_vars, lr=args.adv_train_lr, betas=(args.beta1, args.beta2))

    ### labels

    real = torch.ones(args.batch_size, 1, args.image_size // 8, args.image_size // 8).to(device)
    fake = torch.zeros(args.batch_size, 1, args.image_size // 8, args.image_size // 8).to(device)

    ### train loop

    for batchs in tqdm(range(args.total_iter)):
        
        ## next batch

        input_photo = next(iter(face_photo_loader)).to(device)
        input_cartoon = next(iter(face_cartoon_loader)).to(device)

        ## train G

        G_optim.zero_grad()

        output = G(input_photo)
        output = GuidedFilter(r=1)(input_photo, output)

        inter_out = output.detach().cpu().numpy()
        if args.use_enhance:
            input_superpixel = udf.selective_adacolor(inter_out, power=1.2)   ## Adaptive coloring algorithm
        else:
            input_superpixel = udf.simple_superpixel(inter_out, seg_num=200)
        input_superpixel = torch.tensor(input_superpixel).to(device)

        blur_fake = GuidedFilter(r=5, eps=2e-1)(output, output)
        blur_cartoon = GuidedFilter(r=5, eps=2e-1)(input_cartoon, input_cartoon)
        gray_fake, gray_cartoon = udf.color_shift(output, input_cartoon)

        # g_loss_blur = BCE_loss(Ds(blur_fake), real)
        # g_loss_gray = BCE_loss(Dt(gray_fake), real)
        # print(gray_cartoon.shape)
        # print(Ds(gray_cartoon).shape)  # ([16, 1, 16, 16])
        
        g_loss_blur = MSE_loss(Ds(blur_fake), real)
        g_loss_gray = MSE_loss(Dt(gray_fake), real)
        
        tv_loss = loss.total_variation_loss(output)

        in_vgg = VGG.build_conv4_4(input_photo)
        out_vgg = VGG.build_conv4_4(output)
        super_vgg = VGG.build_conv4_4(input_superpixel)
        C, H, W = in_vgg.shape[1:]
        photo_loss = L1_loss(in_vgg, out_vgg) / (C * H * W)
        superpixel_loss = L1_loss(super_vgg, out_vgg) / (C * H * W)
        # C, H, W = VGG(input_photo).shape[1:]
        # photo_loss = L1_loss(VGG(input_photo), VGG(output)) / (C * H * W)
        # superpixel_loss = L1_loss(VGG(input_superpixel), VGG(output)) / (C * H * W)

        weights = [args.w0, args.w1, args.w2, args.w3, args.w4]
        recon_loss = weights[3] * photo_loss + weights[4] * superpixel_loss
        g_loss = weights[0] * tv_loss + weights[1] * g_loss_blur + weights[2] * g_loss_gray + recon_loss

        g_loss.backward()
        G_optim.step()

        ## train D

        D_optim.zero_grad()

        output = G(input_photo)
        output = GuidedFilter(r=1)(input_photo, output)

        blur_fake = GuidedFilter(r=5, eps=2e-1)(output, output)
        blur_cartoon = GuidedFilter(r=5, eps=2e-1)(input_cartoon, input_cartoon)
        gray_fake, gray_cartoon = udf.color_shift(output, input_cartoon)

        # d_loss_blur = 0.5 * (BCE_loss(Ds(blur_fake), fake) + BCE_loss(Ds(blur_cartoon), real))
        d_loss_blur = 0.5 * (MSE_loss(Ds(blur_fake), fake) + MSE_loss(Ds(blur_cartoon), real))
        # d_loss_gray = 0.5 * (BCE_loss(Dt(gray_fake), fake) + BCE_loss(Dt(gray_cartoon), real))
        d_loss_gray = 0.5 * (MSE_loss(Dt(gray_fake), fake) + MSE_loss(Dt(gray_cartoon), real))
        d_loss = d_loss_blur + d_loss_gray

        d_loss.backward()
        D_optim.step()

        ## print losses

        if args.checkpoint > 0:
            iters = batchs + 1 + args.checkpoint
        else:
            iters = batchs + 1

        if np.mod(iters, 50) == 0:
            save_image((input_photo+1)/2, f'playground/super_{args.model_version}.{iters}_in.png', nrow=4)
            save_image((output+1)/2, f'playground/super_{args.model_version}.{iters}_out.png', nrow=4)
            save_image((input_superpixel+1)/2, f'playground/super_{args.model_version}.{iters}_super.png', nrow=4)
            print(f" batchs: {iters}  |  g_loss: {g_loss:>5f}  |  d_loss: {d_loss:>5f}  |  tv_loss: {weights[0] * tv_loss:>5f}  |  g_loss_blur: {weights[1] * g_loss_blur:>5f}  |  g_loss_gray: {weights[2] * g_loss_gray:>5f}  |  photo_loss: {weights[3] * photo_loss:>5f}  |  superpixel_loss: {weights[4] * superpixel_loss:>5f} ")
            if np.mod(iters, 500) == 0:
                torch.save(G.state_dict(), f"./checkpoints/saved_models/gen{args.model_version}_batch_{iters}.pth")
                # torch.save(Dt.state_dict(), f"./checkpoints/saved_models/textural{args.model_version}_disc_batch_{iters}.pth")
                # torch.save(Ds.state_dict(), f"./checkpoints/saved_models/surface{args.model_version}_disc_batch_{iters}.pth")

            
            