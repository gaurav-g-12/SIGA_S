import os
import sys
import time
import random
import string
import argparse
import time
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '6,5'
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms

from utils import CharsetMapper
from dataset1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from Parallel_test import benchmark_all_eval, validate
import PIL
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img = TF.normalize(img, self.mean, self.std)
        return img

def FC_pred(preds, converter):
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data)
    return preds_str


crop_main_dir = '/home/gaurav/scratch/TextBookPages/results/LineLevel/Detection/craft/crops'
output_file_path = '/home/gaurav/scratch/TextBookPages/results/LineLevel/Recognization/siga/craft/flex'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /home/jcc/GTK/Text Recognition Dataset/training/label
    # /media/xr/guantongkun/downloads/data_lmdb_release/training/label
    
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
    parser.add_argument("--test_batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping value. default=5")
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, default='TSBA', help="CRNN|TRBA")
    parser.add_argument("--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN", )
    parser.add_argument("--input_channel", type=int, default=3,
                        help="the number of input channel of Feature extractor", )
    parser.add_argument("--output_channel", type=int, default=384,
                        help="the number of output channel of Feature extractor", )
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the LSTM hidden state")
    """ Data processing """
    parser.add_argument("--batch_max_length", type=int, default=25, help="maximum-label-length")
    parser.add_argument("--imgH", type=int, default=48, help="the height of the input image")
    parser.add_argument("--imgW", type=int, default=160, help="the width of the input image")
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        # default="abcdefghijklmnopqrstuvwxyz1234567890",
        help="character label",
    )
    parser.add_argument("--sensitive", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--NED", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--Aug", action="store_true", help="whether to use augmentation |None|Blur|Crop|Rot|", )
    """ Semi-supervised learning """
    parser.add_argument("--semi", type=str, default="None",
                        help="whether to use semi-supervised learning |None|PL|MT|", )
    parser.add_argument("--MT_C", type=float, default=1, help="Mean Teacher consistency weight")
    parser.add_argument("--MT_alpha", type=float, default=0.999, help="Mean Teacher EMA decay")
    parser.add_argument("--model_for_PseudoLabel", default="", help="trained model for PseudoLabel")
    parser.add_argument("--self_pre", type=str, default="RotNet",
                        help="whether to use `RotNet` or `MoCo` pretrained model.", )
    """ exp_name and etc """
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument("--manual_seed", type=int, default=111, help="for random seed setting")
    # ./saved_models/TRBA_synth_SA_/best_accuracy.pth
    parser.add_argument("--saved_model", default="/home/gaurav/scratch/SIGA/SIGA_S/model94/167000.pth", help="path to model to continue training")
    parser.add_argument("--Iterable_Correct", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--benchmark_all_eval", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--language", action="store_true", help="For Normalized edit_distance")
    opt = parser.parse_args()


    opt.character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    opt.Transformation = "TPS"
    opt.FeatureExtraction = "SVTR"
    opt.SequenceModeling = "BiLSTM"
    opt.Prediction = "Attn"

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  
    cudnn.deterministic = True

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    
    converter = CharsetMapper(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    model.load_state_dict(torch.load(opt.saved_model)['net'], strict=False)
    
    model.to(device)

    transform = ResizeNormalize((opt.imgW, opt.imgH))


    for subfolder in sorted(os.listdir(crop_main_dir), key=lambda x: int(os.path.splitext(x)[0].split('/')[-1])):
        subfolderPath = os.path.join(crop_main_dir, subfolder)

        txt_file_name = output_file_path + f'/{subfolder}.txt'

        lineCount = 0
        batch_size = 1

        with open(txt_file_name, 'w') as f:
            for image_name in sorted(os.listdir(subfolderPath), key=lambda x: (int(os.path.splitext(x)[0].split('/')[-1].split('_')[0]), int(os.path.splitext(x)[0].split('/')[-1].split('_')[-1].split('.')[0]))):

                image_path = os.path.join(subfolderPath, image_name)
                # print(image_path)

                image = PIL.Image.open(image_path).convert("RGB")
                image_tensors = transform(image)
                image_tensors = image_tensors.to(device)
                image_tensors = image_tensors.unsqueeze(0)

                # print(image_tensors.shape)

                length_for_loss = None
                masks = None

                text_for_pred = (torch.LongTensor(batch_size).fill_(95).to(device))
                share_pred1 = model(image_tensors, masks, length_for_loss, text=text_for_pred, is_train=False)
                pred = FC_pred(share_pred1, converter)
                print(pred)


                if int(image_name.split('_')[0]) == lineCount:
                    f.write(pred[0]+' ')
                
                else:
                    lineCount = int(image_name.split('_')[0])
                    f.write('\n')
                    f.write(pred[0]+' ')
                
            f.write('\n')

