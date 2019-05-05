import os.path
import random
import torchvision.transforms as transforms
import torch
import math
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.folder)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = h
        if (self.opt.readnum==4):
            assert (self.opt.loadSize >= self.opt.fineSize)
            w2 = int(w / 4)
            A1 = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B1 = AB.crop((w2, 0, 2 * w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A2 = AB.crop((2 * w2, 0, 3 * w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B2 = AB.crop((3 * w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A1 = transforms.ToTensor()(A1)
            B1 = transforms.ToTensor()(B1)
            A2 = transforms.ToTensor()(A2)
            B2 = transforms.ToTensor()(B2)

            A1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A1)
            B1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B1)
            A2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A2)
            B2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B2)
            B1 = B1 / torch.norm(B1, 2, 0, out=None, keepdim=True)
            B2 = B2 / torch.norm(B2, 2, 0, out=None, keepdim=True)

            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

            num_par = self.opt.num_par
            par_len = 4
            i = 1
            if output_nc == 3:
                my_parameter = torch.Tensor([float(AB_path[-((par_len + 1) * i + 3):-((par_len + 1) * (i - 1) + 4)])])
                for i in range(2, num_par + 1):
                    temp = torch.Tensor([float(AB_path[-((par_len + 1) * i + 3):-((par_len + 1) * (i - 1) + 4)])])
                    my_parameter = torch.cat((temp, my_parameter), 0)
            return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2, 'par': my_parameter, 'Img_paths': AB_path}
        else:
            raise ValueError('--the number of images needed to read  %s.' % self.opt.readnum)

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


































