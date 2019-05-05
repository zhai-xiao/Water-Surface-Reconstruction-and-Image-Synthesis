import torch
import itertools
import torch.nn.functional as F
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable

class SolverModel(BaseModel):
    def name(self):
        return 'SolverModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default model did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_c', type=float, default=10.0, help='weight for cos loss')
            parser.add_argument('--alpha_i1', type=float, default=10.0,
                                help='weight for L1 loss of image 1')
            parser.add_argument('--beta_i2', type=float, default=10.0, help='weight for L1 loss of image 2')
            parser.add_argument('--gamma_p', type=float, default=10.0, help='weight for param loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses

        self.loss_names = ['G', 'G_A', 'G_B1','G_B2', 'D_B', 'D_A', 'L1_A1', 'L1_B2', 'par', 'L1_B1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A1', 'fake_B1', 'real_B1', 'fake_A1']
        visual_names_B = ['real_B2','fake_A2','real_A2']
        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        self.netG_A = networks.define_G_A( opt.num_par, opt.input_nc, opt.output_nc, opt.n_blocks,opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G_B( opt.num_par, opt.output_nc, opt.input_nc, opt.n_blocks,opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nc+opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc+opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_BA_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPar = torch.nn.L1Loss()
            self.CosineSimilarity = torch.nn.CosineEmbeddingLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # the input includes four images and parameters feature.
        self.real_A1 = input['A1'].to(self.device)
        self.real_B1 = input['B1'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.real_B2 = input['B2'].to(self.device)
        self.real_par = input['par'].to(self.device)
        self.image_paths = input['Img_paths']


    def forward(self):

#
        if self.opt.pre_train:
            self.fake_B1,self.fake_ParImg,self.fake_par = self.netG_A(self.real_A1)
            self.fake_B1 = self.fake_B1 / torch.norm(self.fake_B1, 2, 1, out=None, keepdim=True)
            self.fake_A1 = self.real_A1
            self.fake_A2 = self.real_A2
        else:
            self.fake_B1,self.fake_ParImg,self.fake_par = self.netG_A(self.real_A1)
            self.fake_B1 = self.fake_B1 / torch.norm(self.fake_B1, 2, 1, out=None, keepdim=True)

            self.fake_A1 = self.netG_B(self.real_B1, self.fake_ParImg)
            self.fake_A2 = self.netG_B(self.real_B2, self.fake_ParImg)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_AB1 = self.fake_AB_pool.query(torch.cat((self.real_A1, self.fake_B1), 1))
        real_AB1 = torch.cat((self.real_A1, self.real_B1), 1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_AB1, fake_AB1)

    def backward_D_B(self):
        fake_BA2 = self.fake_BA_pool.query(torch.cat((self.real_B2, self.fake_A2), 1))
        real_BA2 = torch.cat((self.real_B2, self.real_A2), 1)
        fake_BA1 = self.fake_BA_pool.query(torch.cat((self.real_B1, self.fake_A1), 1))
        real_BA1 = torch.cat((self.real_B1, self.real_A1), 1)
        self.loss_D_B = self.backward_D_basic(self.netD_B, real_BA1, fake_BA1)+self.backward_D_basic(self.netD_B, real_BA2, fake_BA2)

    def backward_G(self):


        if self.opt.pre_train:
            self.y_L1 = Variable(torch.ones(self.real_B2.size(2), self.real_B2.size(3))).to(self.device)# y_l1 is for calculating the CosineSimilarity below
            self.loss_L1_A1 = self.CosineSimilarity(self.fake_B1, self.real_B1, self.y_L1)
            self.loss_par = self.criterionL1(self.fake_par, self.real_par)
            self.loss_G_A = self.criterionGAN(self.netD_A(torch.cat((self.real_A1, self.fake_B1), 1)), True)
            self.loss_G_B1 =0
            self.loss_G_B2 =0
            self.loss_L1_B1 = 0
            self.loss_L1_B2 = 0
            self.loss_D_B = 0
            self.loss_N=self.loss_G_A + self.loss_L1_A1 * self.opt.lambda_c
            self.loss_I=self.loss_L1_B1 *self.opt.alpha_i1+ self.loss_G_B1
            self.loss_I2 =self.loss_L1_B2 * self.opt.alpha_i1 + self.loss_G_B2
            self.loss_G = self.loss_par * self.opt.gamma_p+self.loss_N + self.loss_I + self.loss_I2
        else:
            self.y_L1 = Variable(torch.ones(self.real_B2.size(2), self.real_B2.size(3))).to(self.device)
            self.loss_L1_A1 = self.CosineSimilarity(self.fake_B1, self.real_B1,self.y_L1)
            self.loss_par = self.criterionL1(self.fake_par, self.real_par)
            self.loss_G_A = self.criterionGAN(self.netD_A(torch.cat((self.real_A1, self.fake_B1), 1)), True)
            self.loss_G_B1 = self.criterionGAN(self.netD_B(torch.cat((self.real_B1, self.fake_A1), 1)), True)
            self.loss_G_B2 = self.criterionGAN(self.netD_B(torch.cat((self.real_B2, self.fake_A2), 1)), True)
            self.loss_L1_B1 = self.criterionL1(self.fake_A1, self.real_A1)*10
            self.loss_L1_B2 = self.criterionL1(self.fake_A2, self.real_A2)*10
            self.loss_N=self.loss_G_A + self.loss_L1_A1 * self.opt.lambda_c
            self.loss_I=self.loss_L1_B1 *self.opt.alpha_i1+ self.loss_G_B1
            self.loss_I2 =self.loss_L1_B2 * self.opt.alpha_i1 + self.loss_G_B2
            self.loss_G = self.loss_par * self.opt.gamma_p+self.loss_N + self.loss_I + self.loss_I2

        self.loss_G.backward()



    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        if not self.opt.pre_train:
            self.backward_D_B()
        self.optimizer_D.step()



