import torch
from torch import nn
from networks import common_freq as common


class TwoBranch(nn.Module):
    def __init__(self, args):
        super(TwoBranch, self).__init__()

        num_group = 4
        num_every_group = args.base_num_every_group
        self.args = args

        self.init_T2_frq_branch(args)
        self.init_T2_spa_branch(args, num_every_group)
        self.init_T2_fre_spa_fusion(args)

        self.init_T1_frq_branch(args)
        self.init_T1_spa_branch(args, num_every_group)

        self.init_modality_fre_fusion(args)
        self.init_modality_spa_fusion(args)
        

    def init_T2_frq_branch(self, args):
        ### T2frequency branch
        modules_head_fre = [common.ConvBNReLU2D(1, out_channels=args.num_features,
                                            kernel_size=3, padding=1, act=args.act)]
        self.head_fre = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [common.DownSample(args.num_features, False, False),
                            common.FreBlock9(args.num_features, args)
                        ]

        self.down1_fre = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_down2_fre = [common.DownSample(args.num_features, False, False),
                        common.FreBlock9(args.num_features, args)
                        ]
        self.down2_fre = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_down3_fre = [common.DownSample(args.num_features, False, False),
                        common.FreBlock9(args.num_features, args)
                        ]
        self.down3_fre = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_neck_fre = [common.FreBlock9(args.num_features, args)
                        ]
        self.neck_fre = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_up1_fre = [common.UpSampler(2, args.num_features),
                        common.FreBlock9(args.num_features, args)
                        ]
        self.up1_fre = nn.Sequential(*modules_up1_fre)
        self.up1_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_up2_fre = [common.UpSampler(2, args.num_features),
                    common.FreBlock9(args.num_features, args)
                        ]
        self.up2_fre = nn.Sequential(*modules_up2_fre)
        self.up2_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_up3_fre = [common.UpSampler(2, args.num_features),
                    common.FreBlock9(args.num_features, args)
                        ]
        self.up3_fre = nn.Sequential(*modules_up3_fre)
        self.up3_fre_mo = nn.Sequential(common.FreBlock9(args.num_features, args))

        # define tail module
        modules_tail_fre = [
            common.ConvBNReLU2D(args.num_features, out_channels=args.num_channels, kernel_size=3, padding=1,
                        act=args.act)]
        self.tail_fre = nn.Sequential(*modules_tail_fre)

    def init_T2_spa_branch(self, args, num_every_group):
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(1, out_channels=args.num_features,
                                            kernel_size=3, padding=1, act=args.act)]
        self.head = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1 = nn.Sequential(*modules_down1)


        self.down1_mo = nn.Sequential(common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_down2 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2 = nn.Sequential(*modules_down2)

        self.down2_mo = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_down3 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3 = nn.Sequential(*modules_down3)
        self.down3_mo = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_neck = [common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.neck = nn.Sequential(*modules_neck)

        self.neck_mo = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_up1 = [common.UpSampler(2, args.num_features),
                       common.ResidualGroup(
                           args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up1 = nn.Sequential(*modules_up1)

        self.up1_mo = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_up2 = [common.UpSampler(2, args.num_features),
                       common.ResidualGroup(
                           args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up2 = nn.Sequential(*modules_up2)
        self.up2_mo = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))


        modules_up3 = [common.UpSampler(2, args.num_features),
                       common.ResidualGroup(
                           args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up3 = nn.Sequential(*modules_up3)
        self.up3_mo = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        # define tail module
        modules_tail = [
            common.ConvBNReLU2D(args.num_features, out_channels=args.num_channels, kernel_size=3, padding=1,
                         act=args.act)]

        self.tail = nn.Sequential(*modules_tail)

    def init_T2_fre_spa_fusion(self, args):
        ### T2 frq & spa fusion part
        conv_fuse = []
        for i in range(14):
            conv_fuse.append(common.FuseBlock7(args.num_features))
        self.conv_fuse = nn.Sequential(*conv_fuse)

    def init_T1_frq_branch(self, args):
        ### T2frequency branch
        modules_head_fre = [common.ConvBNReLU2D(1, out_channels=args.num_features,
                                            kernel_size=3, padding=1, act=args.act)]
        self.head_fre_T1 = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [common.DownSample(args.num_features, False, False),
                            common.FreBlock9(args.num_features, args)
                        ]

        self.down1_fre_T1 = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_down2_fre = [common.DownSample(args.num_features, False, False),
                        common.FreBlock9(args.num_features, args)
                        ]
        self.down2_fre_T1 = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_down3_fre = [common.DownSample(args.num_features, False, False),
                        common.FreBlock9(args.num_features, args)
                        ]
        self.down3_fre_T1 = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.num_features, args))

        modules_neck_fre = [common.FreBlock9(args.num_features, args)
                        ]
        self.neck_fre_T1 = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo_T1 = nn.Sequential(common.FreBlock9(args.num_features, args))

    def init_T1_spa_branch(self, args, num_every_group):
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(1, out_channels=args.num_features,
                                            kernel_size=3, padding=1, act=args.act)]
        self.head_T1 = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1_T1 = nn.Sequential(*modules_down1)


        self.down1_mo_T1 = nn.Sequential(common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_down2 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2_T1 = nn.Sequential(*modules_down2)

        self.down2_mo_T1 = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_down3 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3_T1 = nn.Sequential(*modules_down3)
        self.down3_mo_T1 = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

        modules_neck = [common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.neck_T1 = nn.Sequential(*modules_neck)

        self.neck_mo_T1 = nn.Sequential(common.ResidualGroup(
            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None))

    
    def init_modality_fre_fusion(self, args):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(common.Modality_FuseBlock6(args.num_features))
        self.conv_fuse_fre = nn.Sequential(*conv_fuse)

    def init_modality_spa_fusion(self, args):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(common.Modality_FuseBlock6(args.num_features))
        self.conv_fuse_spa = nn.Sequential(*conv_fuse)

    def forward(self, main, aux):
        #### T1 fre encoder
        t1_fre = self.head_fre_T1(aux) # 128

        down1_fre_t1 = self.down1_fre_T1(t1_fre)# 64
        down1_fre_mo_t1 = self.down1_fre_mo_T1(down1_fre_t1)

        down2_fre_t1 = self.down2_fre_T1(down1_fre_mo_t1) # 32
        down2_fre_mo_t1 = self.down2_fre_mo_T1(down2_fre_t1)

        down3_fre_t1 = self.down3_fre_T1(down2_fre_mo_t1) # 16
        down3_fre_mo_t1 = self.down3_fre_mo_T1(down3_fre_t1)

        neck_fre_t1 = self.neck_fre_T1(down3_fre_mo_t1) # 16
        neck_fre_mo_t1 = self.neck_fre_mo_T1(neck_fre_t1)


        #### T2 fre encoder and T1 & T2 fre fusion
        x_fre = self.head_fre(main) # 128
        x_fre_fuse = self.conv_fuse_fre[0](t1_fre, x_fre)

        down1_fre = self.down1_fre(x_fre_fuse)# 64
        down1_fre_mo = self.down1_fre_mo(down1_fre)
        down1_fre_mo_fuse = self.conv_fuse_fre[1](down1_fre_mo_t1, down1_fre_mo)

        down2_fre = self.down2_fre(down1_fre_mo_fuse) # 32
        down2_fre_mo = self.down2_fre_mo(down2_fre)
        down2_fre_mo_fuse = self.conv_fuse_fre[2](down2_fre_mo_t1, down2_fre_mo)

        down3_fre = self.down3_fre(down2_fre_mo_fuse) # 16
        down3_fre_mo = self.down3_fre_mo(down3_fre)
        down3_fre_mo_fuse = self.conv_fuse_fre[3](down3_fre_mo_t1, down3_fre_mo)

        neck_fre = self.neck_fre(down3_fre_mo_fuse) # 16
        neck_fre_mo = self.neck_fre_mo(neck_fre)
        neck_fre_mo_fuse = self.conv_fuse_fre[4](neck_fre_mo_t1, neck_fre_mo)


        #### T2 fre decoder
        neck_fre_mo = neck_fre_mo_fuse + down3_fre_mo_fuse

        up1_fre = self.up1_fre(neck_fre_mo) # 32
        up1_fre_mo = self.up1_fre_mo(up1_fre)
        up1_fre_mo = up1_fre_mo + down2_fre_mo_fuse

        up2_fre = self.up2_fre(up1_fre_mo) # 64
        up2_fre_mo = self.up2_fre_mo(up2_fre)
        up2_fre_mo = up2_fre_mo + down1_fre_mo_fuse

        up3_fre = self.up3_fre(up2_fre_mo) # 128
        up3_fre_mo = self.up3_fre_mo(up3_fre)
        up3_fre_mo = up3_fre_mo + x_fre_fuse

        res_fre = self.tail_fre(up3_fre_mo)

        #### T1 spa encoder
        x_t1 = self.head_T1(aux)  # 128

        down1_t1 = self.down1_T1(x_t1) # 64
        down1_mo_t1 = self.down1_mo_T1(down1_t1)

        down2_t1 = self.down2_T1(down1_mo_t1) # 32
        down2_mo_t1 = self.down2_mo_T1(down2_t1)  # 32

        down3_t1 = self.down3_T1(down2_mo_t1) # 16
        down3_mo_t1 = self.down3_mo_T1(down3_t1)  # 16

        neck_t1 = self.neck_T1(down3_mo_t1) # 16
        neck_mo_t1 = self.neck_mo_T1(neck_t1)

        #### T2 spa encoder and fusion
        x = self.head(main)  # 128
        
        x_fuse = self.conv_fuse_spa[0](x_t1, x)
        down1 = self.down1(x_fuse) # 64
        down1_fuse = self.conv_fuse[0](down1_fre, down1)
        down1_mo = self.down1_mo(down1_fuse)
        down1_fuse_mo = self.conv_fuse[1](down1_fre_mo_fuse, down1_mo)

        down1_fuse_mo_fuse = self.conv_fuse_spa[1](down1_mo_t1, down1_fuse_mo)
        down2 = self.down2(down1_fuse_mo_fuse) # 32
        down2_fuse = self.conv_fuse[2](down2_fre, down2)
        down2_mo = self.down2_mo(down2_fuse)  # 32
        down2_fuse_mo = self.conv_fuse[3](down2_fre_mo, down2_mo)

        down2_fuse_mo_fuse = self.conv_fuse_spa[2](down2_mo_t1, down2_fuse_mo)
        down3 = self.down3(down2_fuse_mo_fuse) # 16
        down3_fuse = self.conv_fuse[4](down3_fre, down3)
        down3_mo = self.down3_mo(down3_fuse)  # 16
        down3_fuse_mo = self.conv_fuse[5](down3_fre_mo, down3_mo)

        down3_fuse_mo_fuse = self.conv_fuse_spa[3](down3_mo_t1, down3_fuse_mo)
        neck = self.neck(down3_fuse_mo_fuse) # 16
        neck_fuse = self.conv_fuse[6](neck_fre, neck)
        neck_mo = self.neck_mo(neck_fuse)
        neck_mo = neck_mo + down3_mo
        neck_fuse_mo = self.conv_fuse[7](neck_fre_mo, neck_mo)

        neck_fuse_mo_fuse = self.conv_fuse_spa[4](neck_mo_t1, neck_fuse_mo)
        #### T2 spa decoder
        up1 = self.up1(neck_fuse_mo_fuse) # 32
        up1_fuse = self.conv_fuse[8](up1_fre, up1)
        up1_mo = self.up1_mo(up1_fuse)
        up1_mo = up1_mo + down2_mo
        up1_fuse_mo = self.conv_fuse[9](up1_fre_mo, up1_mo)

        up2 = self.up2(up1_fuse_mo) # 64
        up2_fuse = self.conv_fuse[10](up2_fre, up2)
        up2_mo = self.up2_mo(up2_fuse)
        up2_mo = up2_mo + down1_mo
        up2_fuse_mo = self.conv_fuse[11](up2_fre_mo, up2_mo)

        up3 = self.up3(up2_fuse_mo) # 128

        up3_fuse = self.conv_fuse[12](up3_fre, up3)
        up3_mo = self.up3_mo(up3_fuse)

        up3_mo = up3_mo + x
        up3_fuse_mo = self.conv_fuse[13](up3_fre_mo, up3_mo)
        # import matplotlib.pyplot as plt
        # plt.axis('off')
        # plt.imshow((255*up3_fre_mo[0].detach().cpu().numpy()[0]))
        # plt.savefig('up3_fre_mo.jpg', bbox_inches='tight', pad_inches=0)
        # plt.clf() 

        # plt.axis('off')
        # plt.imshow((255*up3_mo[0].detach().cpu().numpy()[0]))
        # plt.savefig('up3_mo.jpg', bbox_inches='tight', pad_inches=0)
        # plt.clf() 

        # plt.axis('off')
        # plt.imshow((255*up3_fuse_mo[0].detach().cpu().numpy()[0]))
        # plt.savefig('up3_fuse_mo.jpg', bbox_inches='tight', pad_inches=0)
        # plt.clf() 
        # breakpoint()

        res = self.tail(up3_fuse_mo)

        return {'img_out': res + main, 'img_fre': res_fre + main}

def make_model(args):
    return TwoBranch(args)

