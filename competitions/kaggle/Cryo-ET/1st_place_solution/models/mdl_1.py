import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Beta
from monai.networks.nets.flexible_unet import SegmentationHead, UNetDecoder, FLEXUNET_BACKBONE


class PatchedUNetDecoder(UNetDecoder):
    
    """add functionality to output all feature maps"""
    
    def forward(self, features: list[torch.Tensor], skip_connect: int = 4):
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        out = []
        x = features[0]
        out += [x]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)
            out += [x]
        return out


class FlexibleUNet(nn.Module):
    """
    A flexible implementation of UNet-like encoder-decoder architecture. 
    
    (Adjusted to support PatchDecoder and multi segmentation heads)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: str | tuple = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: str | tuple = ("relu", {"inplace": True}),
        dropout: float | tuple = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
    ) -> None:
        """
        A flexible implement of UNet, in which the backbone/encoder can be replaced with
        any efficient or residual network. Currently the input must have a 2 or 3 spatial dimension
        and the spatial size of each dimension must be a multiple of 32 if is_pad parameter
        is False.
        Please notice each output of backbone must be 2x downsample in spatial dimension
        of last output. For example, if given a 512x256 2D image and a backbone with 4 outputs.
        Spatial size of each encoder output should be 256x128, 128x64, 64x32 and 32x16.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            backbone: name of backbones to initialize, only support efficientnet and resnet right now,
                can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2, resnet10, ..., resnet200].
            pretrained: whether to initialize pretrained weights. ImageNet weights are available for efficient networks
                if spatial_dims=2 and batch norm is used. MedicalNet weights are available for residual networks
                if spatial_dims=3 and in_channels=1. Default to False.
            decoder_channels: number of output channels for all feature maps in decoder.
                `len(decoder_channels)` should equal to `len(encoder_channels) - 1`,default
                to (256, 128, 64, 32, 16).
            spatial_dims: number of spatial dimensions, default to 2.
            norm: normalization type and arguments, default to ("batch", {"eps": 1e-3,
                "momentum": 0.1}).
            act: activation type and arguments, default to ("relu", {"inplace": True}).
            dropout: dropout ratio, default to 0.0.
            decoder_bias: whether to have a bias term in decoder's convolution blocks.
            upsample: upsampling mode, available options are``"deconv"``, ``"pixelshuffle"``,
                ``"nontrainable"``.
            pre_conv:a conv block applied before upsampling. Only used in the "nontrainable" or
                "pixelshuffle" mode, default to `default`.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            is_pad: whether to pad upsampling features to fit features from encoder. Default to True.
                If this parameter is set to "True", the spatial dim of network input can be arbitrary
                size, which is not supported by TensorRT. Otherwise, it must be a multiple of 32.
        """
        super().__init__()

        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid model_name {backbone} found, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        encoder = FLEXUNET_BACKBONE.register_dict[backbone]
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_parameters = encoder["parameter"]
        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError("The backbone init method must have spatial_dims, in_channels and pretrained parameters.")
        encoder_feature_num = encoder["feature_number"]
        if encoder_feature_num > 5:
            raise ValueError("Flexible unet can only accept no more than 5 encoder feature maps.")

        decoder_channels = decoder_channels[:encoder_feature_num]
        self.skip_connect = encoder_feature_num - 1
        encoder_parameters.update({"spatial_dims": spatial_dims, "in_channels": in_channels, "pretrained": pretrained})
        encoder_channels = tuple([in_channels] + list(encoder["feature_channel"]))
        encoder_type = encoder["type"]
        self.encoder = encoder_type(**encoder_parameters)
        print(decoder_channels)
        
        
        
        self.decoder = PatchedUNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=pre_conv,
            align_corners=None,
            is_pad=is_pad,
        )
        self.segmentation_heads = nn.ModuleList([SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            act=None,
        ) for decoder_channel in decoder_channels[:-1]])

    def forward(self, inputs: torch.Tensor):

        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)[1:-1]
        x_seg = [self.segmentation_heads[i](decoder_out[i]) for i in range(len(decoder_out))]

        return x_seg



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])





class DenseCrossEntropy(nn.Module):
    def __init__(self, class_weights=None):
        super(DenseCrossEntropy, self).__init__()

        self.class_weights = class_weights
    
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float)

        loss = -logprobs * target
        
        class_losses = loss.mean((0,2,3,4))
        if self.class_weights is not None:
            loss = (class_losses * self.class_weights.to(class_losses.device)).sum() #/ class_weights.sum() 
        else:
            
            loss = class_losses.sum()
        return loss, class_losses

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))
        
        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]
                
        if Z:
            return X, Y, Z

        return X, Y
    
def to_ce_target(y):
    # bs, c, h, w, d
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y,y_bg], 1)
    y = y / y.sum(1, keepdim=True)
    return y

class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes
        self.classes = cfg.classes
        
        self.backbone = FlexibleUNet(**cfg.backbone_args)


        self.mixup = Mixup(cfg.mixup_beta)
        
        print(f'Net parameters: {human_format(count_parameters(self))}')
        self.lvl_weights = torch.from_numpy(cfg.lvl_weights)
        self.loss_fn = DenseCrossEntropy(class_weights=torch.from_numpy(cfg.class_weights))  
           
    def forward(self, batch):

        x = batch['input']
        if "target" in batch.keys():
            y = batch["target"]
        if self.training:
            if torch.rand(1)[0] < self.cfg.mixup_p:
                x, y = self.mixup(x,y)
        out = self.backbone(x)
        
        

        outputs = {}

        if "target" in batch.keys():
            ys = [F.adaptive_max_pool3d(y, item.shape[-3:])  for item in out]
            losses = torch.stack([self.loss_fn(out[i], to_ce_target(ys[i]))[0] for i in range(len(out))])
            lvl_weights = self.lvl_weights.to(losses.device)
            loss = (losses * lvl_weights).sum() / lvl_weights.sum()
            outputs['loss'] = loss
        if not self.training:
            outputs["logits"] = out[-1]
            if 'location' in batch:
                outputs["location"] = batch['location']

        return outputs


class TestNet(nn.Module):

    def __init__(self, **backbone_args):
        super(TestNet, self).__init__() 
        
        self.backbone = FlexibleUNet(**backbone_args)
        
        
    def forward(self, x):
        #x shape is bs, c, h, w, d
        out = self.backbone(x)
        #out shape is bs, 7, h//2, w//2, d//2
        logits = out[-1] # for heatmap do softmax + reorder classes .softmax(1)[:,[0,2,3,4,5,1]]
        return logits
    
    
# import torch
# import monai.transforms as mt
# import zarr
# import numpy as np

# def preprocess_img(zarr_fn, transforms):
#     img = np.array(zarr.open(zarr_fn)[0]).transpose(2,1,0)
#     img = transforms({'image':img})['image']
#     return img

# backbone_args = dict(spatial_dims=3,    
#                          in_channels=1,
#                          out_channels=6,
#                          backbone='resnet34',
#                          pretrained=False)

# net = TestNet(**backbone_args)
# sd = torch.load(CKPT_FILE)['model']
# net.eval().cuda()
# net.load_state_dict(sd)

# static_transforms = mt.Compose([mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),mt.NormalizeIntensityd(keys="image"),])
# zarr_fn = '/mount/cryo/data/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr'
# img = preprocess_img(zarr_fn,static_transforms) # torch.Size([1, 630, 630, 184])

# patch = img[None, :, :96, :96, :96] # torch.Size([1, ,1 96, 96, 96])

# logits = net(patch.cuda())
# proba_heatmap = logits.softmax(1)[:,[0,2,3,4,5,1]]