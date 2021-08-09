import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.model import UnetDecoder, SegmentationHead

from torch.nn.parameter import Parameter
from torch.nn import functional as F

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)

        self.eps = eps

    def forward(self, x):
        #print(x.shape)
        #print(self.p)
        ret = gem(x, p=self.p, eps=self.eps)
        return ret.squeeze(2).squeeze(2)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=None)

    def forward(self, preds, targets):
       
        bce_loss = self.loss_fct(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5, (1 - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        loss = loss.mean()

        return loss

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.n_classes = len(cfg.label_cols)
        in_chans = 1
        if cfg.backbone == 'tf_efficientnet_l2_ns':
            backbone_out = 1376
            encoder_channels = (1,72, 104, 176, 480, 1376)
        elif 'tf_efficientnet_b8' in cfg.backbone:
            backbone_out = 704
            encoder_channels = (1,32, 56, 88, 248, 704)
        elif "efficientnet" in cfg.backbone:
            backbone_out = 640
            encoder_channels = (1,32, 48, 80, 224, 640)
        elif "resnest" in cfg.backbone:
            backbone_out = 2048
            encoder_channels = (1,128, 256, 512, 1024, 2048)
        else:
            backbone_out = 2048
            encoder_channels = (1,64, 256, 512, 1024, 2048)


        
        self.encoder = encoder = timm.create_model(cfg.backbone, pretrained=cfg.pretrained,features_only=True,in_chans=1)
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=cfg.seg_dim,
            activation=None,
            kernel_size=3,
        )
        

        if cfg.pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif cfg.pool == 'gem':
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        self.head_in_units = backbone_out
        self.head = nn.Linear(self.head_in_units, self.n_classes)
        
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1,1,1,1,1,1,1,1,1,1,1]).to(cfg.device), reduction="mean")
        self.bce_seg = nn.BCEWithLogitsLoss(reduction="mean")
        self.w = cfg.seg_weight

        self.cfg = cfg

        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.pretrained_weights is not None:
            self.load_state_dict(torch.load(cfg.pretrained_weights, map_location='cpu')['model'], strict=True)
            print('weights loaded from',cfg.pretrained_weights)

    def forward(self, batch):

        x_in = batch['input']
        enc_out = self.encoder(x_in)

        #print(enc_out[-1].shape)
        
            
        
        x = enc_out[-1]
        if self.cfg.pool == "gem":
            x = self.pool(x)
        else:
            x = self.pool(x)[:,:,0,0]
        x = self.dropout(x)
        logits = self.head(x)
    
        if self.cfg.calc_loss:
            cls_loss = self.bce_cls(logits,batch['target'])
        else:
            cls_loss = torch.Tensor(0).to(x.device)

        if batch['is_annotated'].sum() > 0 and not self.cfg.ignore_seg:
            ia = batch['is_annotated']>0


            decoder_out = self.decoder(*[x_in] + enc_out)
            decoder_out = self.dropout(decoder_out)    

            x_seg = self.segmentation_head(decoder_out)
            
            seg_loss = self.bce_seg(x_seg[ia].permute(0,2,3,1), batch['mask'][ia].permute(0,2,3,1))
            
        else:
            seg_loss = torch.zeros_like(cls_loss)

        loss =  cls_loss + self.w * seg_loss
    
        if self.training == True:
            return {'logits': logits,
                    'loss':loss,
                'cls_loss':cls_loss,
                'seg_loss':seg_loss}
        else:
            return {'logits': logits,
               # 'segments': x_seg.detach(),
                'loss':loss,
                'cls_loss':cls_loss,
                'seg_loss':seg_loss}
