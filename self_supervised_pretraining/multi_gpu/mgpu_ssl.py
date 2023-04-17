# --------------------------------------------------------
# Put MONAI License File here
# --------------------------------------------------------
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from torch.utils.data import DataLoader, DistributedSampler
from .utils.utils import collate_fn, reduce_tensor, save_checkpoint, TensorboardLogger

from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)

from monai.networks.nets import ViTAutoEnc
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
)

from monai.losses import ContrastiveLoss
from torch.nn import L1Loss


def parse_option():
    parser = argparse.ArgumentParser('ViT Self-Supervised Learning', add_help=False)

    # Set Paths for running SSL training
    parser.add_argument('--data_root', default="/workspace/datasets/tcia/", type=str, help='path to data root')
    parser.add_argument('--json_path', default="./datalists/tcia/dataset_split.json", type=str,
                        help='Json file path for list of data samples')
    parser.add_argument('--logdir_path', default="/to/be/defined", type=str, help='output log directory')

    # Distributed Training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # DL Training Hyper-parameters
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help="batch size for single GPU")

    ######TODO: LEGACY HYPER-PARAMS, LEVERAGE & THEN DELETE ######
    parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--num_classes', default=0, type=int, help='number of input channels')
    parser.add_argument('--window_size', default=(7, 7, 7), type=tuple, help='window size')
    parser.add_argument('--patch_size', default=(2, 2, 2), type=tuple, help='window size')
    parser.add_argument('--mask_patch_size', default=16, type=int, help='window size')
    parser.add_argument('--img_size', default=96, type=int, help='image size')
    parser.add_argument('--num_heads', default=[3, 6, 12, 24], type=list, help='number of heads')
    parser.add_argument('--depths', default=[2, 2, 2, 2], type=list, help='number of depths')
    parser.add_argument('--embed_dim', default=48, type=int, help='embedding dimention')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio')
    parser.add_argument('--drop_rate', default=0.0, type=float, help='drop rate')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='attention drop rate')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path rate')
    parser.add_argument('--layer_decay', default=1.0, type=float, help='layer decay')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--mask_ratio', default=0.6, type=float, help='drop path rate')

    parser.add_argument('--optimizer_name', type=str, default='adamw', help='optimizer name')
    parser.add_argument('--momentum', default=0.9, type=float, help='optimizer momentum')
    parser.add_argument('--base_lr', default=5e-4, type=float, help='base learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='optimizer betas')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps')
    parser.add_argument('--decoder', type=str, default='upsample', help='decoder type')
    parser.add_argument('--loss_type', type=str, default='mask_only', help='decoder type')

    parser.add_argument('--amp_opt_level', type=str, default='O1', help='amp opt level')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='number of epochs')
    parser.add_argument('--warmpup_epoch', default=20, type=int, help='warmup epoch')
    parser.add_argument('--decay_epoch', default=30, type=int, help='warmup epoch')
    parser.add_argument('--save_freq', default=1, type=int, help='saving frequency')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--accumulate_step', default=0, type=int, help='accumulation step')
    parser.add_argument('--clip_grad', default=1, type=int, help='saving frequency')
    parser.add_argument('--seed', default=0, type=int, help='seed')

    parser.add_argument('--lr_scheduler_name', type=str, default='cosine', help='learning rate scheduler name')
    parser.add_argument('--min_lr', default=5e-6, type=float, help='min learning rate')
    parser.add_argument('--warmup_lr', default=5e-7, type=float, help='warmup lr')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='lr decay rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='lr gamma')
    parser.add_argument('--auto_resume', default=True, type=bool)
    parser.add_argument('--iso_spacing', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--model_type', type=str, default='swin', help='model type')
    parser.add_argument('--cache_dataset', default=True, action='store_true')
    parser.add_argument('--thread_loader', default=True, action='store_true')
    parser.add_argument('--onlycovid', default=False, action='store_true')
    parser.add_argument('--only_ten_k', default=False, action='store_true')

    parser.add_argument('--cache_rate', default=0.5, type=float, help='drop path rate')
    parser.add_argument('--sw_batch_size', default=1, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')

    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--deterministic', help='set seed for deterministic training', action='store_true')
    parser.add_argument('--use_grad_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")

    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--decoder_off', action='store_true')
    parser.add_argument('--encoder_off', action='store_true')
    parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str,
                        help='pretrained checkpoint directory')

    # parser.add_argument('--tag', help='tag of experiment')

    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
    # parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--out_channels', default=14, type=int, help='number of output channels')
    # parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--feature_size', default=48, type=int, help='feature size')
    parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
    parser.add_argument('--choice', default="mae", type=str, help='choice')
    parser.add_argument('--inf', default="notsim", type=str, help='choice')

    parser.add_argument('--variance', default=0.1, type=float, help='')
    parser.add_argument('--interpolate', default=4, type=float, help='')
    parser.add_argument('--temperature', default=0.07, type=float, help='drop path rate')
    parser.add_argument('--mm_con', default=0.02, type=float, help='drop path rate')

    args = parser.parse_args()

    return args


def main(args):
    json_path = args.json_path
    data_root = args.data_root
    logdir_path = args.logdir_path

    # Define training transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
            RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
            # Please note that that if image, image_2 are called via the same transform call because of the determinism
            # they will get augmented the exact same way which is not the required case here, hence two calls are made
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
        ]
    )

    # Build the data loader
    train_list = load_decathlon_datalist(data_list_file_path=json_path,
                                         is_segmentation=False,
                                         data_list_key="training",
                                         base_dir=data_root)

    val_list = load_decathlon_datalist(data_list_file_path=json_path,
                                       is_segmentation=False,
                                       data_list_key="validation",
                                       base_dir=data_root)

    # TODO Delete the below print statements
    print('List of training samples: {}'.format(train_list))
    print('List of validation samples: {}'.format(val_list))

    print('Total training data are {} and validation data are {}'.format(len(train_list), len(val_list)))

    train_dataset = CacheDataset(data=train_list, transform=train_transforms, cache_rate=1.0, num_workers=8)
    val_dataset = CacheDataset(data=val_list, transform=train_transforms, cache_rate=1.0, num_workers=8)

    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=dist.get_world_size(),
                                       rank=dist.get_rank(),
                                       shuffle=True)

    val_sampler = DistributedSampler(val_dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler,
                            num_workers=8,
                            pin_memory=True,
                            collate_fn=collate_fn)

    # Data Loaders Built

    # Define the model, losses and optimizer
    model = ViTAutoEnc(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        pos_embed="conv",
        hidden_size=768,
        mlp_dim=3072,
    )

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.base_lr,
                                 weight_decay=args.reg_weight)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)
    model_without_ddp = model.module

    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    loss_funcs = [recon_loss, contrastive_loss]

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))

    print('Training Begins ...')

    start_time = time.time()
    val_loss_best = 1e9

    for epoch in range(args.epochs):
        train_loss_avg, train_l1_avg, train_cl_avg = train_one_epoch(args=args,
                                                                     model=model,
                                                                     data_loader=train_loader,
                                                                     optimizer=optimizer,
                                                                     epoch=epoch,
                                                                     loss_functions=loss_funcs)

        val_loss_avg = validate(data_loader=val_loader,
                                model=model,
                                loss_functions=loss_funcs)

        if dist.get_rank() == 0:
            # log_writer.update(loss_val_avg=val_loss_avg, head="perf", step=epoch)
            log_writer.update(loss_val_L1=val_loss_avg, head="perf", step=epoch)
            # log_writer.update(loss_val_MM=val_MM_avg, head="perf", step=epoch)
            log_writer.update(loss_train_avg=train_loss_avg, head="perf", step=epoch)
            log_writer.update(loss_train_L1=train_l1_avg, head="perf", step=epoch)
            log_writer.update(loss_train_MM=train_cl_avg, head="perf", step=epoch)
            if val_loss_avg <= val_loss_best:
                save_checkpoint(args, epoch, model_without_ddp, 0., optimizer, best_model=True)

        if dist.get_rank() == 0 and (epoch == (args.epoch - 1)):
            save_checkpoint(args, epoch, model_without_ddp, 0., optimizer, best_model=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(args,
                    model,
                    data_loader,
                    optimizer,
                    epoch,
                    loss_functions):
    model.train()
    # optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()
    loss_meter = AverageMeter()

    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, batch_data in enumerate(data_loader):
        inputs, inputs_2, gt_input = (
            batch_data["image"].cuda(non_blocking=True),
            batch_data["image_2"].cuda(non_blocking=True),
            batch_data["gt_image"].cuda(non_blocking=True),
        )

        optimizer.zero_grad()

        outputs_v1, hidden_v1 = model(inputs)
        outputs_v2, hidden_v2 = model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss = loss_functions[0](outputs_v1, gt_input)
        cl_loss = loss_functions[1](flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        total_loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        r_loss_t = reduce_tensor(r_loss)
        cl_loss_t = reduce_tensor(cl_loss)
        total_loss_t = reduce_tensor(total_loss)

        loss_l1_meter.update(r_loss_t.item(), inputs.size(0))
        loss_cont_meter.update(cl_loss_t.item(), inputs.size(0))
        loss_meter.update(total_loss_t.item(), inputs.size(0))

        # norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        print(
            f'Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t'
            f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            f'loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t'
            f'loss_MM {loss_cont_meter.val:.4f} ({loss_cont_meter.avg:.4f})\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
            f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg, loss_l1_meter.avg, loss_cont_meter.avg


@torch.no_grad()
def validate(data_loader, model, loss_functions):
    model.eval()
    loss_l1_meter = AverageMeter()

    for idx, batch_data in enumerate(data_loader):
        inputs, gt_input = (
            batch_data["image"].cuda(non_blocking=True),
            batch_data["gt_image"].cuda(non_blocking=True),
        )

        outputs, outputs_v2 = model(inputs)
        val_loss = loss_functions[0](outputs, gt_input)
        loss = reduce_tensor(val_loss)
        loss_l1_meter.update(loss.item(), inputs.size(0))

        print(
            f'Test: [{idx}/{len(data_loader)}]\t'
            f'Loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t')
    print(f' * Val Loss {loss_l1_meter.avg:.3f}')

    return loss_l1_meter.avg

if __name__ == '__main__':
    args = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # rank = int(os.environ["RANK"])
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = args.seed + dist.get_rank()

    if args.deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)

    cudnn.benchmark = True

    if dist.get_rank() == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        if not os.path.exists(args.logdir_path):
            os.makedirs(args.logdir_path, exist_ok=True)

    if dist.get_rank() == 0:
        log_writer = TensorboardLogger(log_dir=args.logdir_path)

    main(args)