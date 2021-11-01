
#dependencies to load .tiff WSI image
#pip install tifffile imagecodecs

import collections.abc
import os
import json
import time
import shutil
import argparse

import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
# import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp

import tifffile #pip install it
from monai import transforms, data
from monai.data.image_reader import WSIReader

from monai.data.image_reader import WSIReader
from monai.networks.nets import MilMode, MILModel #to be added
from monai.apps.pathology.transforms import TileOnGridd #to be added

def parse_args():

    parser = argparse.ArgumentParser(description='MIL example')

    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--max_tiles', default=44, type=int)
    parser.add_argument('--tile_size', default=256, type=int)
    parser.add_argument('--data_dir', default='/PandaChallenge2020/train_images/')

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--fold', default=0, type=int)
    # parser.add_argument('--checkpoint_fromscratch', action='store_true')

    parser.add_argument('--optim_lr', default=3e-5, type=float)
    parser.add_argument('--optim_trans_lr', default=None, type=float)

    parser.add_argument('--quick', action='store_true') #distributed multi gpu
    parser.add_argument('--amp', action='store_true') #experimental
    parser.add_argument('--val_every', default=1, type=int)

    parser.add_argument('--mil_mode', default='att')
    parser.add_argument('--reg_weight', default=0, type=float)
    parser.add_argument('--optim_trans_reg_weight', default=0, type=float)
    parser.add_argument('--validation_only', action='store_true')

    ###for multigpu
    parser.add_argument('--distributed', action='store_true') #distributed multi gpu
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,  help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--workers', default=2, type=int)

    parser.add_argument('--tf', action='store_true')
    parser.add_argument('--normback', action='store_true')
    parser.add_argument('--collate', action='store_true')


    args = parser.parse_args()

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    return args



#can be removed
class NormalizeBackground():

    def __init__( self, keys=['image'], tile_background_val=255):
        super().__init__()
        self.keys = keys
        self.tile_background_val=tile_background_val

    def __call__(self, data):

        if self.tile_background_val >0:
            d = dict(data)

            for key in self.keys:
                img = d[key]
                # print('NormalizeBackground', img.shape, img.dtype)
                d[key] = (self.tile_background_val - img).astype(np.float32) / float(self.tile_background_val)
        return d

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        if not np.isfinite(np.sum(val)):
            val = 0

        if n > 0:
            self.val = val
            self.sum += val * n

            self.count += n
            self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(tensor_list, out_numpy=False):
    """gathers values in tensor_list from all gpus"""
    world_size = torch.distributed.get_world_size()

    tensor_list_out = []
    for tensor in tensor_list:
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gather_list, tensor)

        if out_numpy:
            gather_list = [t.cpu().numpy() for t in gather_list] #convert to numpy

        tensor_list_out.append(gather_list)

    return tensor_list_out


def train_epoch(model, loader, optimizer, scaler, epoch, args):
    '''One train epoch over the dataset'''

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    run_loss = AverageMeter()
    run_acc = AverageMeter()

    start_time = time.time()

    for idx, batch_data in enumerate(loader):

        data, target = batch_data['image'].cuda(args.rank),  batch_data['label'].cuda(args.rank)

        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = criterion(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


        acc = (logits.sigmoid().sum(1).detach().round() == target.sum(1).round()).float().mean()

        if args.distributed:
            loss_list, acc_list = distributed_all_gather([loss, acc], out_numpy=True)
            loss = np.mean(np.stack(loss_list, axis=0))
            acc = np.mean(np.stack(acc_list, axis=0), axis=0)
        else:
            loss = loss.item()
            acc = acc.cpu().numpy()


        run_loss.update(loss, n=args.batch_size * args.world_size)
        run_acc.update(acc, n=args.batch_size * args.world_size)

        if args.rank==0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'acc: {:.4f}'.format(run_acc.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()

    return run_loss.avg, run_acc.avg


def val_epoch(model, loader, epoch, args, max_tiles=None):
    '''One validation epoch over the dataset'''

    model.eval()

    model2 = model if not args.distributed else model.module
    has_extra_outputs = (model2.mil_mode=='att_trans_pyramid')
    extra_outputs = model2.extra_outputs
    calc_head = model2.calc_head


    criterion = nn.BCEWithLogitsLoss()

    run_loss = AverageMeter()
    run_acc = AverageMeter()
    start_time = time.time()


    PREDS = []
    TARGETS = []

    with torch.no_grad():

        for idx, batch_data in enumerate(loader):

            data, target = batch_data['image'].cuda(args.rank), batch_data['label'].cuda(args.rank)

            with autocast(enabled=args.amp):

                if max_tiles is not None and data.shape[1] > max_tiles:

                    # print('data is too large', data.shape, 'max_tiles', max_tiles)
                    logits = []
                    logits2 =  []
                    sh = data.shape

                    for i in range(int(np.ceil(data.shape[1]/float(max_tiles)))):
                        data_slize = data[:,i*max_tiles:(i+1)*max_tiles]
                        # print('taking slice0', i, data_slize.shape)
                        logits_slize = model(data_slize, no_head=True)
                        # print('taking slice', i, data_slize.shape, logits_slize.shape)
                        logits.append(logits_slize)

                        if has_extra_outputs:
                            logits2.append([extra_outputs['layer1'], extra_outputs['layer2'], extra_outputs['layer3'], extra_outputs['layer4']])


                    logits = torch.cat(logits, dim=1)
                    # print('combined slice  logits', logits.shape)
                    if has_extra_outputs:
                        extra_outputs['layer1'] = torch.cat([l[0] for l in logits2], dim=1)
                        extra_outputs['layer2'] = torch.cat([l[1] for l in logits2], dim=1)
                        extra_outputs['layer3'] = torch.cat([l[2] for l in logits2], dim=1)
                        extra_outputs['layer4'] = torch.cat([l[3] for l in logits2], dim=1)

                    logits = calc_head(logits)
                    # print('final slice  logits', logits.shape)

                else: #normal
                    logits = model(data)


                loss = criterion(logits, target)


            pred = logits.sigmoid().sum(1).detach().round()
            target = target.sum(1).round()
            acc = (pred == target).float().mean()

            if args.distributed:
                loss_list, acc_list, pred_list, target_list = distributed_all_gather([loss, acc, pred, target], out_numpy=True)
                loss = np.mean(np.stack(loss_list, axis=0))
                acc = np.mean(np.stack(acc_list, axis=0), axis=0)

                PREDS.extend(pred_list)
                TARGETS.extend(target_list)

            else:
                loss = loss.item()
                acc = acc.cpu().numpy()

                PREDS.append(pred.cpu().numpy())
                TARGETS.append(target.cpu().numpy())


            run_loss.update(loss, n=args.batch_size * args.world_size)
            run_acc.update(acc, n=args.batch_size * args.world_size)

            if args.rank==0:
                print('Val epoch {}/{} {}/{}'.format(epoch, args.epochs, idx, len(loader)),
                      'loss: {:.4f}'.format(run_loss.avg),
                      'acc: {:.4f}'.format(run_acc.avg),
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()


        #Calculate QWK metric (Quadratic Weigted Kappa)
        #https://en.wikipedia.org/wiki/Cohen%27s_kappa

        PREDS = np.concatenate(PREDS)
        TARGETS = np.concatenate(TARGETS)
        qwk = cohen_kappa_score(PREDS.astype(np.float64), TARGETS.astype(np.float64), weights='quadratic')


    return run_loss.avg, run_acc.avg, qwk


def save_checkpoint(model, epoch, args, filename='model.tar', best_acc=0):
    '''Save checkpoint'''

    state_dict = model.state_dict() if not args.distributed else  model.module.state_dict()

    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }

    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)




###  can be removed when WSIReader supports tiffile
class LoadTiffd():
    """
    Load TIFF image

    Args:
        keys: keys of the corresponding items to be transformed
            Defaults to ``['image']``.
        level: page index to read from multi-page TIFF file, where 0 correspond to highest resolution
            Defaults to ``1``.

    """
    def __init__( self, keys=['image'], level=1) :
        super().__init__()
        self.keys = keys
        self.level=level


    def __call__(self, data):

        if isinstance(data, str):
            data = json.loads(data)

        d = dict(data)
        for key in self.keys:
            image_path = d[key]

            image = tifffile.imread(image_path, key=self.level)
            image = image.transpose(2,0,1) #convert to 3xWxH

            d[key] = image

        return d

class LabelEncodeIntegerGraded(transforms.Transform):
    """
    Convert to integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed
            Defaults to ``['label']``.

    """
    def __init__( self, num_classes, keys=['label']) :
        super().__init__()
        self.keys = keys
        self.num_classes=num_classes

    def __call__(self, data):

        d = dict(data)

        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.
            # lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner

            d[key] = lz

        return d


# can be removed still here for speed comparison
class SimpleTileFlip():
    """
    Random flip, transpose of 2D images in a batch.  It operates on the batch data,
    and cycles through all images in the batch to apply random flips.  The flips and transpose are views (without memory copy)

    Args:
        keys: keys of the corresponding items to be transformed
            Defaults to ``['image']``.
        p: probability of transform
            Defaults to ``0.5``.

    """
    def __init__( self, keys, p=0.5) :
        super().__init__()
        self.keys = keys
        self.p=p

    def __call__(self, data):

        d = dict(data)

        for key in self.keys:
            images = d[key]
            images2=[]

            for i in range(images.shape[0]):

                img = images[i]
                if np.random.random() > self.p: img = np.flip(img, axis=1)
                if np.random.random() > self.p: img = np.flip(img, axis=2)
                if np.random.random() > self.p: img = np.transpose(img, axes=(0,2,1))
                images2.append(img)

            d[key] = np.stack(images2, axis=0)

        return d


def main():

    args = parse_args()
    args.mil_mode=MilMode(args.mil_mode)

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.optim_lr = ngpus_per_node * args.optim_lr / 2 #heuristic to scale up learning rate in multigpu setup
        args.world_size = ngpus_per_node * args.world_size

        print('Multi-gpu', ngpus_per_node, 'rescaled lr', args.optim_lr)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)

from torch.utils.data.dataloader import default_collate
def list_data_collate(batch: collections.abc.Sequence):
    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def main_worker(gpu, args):

    args.gpu = gpu


    if args.distributed:
        args.rank = args.rank * torch.cuda.device_count() + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    print(args.rank, ' gpu', args.gpu)

    torch.cuda.set_device(args.gpu) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    if args.rank==0:
        print('Batch size is:', args.batch_size, 'epochs', args.epochs)

    #############

    with open('panda_training_fold0.json') as json_file:
        training_list = json.load(json_file)
        for d in training_list:
            d['image'] = os.path.join(args.data_dir, d['image'])

    with open('panda_validation_fold0.json') as json_file:
        validation_list = json.load(json_file)
        for d in validation_list:
            d['image'] = os.path.join(args.data_dir, d['image'])


    load_image_transform =  LoadTiffd(keys=["image"]) if args.tf else transforms.LoadImageD(keys=["image"], reader=WSIReader, backend="cuCIM", dtype=np.uint8, level=1)
    normalize_image_transform = NormalizeBackground() if args.normback else transforms.ScaleIntensityRangeD(keys=["image"], a_min=np.float32(255), a_max=np.float32(0), b_min=np.float32(0), b_max=np.float32(1), clip=False)


    if args.collate:
        train_transform = transforms.Compose(
            [
                load_image_transform,
                LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes), #encode label
                # transforms.Lambdad(keys=["label"], func=lambda x: (np.arange(num_classes)<int(x)).astype(np.float32)), #encode label (same but can't pickle for multiprocessing in dataloader)
                TileOnGridd(keys=["image"], tile_count=args.max_tiles, tile_size=args.tile_size, tile_random_offset=True, tile_all=False, tile_background_val=255, return_list_of_dicts=True),
                transforms.RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                transforms.RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
                transforms.RandRotate90d(keys=["image"], prob=0.5),
                normalize_image_transform,
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        valid_transform = transforms.Compose(
            [
                load_image_transform,
                LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes), #encode label
                TileOnGridd(keys=["image"], tile_count=args.max_tiles, tile_size=args.tile_size, tile_random_offset=False, tile_all=True, tile_background_val=255, return_list_of_dicts=True),
                normalize_image_transform,
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                load_image_transform,
                LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes), #encode label
                TileOnGridd(keys=["image"], tile_count=args.max_tiles, tile_size=args.tile_size, tile_random_offset=True, tile_all=False, tile_background_val=255, return_list_of_dicts=False),
                SimpleTileFlip(keys=["image"]),
                normalize_image_transform,
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        valid_transform = transforms.Compose(
            [
                load_image_transform,
                LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes), #encode label
                TileOnGridd(keys=["image"], tile_count=args.max_tiles, tile_size=args.tile_size, tile_random_offset=False, tile_all=True, tile_background_val=255, return_list_of_dicts=False),
                normalize_image_transform,
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )



    if args.quick: #TODO, remove later
        training_list = training_list[:16]
        validation_list = validation_list[:16]

    dataset_train = data.Dataset(data=training_list, transform=train_transform)
    dataset_valid = data.Dataset(data=validation_list, transform=valid_transform)

    train_sampler = DistributedSampler(dataset_train) if args.distributed else None
    val_sampler = DistributedSampler(dataset_valid, shuffle=False) if args.distributed else None

    collate_fn=None

    if args.collate:
        collate_fn=list_data_collate



    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=False, multiprocessing_context='spawn', sampler=train_sampler, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False, multiprocessing_context='spawn', sampler=val_sampler, collate_fn=collate_fn)

    print('Data train:', len(dataset_train), 'validation:', len(dataset_valid))


    os.environ["TORCH_HOME"] = "../../torchhome" #TODO, remove later
    model = MILModel(num_classes=args.num_classes, pretrained=True, mil_mode=args.mil_mode)

    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint: best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))


    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    params = model.parameters()

    if args.mil_mode in [MilMode.ATT_TRANS, MilMode.ATT_TRANS_PYRAMID]:
        m = model if not args.distributed else model.module
        params = [{'params': list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
                  {'params':  list(m.transformer.parameters()), 'lr': 3e-6, 'weight_decay': 0.1 }]


    optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay = args.reg_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)



    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank==0: print('Writing Tensorboard logs to ', writer.log_dir)
    else:
        writer = None

    if args.validation_only:
        epoch_time = time.time()
        val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=0, args=args, max_tiles=args.max_tiles)
        if args.rank == 0:
            print('Final validation loss: {:.4f}'.format(val_loss), 'acc: {:.4f}'.format(val_acc), 'qwk: {:.4f}'.format(qwk), 'time {:.2f}s'.format(time.time() - epoch_time))

        exit(0)


    ###RUN TRAINING
    n_epochs = args.epochs
    val_acc_max = 0.

    scaler = None
    if args.amp:  # new native amp
        scaler = GradScaler()


    for epoch in range(start_epoch, n_epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            torch.distributed.barrier() #synch processess

        print(args.rank, time.ctime(), 'Epoch:', epoch)

        epoch_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args)

        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, n_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'acc: {:.4f}'.format(train_acc), 'time {:.2f}s'.format(time.time() - epoch_time))

        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)


        if args.distributed:
            torch.distributed.barrier() #synch processess


        b_new_best=False
        val_acc = 0
        if ( (epoch+1) % args.val_every == 0):

            epoch_time = time.time()
            val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=epoch, args=args, max_tiles=args.max_tiles)
            if args.rank == 0:
                print('Final validation  {}/{}'.format(epoch, n_epochs - 1), 'loss: {:.4f}'.format(val_loss),
                      'acc: {:.4f}'.format(val_acc),'qwk: {:.4f}'.format(qwk),  'time {:.2f}s'.format(time.time() - epoch_time))
                if writer is not None:
                    writer.add_scalar('val_loss', val_loss, epoch)
                    writer.add_scalar('val_acc', val_acc, epoch)
                    writer.add_scalar('val_qwk', qwk, epoch)

                val_acc = qwk

                if val_acc > val_acc_max:
                    print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(val_acc_max, val_acc))
                    val_acc_max = val_acc
                    b_new_best = True


        if args.rank == 0 and args.logdir is not None:
            save_checkpoint(model, epoch, args, best_acc=val_acc, filename='model_final.tar')
            if b_new_best:
                print('Copying to model.tar new best model!!!!')
                shutil.copyfile(os.path.join(args.logdir, 'model_final.tar'), os.path.join(args.logdir, 'model.tar'))

        scheduler.step()

    print('ALL DONE')

if __name__ == '__main__':
    main()