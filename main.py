
import torch

import os 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor                 
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pixel_level_contrastive_learning import PixelCL

from torch.utils.data import DataLoader
from torchvision.models  import resnet34
from torchvision.transforms import ToTensor, Compose, Resize

import argparse

from dataset import  TinyImageNetWithDepth



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, 
                    n_epochs,
                    warmup_epochs,
                    batch_size,
                    unlabeled_data_dir,
                    depth_data_dir,lr,
                    **kwargs):

        super().__init__()
        self.learner = PixelCL(net, **kwargs)
        self.unlabeled_data_dir = unlabeled_data_dir
        self.depth_data_dir = depth_data_dir
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.dataset  = TinyImageNetWithDepth(self.unlabeled_data_dir,
                                              self.depth_data_dir,
                                              transform=Compose([Resize(244),ToTensor()]))


    def forward(self, batch):
        imgs, depths = batch
        return self.learner(imgs,depths)

    def train_dataloader(self):

        train_loader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers= 12,
                                pin_memory=True,
                                shuffle=True)
        return train_loader


    def training_step(self, batch, _):
        loss = self.forward(batch)
        self.log('train_loss',loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                                lr=self.lr * self.batch_size / 256,
                                weight_decay=1.5e-6 )

        scheduler = LinearWarmupCosineAnnealingLR(opt,
                                                  warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.n_epochs,
                                                  eta_min =1e-5) #CosineAnnealingLR(opt, T_max = self.n_epochs, eta_min = 1e-5,)
        scheduler = {'scheduler': scheduler,'interval': 'step','frequency': 1}
        return [opt], [scheduler]


    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict



class MyDDP(pl.plugins.training_type.ddp.DDPPlugin):
    def configure_ddp(self, model, device_ids):
        model = pl.overrides.data_parallel.LightningDistributedDataParallel(model, device_ids, find_unused_parameters=True)
        return model





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",'-d', help="Root folder of the dataset.",
                    type=str,default='./images', required=False)

    parser.add_argument("--depth_dir",'-dp', help="Root folder of the depthmaps",
                    type=str,default='./depthmaps', required=False)

    parser.add_argument("--output_dir", '-o' ,help="Path for checkpoint saving",
                    type=str,default='./results/', required=False)


    parser.add_argument("--n_epochs", help="Number of finetuning epochs",
                    type=int,default=1000)

    parser.add_argument("--batch_size", '-b' ,help="Self-explanatory",
                    type=int,default=20)

    parser.add_argument("--lr",'-l' ,help="Base Learning Rate",
                    type=float, default=3e-3)

    parser.add_argument("--n_gpus", help="Number of GPUS",
                    type=float, default=1)

    parser.add_argument("--resume_epoch", help="Epoch from where to resume (0 for beginning)",
                    type=float, default=0)

    parser.add_argument("--warmup_epochs", help="Number of warmup epochs",
                    type=float, default=10)

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()

    unlabeled_data_dir = args.data_dir
    depth_dir = args.depth_dir
    results_dir = args.output_dir
    n_epochs = args.n_epochs
    warmup_epochs = args.warmup_epochs
    batch_size = args.batch_size
    resume_epoch = args.resume_epoch
    n_gpus = args.n_gpus


    resnet = resnet34(pretrained=False)

    learner = SelfSupervisedLearner(
        resnet,
        n_epochs = n_epochs,
        warmup_epochs= warmup_epochs,
        batch_size = batch_size,
        unlabeled_data_dir = unlabeled_data_dir,
        depth_data_dir=depth_dir,
        hidden_layer_pixel = 'layer4',  # leads to output of 8x8 feature map for pixel-level learning
        hidden_layer_instance = -2,     # leads to output for instance-level learning
        image_size = 244,
        lr = args.lr,
        projection_size = 256,           # the projection size
        projection_hidden_size = 2048,   # the hidden dimension of the MLP for both the projection and prediction
        moving_average_decay = 0.996,      # the moving average decay factor for the target encoder, already set at what paper recommends
        plusplus = True,
    )


    ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='PixelCL++-{epoch:02d}',
    )

       
    lr_logger = LearningRateMonitor(logging_interval='step')
    gpu_mon = GPUStatsMonitor()
    my_ddp = MyDDP()


    if resume_epoch == 0 :
        trainer = pl.Trainer(
            gpus = n_gpus,
            max_epochs = n_epochs,
            accumulate_grad_batches = 1,
            sync_batchnorm = True,
            callbacks = [ckpt_callback, lr_logger, gpu_mon],
            accelerator = 'ddp' if n_gpus > 1 else None,
            log_every_n_steps=10,
            plugins = [my_ddp] if n_gpus > 1 else None,
            precision=16
        )
    else:
        trainer = pl.Trainer(
            gpus = n_gpus,
            max_epochs = n_epochs,
            accumulate_grad_batches = 1,
            sync_batchnorm = True,
            callbacks = [ckpt_callback, lr_logger, gpu_mon],
            accelerator = 'ddp' if n_gpus > 1 else None,
            log_every_n_steps=10,
            plugins = [my_ddp] if n_gpus > 1 else None,
            precision=16,
            resume_from_checkpoint= os.path.join(results_dir,'PixelCL++-epoch=%02d.ckpt' % (resume_epoch-1))
            )


    trainer.fit(learner)


