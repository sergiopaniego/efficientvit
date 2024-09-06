import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from tqdm import tqdm
from datetime import datetime


from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter#, get_dist_local_rank, get_dist_size, is_master, sync_tensor
from efficientvit.models.utils import list_join
from efficientvit.segcore.data_provider import SEGDataProvider ####
from efficientvit.segcore.trainer import SEGRunConfig ####
from efficientvit.segcore.trainer.utils import ( ####
    compute_boundary_iou,
    compute_iou,
    dice_loss,
    #loss_masks,
    mask_iou_batch,
    masks_sample_points,
)

__all__ = ["SEGTrainer"] ####


class SEGTrainer(Trainer): ####
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider: SEGDataProvider, ####
        project_name: str,
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )

        self.starting_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #if is_master():
            #self.wandb_log = wandb.init(project="efficientvit-seg") ####
        self.wandb_log = wandb.init(
            project=f"efficientvit-seg-{project_name}",
            name=f"efficientvit-seg-{self.starting_datetime}"
            )


    def _validate(self, model, data_loader, epoch: int, sub_epoch: int) -> dict[str, any]:
        val_loss = AverageMeter(False)
        #val_iou = AverageMeter(False)
        #val_iou_boundary = AverageMeter(False)
        val_loss_ce = AverageMeter(False)
        val_loss_dice = AverageMeter(False)

        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Epoch #{epoch + 1}, Sub Epoch #{sub_epoch+1}",
                #disable=not is_master(),
                file=sys.stdout,
            ) as t:
                for i, data in enumerate(data_loader):
                    image = data["image"].cuda()
                    masks = data["masks"].cuda()
                    batched_input = image
                    output = self.model(batched_input)

                    # Interpolate if neccesary
                    if output.shape[2:] != image.shape[2:]:
                        output = F.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)

                    # Calculate semantic segmentation loss
                    loss_ce = F.cross_entropy(output, masks, reduction='mean')
                    loss_dice = dice_loss(output, masks)  # Asumiendo que hay una función dice_loss definida

                    # Combination of losses
                    loss = loss_ce + loss_dice
                    '''

                    #bboxs = data["bboxs"].cuda() * 2 if image.shape[2] == 512 else data["bboxs"].cuda()
                    #points = data["points"].cuda() * 2 if image.shape[2] == 512 else data["points"].cuda()

                    #bboxs[..., 2] = bboxs[..., 0] + bboxs[..., 2]
                    #bboxs[..., 3] = bboxs[..., 1] + bboxs[..., 3]

                    batched_input = []
                    for b_i in range(len(image)):
                        dict_input = dict()

                        dict_input["image"] = image[b_i]
                        #dict_input["boxes"] = bboxs[b_i]

                        batched_input.append(dict_input)

                    output, iou_predictions = model(batched_input, True)

                    B, M, N, H, W = output.shape
                    output = torch.stack(
                        [
                            output[k][torch.arange(M), iou_predictions[k].argmax(-1).squeeze()]
                            for k in range(len(output))
                        ],
                        dim=0,
                    )
                    output = (
                        F.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear")
                        .reshape(-1, image.shape[2], image.shape[3])
                        .unsqueeze(1)
                    )
                    masks = masks.reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)

                    loss_mask, loss_dice = loss_masks(output, masks, len(output))
                    loss = loss_mask * 20 + loss_dice

                    iou = compute_iou(output, masks * 255)
                    boundary_iou = compute_boundary_iou(output, masks * 255)

                    #loss = sync_tensor(loss)
                    #iou = sync_tensor(iou)
                    #boundary_iou = sync_tensor(boundary_iou)
                    '''
                    #val_loss.update(loss, image.shape[0] * get_dist_size())
                    #val_iou.update(iou, image.shape[0] * get_dist_size())
                    #val_iou_boundary.update(boundary_iou, image.shape[0] * get_dist_size())
                    val_loss.update(loss, image.shape[0])
                    val_loss_ce.update(loss_ce, image.shape[0])
                    val_loss_dice.update(loss_dice, image.shape[0])
                    #val_iou.update(iou, image.shape[0])
                    #val_iou_boundary.update(boundary_iou, image.shape[0])
                    
                    t.set_postfix(
                        {
                            "loss": val_loss.avg,
                            "loss_ce": val_loss_ce.avg,
                            "loss_dice": val_loss_dice.avg,
                            #"iou": val_iou.avg,
                            #"boundary_iou": val_iou_boundary.avg,
                            #"bs": image.shape[0] * get_dist_size(),
                            "bs": image.shape[0],
                        }
                    )
                    t.update()

        #if is_master():
            #self.wandb_log.log(
            #    {"val_loss": val_loss.avg, "val_iou": val_iou.avg, "val_boundary_iou": val_iou_boundary.avg}
            #)
        self.wandb_log.log(
            {
                "val_loss": val_loss.avg, 
                "val_loss_ce": val_loss_ce.avg,
                "val_loss_dice": val_loss_dice.avg,
                #"val_iou": val_iou.avg, 
                #"val_boundary_iou": val_iou_boundary.avg
                }
        )

        return {
            "val_loss": val_loss.avg,
            #"val_iou": val_iou.avg,
            #"val_boundary_iou": val_iou_boundary.avg,
        }

    def validate(self, model=None, data_loader=None, epoch=0, sub_epoch=0) -> dict[str, any]:
        model = model or self.eval_network
        if data_loader is None:
            data_loader = self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch, sub_epoch)

    def before_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        image = feed_dict["image"].cuda()
        masks = feed_dict["masks"].cuda()
        #bboxs = feed_dict["bboxs"].cuda() * 2 if image.shape[2] == 512 else feed_dict["bboxs"].cuda()
        #points = feed_dict["points"].cuda() * 2 if image.shape[2] == 512 else feed_dict["points"].cuda()

        #bboxs[..., 2] = bboxs[..., 0] + bboxs[..., 2]
        #bboxs[..., 3] = bboxs[..., 1] + bboxs[..., 3]

        return {
            "image": image,
            "masks": masks,
            #"points": points,
            #"bboxs": bboxs,
        }

    def run_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        image = feed_dict["image"]
        masks = feed_dict["masks"]
        
        batched_input = image
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            output = self.model(batched_input)

            # Interpolate if neccesary
            if output.shape[2:] != image.shape[2:]:
                output = F.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)

            # Calculate semantic segmentation loss
            loss_ce = F.cross_entropy(output, masks, reduction='mean')
            loss_dice = dice_loss(output, masks)  # Asumiendo que hay una función dice_loss definida

            # Combination of losses
            loss = loss_ce + loss_dice

        self.scaler.scale(loss).backward()

        return {"loss": loss, "loss_ce": loss_ce, "loss_dice": loss_dice, "output": output}

    def _train_one_sub_epoch(self, epoch: int, sub_epoch: int) -> dict[str, any]:
        train_loss = AverageMeter(False)
        train_loss_ce = AverageMeter(False)
        train_loss_dice = AverageMeter(False)

        with tqdm(
            total=len(self.data_provider.train),
            desc=f"Train Epoch #{epoch + 1}, Sub Epoch #{sub_epoch + 1}",
            #disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for i, data in enumerate(self.data_provider.train):
                feed_dict = data

                # preprocessing
                feed_dict = self.before_step(feed_dict)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                output_dict = self.run_step(feed_dict)
                # update: optimizer, lr_scheduler
                self.after_step()

                loss = output_dict["loss"]
                loss_ce = output_dict["loss_ce"]
                loss_dice = output_dict["loss_dice"]
                #loss = sync_tensor(loss)
                #train_loss.update(loss, data["image"].shape[0] * get_dist_size())
                train_loss.update(loss, data["image"].shape[0])
                train_loss_ce.update(loss_ce, data["image"].shape[0])
                train_loss_dice.update(loss_dice, data["image"].shape[0])

                #if is_master():
                #    self.wandb_log.log(
                #        {
                #            "train_loss": train_loss.avg,
                #            "epoch": epoch,
                #            "sub_epoch": sub_epoch,
                #            "learning_rate": sorted(set([group["lr"] for group in self.optimizer.param_groups]))[0],
                #        }
                #    )
                self.wandb_log.log(
                    {
                        "train_loss": train_loss.avg,
                        "train_loss_ce": train_loss_ce.avg,
                        "train_loss_dice": train_loss_dice.avg,
                        "epoch": epoch,
                        "sub_epoch": sub_epoch,
                        "learning_rate": sorted(set([group["lr"] for group in self.optimizer.param_groups]))[0],
                    }
                )

                t.set_postfix(
                    {
                        "loss": train_loss.avg,
                        "train_loss_ce": train_loss_ce.avg,
                        "train_loss_dice": train_loss_dice.avg,
                        #"bs": data["image"].shape[0] * get_dist_size(),
                        "bs": data["image"].shape[0],
                        "res": data["image"].shape[2],
                        "lr": list_join(
                            sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                            "#",
                            "%.1E",
                        ),
                        "progress": self.run_config.progress,
                    }
                )
                t.update()

        return {
            "train_loss": train_loss.avg,
        }

    def train_one_sub_epoch(self, epoch: int, sub_epoch: int) -> dict[str, any]:
        self.model.train()

        self.data_provider.set_epoch_and_sub_epoch(epoch, sub_epoch)

        train_info_dict = self._train_one_sub_epoch(epoch, sub_epoch)

        return train_info_dict

    def train(self) -> None:

        for sub_epoch in range(self.start_epoch, self.run_config.n_epochs):
            epoch = sub_epoch // self.data_provider.sub_epochs_per_epoch

            train_info_dict = self.train_one_sub_epoch(epoch, sub_epoch)

            val_info_dict = self.validate(epoch=epoch, sub_epoch=sub_epoch)

            # val_iou = val_info_dict["val_iou"]
            val_iou = val_info_dict["val_loss"]
            is_best = val_iou > self.best_val
            self.best_val = max(val_iou, self.best_val)

            print(f'Saving model... {self.starting_datetime}_checkpoint_{epoch}_{sub_epoch}.pt')
            self.save_model(
                only_state_dict=False,
                epoch=sub_epoch,
                model_name=f"{self.starting_datetime}_checkpoint_{epoch}_{sub_epoch}.pt",
            )

    def prep_for_training(self, run_config: SEGRunConfig, amp="fp32") -> None: ####
        self.run_config = run_config
        #self.model = nn.parallel.DistributedDataParallel(
        #    self.model.cuda(),
        #    device_ids=[get_dist_local_rank()],
        #    find_unused_parameters=True,
        #)
        print(self.run_config)
        print(self.data_provider)
        print(self.data_provider.train)
        

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        # amp
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
