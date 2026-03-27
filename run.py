import torch
import torch.distributed as dist
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Distributed setup — torchrun injects LOCAL_RANK / RANK / WORLD_SIZE
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/LP-Diff.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    if rank == 0:
        print('Begin Training-------------------------------------------')

    best_loss = float('inf')
    best_psnr = float('-inf')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Inject distributed info into opt
    opt['local_rank'] = local_rank
    opt['rank'] = rank
    opt['world_size'] = world_size
    if world_size > 1:
        opt['distributed'] = True

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        Logger.setup_logger(None, opt['path']['log'],
                            'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    else:
        tb_logger = None
    logger = logging.getLogger('base')
    if rank == 0:
        logger.info(Logger.dict2str(opt))

    # Initialize WandbLogger
    if opt['enable_wandb'] and rank == 0:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    train_sampler = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            if world_size > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase, sampler=train_sampler)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_sampler = (
                torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
                if world_size > 1 else None
            )
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase, sampler=val_sampler)
    if rank == 0:
        logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    if rank == 0:
        logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state'] and rank == 0:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                epoch_loss_sum += diffusion.get_current_log().get('l_pix', 0.0)
                epoch_loss_count += 1
                # log
                if current_step % opt['train']['print_freq'] == 0 and rank == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        if tb_logger is not None:
                            tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                if current_step % opt['train']['save_checkpoint_freq'] == 0 and rank == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            avg_train_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            if rank == 0:
                logger.info('<epoch:{:3d}, iter:{:8,d}> avg_train_loss: {:.4e}'.format(
                    current_epoch, current_step, avg_train_loss))

            # end-of-epoch validation (after warm-up)
            warm_up = opt['train'].get('val_warmup_epochs', 5)
            if current_epoch >= warm_up:
                if world_size > 1:
                    dist.barrier()

                avg_psnr = 0.0
                idx = 0
                avg_val_loss = 0.0
                if rank == 0:
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')
                for _,  val_data in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    loss = diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                    lr1_img = Metrics.tensor2img(visuals['LR1'])  # uint8
                    lr2_img = Metrics.tensor2img(visuals['LR2'])  # uint8
                    lr3_img = Metrics.tensor2img(visuals['LR3'])  # uint8

                    if rank == 0:
                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr1_img, '{}/{}_{}_lr1.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr2_img, '{}/{}_{}_lr2.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr3_img, '{}/{}_{}_lr3.png'.format(result_path, current_step, idx))
                        if tb_logger is not None:
                            tb_logger.add_image(
                                'Epoch_{}'.format(current_epoch),
                                np.transpose(np.concatenate(
                                    (lr1_img, lr2_img, lr3_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                    avg_psnr += Metrics.calculate_psnr(
                        sr_img, hr_img)
                    avg_val_loss += loss

                    if wandb_logger:
                        wandb_logger.log_image(
                            f'validation_{idx}',
                            np.concatenate((lr1_img, lr2_img, lr3_img, sr_img, hr_img), axis=1)
                        )

                # Aggregate metrics across all GPUs
                if world_size > 1:
                    metrics = torch.tensor(
                        [float(avg_psnr), float(avg_val_loss), float(idx)],
                        device=f'cuda:{local_rank}')
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    avg_psnr = (metrics[0] / metrics[2]).item()
                    avg_val_loss = (metrics[1] / metrics[2]).item()
                else:
                    avg_psnr = avg_psnr / idx
                    avg_val_loss = avg_val_loss / idx
                loss_improved = avg_val_loss < best_loss
                psnr_improved = avg_psnr > best_psnr
                if rank == 0:
                    if loss_improved and psnr_improved:
                        best_loss = avg_val_loss
                        best_psnr = avg_psnr
                        diffusion.save_best_both(current_epoch, current_step)
                    elif loss_improved:
                        best_loss = avg_val_loss
                        diffusion.save_best_loss(current_epoch, current_step)
                    elif psnr_improved:
                        best_psnr = avg_psnr
                        diffusion.save_best_psnr(current_epoch, current_step)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')
                # log
                if rank == 0:
                    logger.info('# Validation # PSNR: {:.4e}  val_loss: {:.4e}'.format(avg_psnr, avg_val_loss))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} loss: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_val_loss))
                    # tensorboard logger
                    if tb_logger is not None:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        if rank == 0:
            logger.info('End of training.')
    else:
        if rank == 0:
            logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        if rank == 0:
            os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr1_img = Metrics.tensor2img(visuals['LR1'])  # uint8
            lr2_img = Metrics.tensor2img(visuals['LR2'])  # uint8
            lr3_img = Metrics.tensor2img(visuals['LR3'])  # uint8

            filename = os.path.basename(os.path.split(diffusion.data['path'][0])[0])

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                if rank == 0:
                    for iter in range(0, sample_num):
                        Metrics.save_img(
                            Metrics.tensor2img(sr_img[iter]), '{}/{}_sr.png'.format(result_path, filename))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                if rank == 0:
                    Metrics.save_img(
                        sr_img, '{}/{}_sr_process.png'.format(result_path, filename))
                    Metrics.save_img(
                        Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_sr.png'.format(result_path, filename))

            if rank == 0:
                Metrics.save_img(
                    hr_img, '{}/{}_hr.png'.format(result_path, filename))
                Metrics.save_img(
                    lr1_img, '{}/{}_lr1.png'.format(result_path, filename))
                Metrics.save_img(
                    lr2_img, '{}/{}_lr2.png'.format(result_path, filename))
                Metrics.save_img(
                    lr3_img, '{}/{}_lr3.png'.format(result_path, filename))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(lr2_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        if rank == 0:
            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
                current_epoch, current_step, avg_psnr, avg_ssim))

            if wandb_logger:
                if opt['log_eval']:
                    wandb_logger.log_eval_table()
                wandb_logger.log_metrics({
                    'PSNR': float(avg_psnr),
                    'SSIM': float(avg_ssim)
                })

    if world_size > 1:
        dist.destroy_process_group()
