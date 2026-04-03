import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
torch.set_float32_matmul_precision('high')
import data as Data
import core.logger as Logger
import core.metrics as Metrics
from model.LPDiff_modules.Multi_tmp_fusion import MTA

def main():
    # Distributed setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/LP-Diff.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # Setup directories for MTA exclusively (only on rank 0)
    exp_name = opt['name'] + '_MTA_only'
    exp_dir = os.path.join('experiments', exp_name)
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'checkpoint'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
        Logger.setup_logger(None, exp_dir, 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')
        logger.info("Starting purely MTA Training")
    else:
        logger = logging.getLogger('base') # dummy logger for other ranks

    # Dataset setup
    dataset_opt = opt['datasets']['train']
    train_set = Data.create_dataset(dataset_opt, 'train')
    
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = Data.create_dataloader(train_set, dataset_opt, 'train', sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = Data.create_dataloader(train_set, dataset_opt, 'train')
    
    val_dataset_opt = opt['datasets']['val']
    val_set = Data.create_dataset(val_dataset_opt, 'val')
    if world_size > 1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
        val_loader = Data.create_dataloader(val_set, val_dataset_opt, 'val', sampler=val_sampler)
    else:
        val_loader = Data.create_dataloader(val_set, val_dataset_opt, 'val')

    if rank == 0:
        logger.info('Datasets loaded.')

    # Model setup
    model = MTA(in_channel=3, out_channel=3).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    # Using the learning rate from config, or fallback to 1e-4
    lr = opt['train']['optimizer'].get('lr', 1e-4) if opt['train']['optimizer'] else 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We will use n_iter to define the length of training and the scheduler limit
    n_iter = opt['train']['n_iter']
    scheduler = CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=1e-6)
    loss_fn = nn.L1Loss().to(device)

    current_step = 0
    current_epoch = 0
    best_psnr = -1.0

    val_check_epochs = opt['train'].get('val_check_epochs', 5)
    print_freq = opt['train'].get('print_freq', 200)

    while current_step < n_iter:
        current_epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(current_epoch)
            
        model.train()
        epoch_loss = 0.0
        epoch_loss_count = 0

        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            
            lr1 = train_data['LR1'].to(device)
            lr2 = train_data['LR2'].to(device)
            lr3 = train_data['LR3'].to(device)
            hr = train_data['HR'].to(device)

            optimizer.zero_grad()
            condition = model(lr1, lr2, lr3)
            loss = loss_fn(condition, hr)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_loss_count += 1

            if current_step % print_freq == 0 and rank == 0:
                progression = (current_step / n_iter) * 100
                logger.info(f"<Epoch: {current_epoch:3d}, Iter: {current_step:8d}/{n_iter} ({progression:.1f}%)> L1 loss: {loss.item():.4e}, lr: {scheduler.get_last_lr()[0]:.4e}")

        avg_loss = epoch_loss / max(epoch_loss_count, 1)
        if rank == 0:
            logger.info(f"<Epoch: {current_epoch:3d}> Average Train L1 Loss: {avg_loss:.4e}")

        # Validation step
        if current_epoch % val_check_epochs == 0:
            if world_size > 1:
                dist.barrier()
                
            model.eval()
            avg_psnr = 0.0
            idx = 0
            
            if rank == 0:
                val_result_path = os.path.join(exp_dir, 'results', str(current_epoch))
                os.makedirs(val_result_path, exist_ok=True)

            with torch.no_grad():
                for _, val_data in enumerate(val_loader):
                    lr1 = val_data['LR1'].to(device)
                    lr2 = val_data['LR2'].to(device)
                    lr3 = val_data['LR3'].to(device)
                    hr = val_data['HR'].to(device)

                    condition = model(lr1, lr2, lr3)
                    bs = condition.shape[0]

                    for b in range(bs):
                        idx += 1
                        sr_img = Metrics.tensor2img(condition[b].cpu())
                        hr_img = Metrics.tensor2img(hr[b].cpu())

                        if rank == 0 and idx <= 10:
                            Metrics.save_img(sr_img, os.path.join(val_result_path, f'{idx}_mta.png'))
                            Metrics.save_img(hr_img, os.path.join(val_result_path, f'{idx}_hr.png'))

                        avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

            # Aggregate metrics across all GPUs
            if world_size > 1:
                metrics = torch.tensor([float(avg_psnr), float(idx)], device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                avg_psnr = (metrics[0] / metrics[1]).item()
            else:
                avg_psnr = avg_psnr / max(idx, 1)

            if rank == 0:
                logger.info(f"# Validation # MTA PSNR: {avg_psnr:.4f}")

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_model_path = os.path.join(exp_dir, 'checkpoint', 'best_mta.pt')
                    
                    # Unwrap model for saving
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({'model_state_dict': model_to_save.state_dict(), 'epoch': current_epoch, 'psnr': best_psnr}, best_model_path)
                    logger.info(f"Saved new best MTA model to {best_model_path}")

    if rank == 0:
        logger.info("MTA Training completed.")
        
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
