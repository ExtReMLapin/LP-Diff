import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.netG = torch.compile(self.netG, mode="default")

        self.schedule_phase = None
        if opt['distributed']:
            assert torch.cuda.is_available()
            self.netG = nn.parallel.DistributedDataParallel(
                self.netG, device_ids=[self.local_rank], find_unused_parameters=False)

        # set loss and load resume state
        self.set_loss()
        lambda_mta = opt['train'].get('lambda_mta', 1.0)
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.netG.module.set_lambda_mta(lambda_mta)
        else:
            self.netG.set_lambda_mta(lambda_mta)
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"], fused=True)
            self.scheduler = CosineAnnealingLR(
                self.optG,
                T_max=opt['train']['n_iter'],
                eta_min=opt['train']['optimizer'].get('lr_min', 1e-6))
            self.log_dict = OrderedDict()
        
            if opt['train']['resume_training']:
                self.load_network()
            if opt['train']["use_prerain_MTA"]:
                checkpoint_path = opt['train']['MTA']
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    self.netG.module.MTA.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.netG.MTA.load_state_dict(checkpoint['model_state_dict'])
                print('Load MTA pretrained model successfully!')
        else:
            self.load_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            l_pix, l_diffusion, l_mta = self.netG(self.data)
            b = self.data['HR'].shape[0]
            # Losses already have reduction='sum' in diffusion.py, just normalize by batch
            l_pix = l_pix / b
            l_diffusion = l_diffusion / b
            l_mta = l_mta / b
            # Combine all losses with lambda_mta weight
            lambda_mta = self.opt['train'].get('lambda_mta', 1.0)
            l_total = l_pix # l_pix already contains the total sum returned by diffusion.py
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        self.optG.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_diffusion'] = l_diffusion.item()
        self.log_dict['l_mta'] = l_mta.item()
        self.log_dict['l_total'] = l_total.item()
        self.log_dict['lr'] = self.optG.param_groups[0]['lr']

    def test(self, continuous=False):
        self.netG.eval()
        with torch.no_grad():
            # Use LR_seq if available, fallback to LR1/2/3 for backward compatibility
            if 'LR_seq' in self.data:
                lr_input = self.data['LR_seq']
            else:
                lr_input = torch.stack([self.data['LR1'], self.data['LR2'], self.data['LR3']], dim=1)
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.SR = self.netG.module.super_resolution(
                    self.netG.module.MTA(*lr_input.unbind(1)), continuous)
            else:
                self.SR = self.netG.super_resolution(
                    self.netG.MTA(*lr_input.unbind(1)), continuous)
        mse_loss = nn.MSELoss()
        loss = mse_loss(self.SR, self.data['HR'])

        self.netG.train()

        return loss

    def sample(self, batch_size=1, continuous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.SR = self.netG.module.sample(batch_size, continuous)
            else:
                self.SR = self.netG.sample(batch_size, continuous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            # out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            out_dict['LR1'] = self.data['LR1'].detach().float().cpu()
            out_dict['LR2'] = self.data['LR2'].detach().float().cpu()
            out_dict['LR3'] = self.data['LR3'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))
    
    def save_best_loss(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_best_loss.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt_best_loss.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))
    
    def save_best_psnr(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_best_psnr.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt_best_psnr.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))
        
    def save_best_both(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_best.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt_best.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path, weights_only=True), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path, weights_only=True)
                self.optG.load_state_dict(opt['optimizer'])
                if opt.get('scheduler') is not None and hasattr(self, 'scheduler'):
                    self.scheduler.load_state_dict(opt['scheduler'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
