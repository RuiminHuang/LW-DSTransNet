from argparse import ArgumentParser

import torch
import torch.utils.data as Data
from data.data import DataLoaderX, GrokCV_NUAA_SIRST, GrokCV_NUDT_SIRST, GrokCV_IRSTD_1k, GrokCV_SIRST_Aug

from model.model import DS_TransNet_Teacher, DS_TransNet_Student, DS_TransNet_Student_v5
from model.model_utils.weight_init import weight_init_normal, weights_init_kaiming
from model.model_utils.optimizer_set import optimizer_set_adam, optimizer_set_adagrad, optimizer_set_sgd
from model.model_utils.lr_scheduler import lr_scheduler_WarmDecayLR, lr_scheduler_WarmConstantDecayLR, lr_scheduler_MultiStepLR, lr_scheduler_CosineAnnealingLR_With_GradualWarmup
from model.model_utils.metric import SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, PD_FA


from utils.tools import denormalize, set_seed, init_env

from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
import numpy as np
from tqdm import tqdm


features = {}

# detach
def get_teacher_multi_output_activation(name):
    def hook(model, input, output):

        if isinstance(output, torch.Tensor):
            features[name] = output.detach()
            
        elif isinstance(output, (tuple, list)):
            features[name] = [item.detach() for item in output if isinstance(item, torch.Tensor)]

        elif isinstance(output, dict):
            features[name] = {k: v.detach() for k, v in output.items() if isinstance(v, torch.Tensor)}
            
    return hook

# don't detach
def get_student_multi_output_activation(name):
    def hook(model, input, output):

        if isinstance(output, torch.Tensor):
            features[name] = output
            
        elif isinstance(output, (tuple, list)):
            features[name] = [item for item in output if isinstance(item, torch.Tensor)]

        elif isinstance(output, dict):
            features[name] = {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}
            
    return hook



def parse_args():

    parser = ArgumentParser(description='Implement of DS_TransNet')
    
    # log
    parser.add_argument('--log_root', type=str, default="logs", help='log dir')
    parser.add_argument('--exp_name', type=str, default="DS_TransNet_Student_v42_NUAA", help='experiment name')
    parser.add_argument('--phase_name', type=str, default="train", help='phase name')
    parser.add_argument('--log_name', type=str, default="log.log", help='log name')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size for training')
    parser.add_argument('--teacher_weight_path', type=str, default="logs/DS_TransNet_Teacher_NUAA/train/20260420023404/best_miou.pth", help='weight for testing')

    # environment
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--seed", type=int, default=3407, help="Torch seed 3407 is all you need")


    # scheduler
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')

    parser.add_argument('--lr', type=float, default=0.001, help='basic learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate')

    parser.add_argument('--warmdecaylr', type=dict, default={'warm_up_epochs': 0}, help="lr_scheduler_WarmDecayLR")
    parser.add_argument('--warmconstantdecaylr', type=dict, default={'warm_up_epochs': 0, 'constant_epochs': 210}, help="lr_scheduler_WarmConstantDecayLR")
    parser.add_argument("--multisteplr", type=dict, default={'milestones': [150, 200, 260], 'gamma': 0.5}, help="lr_scheduler_MultiStepLR")
    parser.add_argument("--cosineAnnealinglr", type=dict, default={'last_epoch': -1}, help="lr_scheduler_CosineAnnealingLR")

    # hyper-parameters for loss function-----------------------------------------------------------------------------------
    parser.add_argument('--seg', type=float, default=1, help='weight for GT')

    parser.add_argument('--alpha', type=float, default=0.05, help='weight for Logit')
    # parser.add_argument('--alpha', type=float, default=0, help='weight for Logit')

    parser.add_argument('--beta', type=float, default=0.02, help='weight for Dice')

    parser.add_argument('--gamma', type=float, default=1, help='weight for RAD')
    # parser.add_argument('--gamma', type=float, default=0, help='weight for RAD')

    parser.add_argument('--encoder', type=float, default=0.005, help='weight for encoder')
    parser.add_argument('--skip', type=float, default=0.005, help='weight for skip')
    parser.add_argument('--decoder', type=float, default=0.005, help='weight for decoder')
    # parser.add_argument('--encoder', type=float, default=0, help='weight for encoder')
    # parser.add_argument('--skip', type=float, default=0, help='weight for skip')
    # parser.add_argument('--decoder', type=float, default=0, help='weight for decoder')
    # hyper-parameters for loss function-----------------------------------------------------------------------------------

    args = parser.parse_args()
    
    return args


class Trainer(object):

    def __init__(self, args):
        self.args = args


        self.trainset = GrokCV_NUAA_SIRST(mode='train')
        self.valset = GrokCV_NUAA_SIRST(mode='test')
        # self.trainset = GrokCV_NUDT_SIRST(mode='train')
        # self.valset = GrokCV_NUDT_SIRST(mode='test')
        # self.trainset = GrokCV_IRSTD_1k(mode='train')
        # self.valset = GrokCV_IRSTD_1k(mode='test')
        # self.trainset = GrokCV_SIRST_Aug(mode='train')
        # self.valset = GrokCV_SIRST_Aug(mode='test')
        
        # self.train_data_loader = Data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)
        # self.val_data_loader = Data.DataLoader(self.valset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)
        self.train_data_loader = DataLoaderX(self.trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)
        self.val_data_loader = DataLoaderX(self.valset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)

        # dir
        self.log_dir = os.path.join(args.log_root, args.exp_name, args.phase_name, time.strftime('%Y%m%d%H%M%S', time.localtime()) )
        if not os.path.exists( self.log_dir ):
            os.makedirs( self.log_dir )
        # log
        self.log_file = os.path.join( self.log_dir, args.log_name )
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        # tensorboard
        self.writer = SummaryWriter(self.log_dir)

        self.writer.add_text(self.log_dir, 'Args:%s' % args)
        # for i, (data, label) in enumerate( tqdm(self.train_data_loader, desc='Export trainset to TensorBoard      ', position=0, leave=True) ):
        #     self.writer.add_images('img/original', denormalize(data, self.trainset.mean, self.trainset.std), i, dataformats='NCHW')
        #     self.writer.add_images('img/processed', data, i, dataformats='NCHW')
        #     self.writer.add_images('label', label, i, dataformats='NCHW')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # teacher
        self.Teacher_net = DS_TransNet_Teacher(mode='test')
        # weight init
        self.Teacher_net.model.load_state_dict(torch.load(args.teacher_weight_path))
        self.Teacher_net.eval()
        self.Teacher_net = self.Teacher_net.to(self.device)
        self.Teacher_net.model = self.Teacher_net.model.to(self.device)


        # student
        self.Student_net = DS_TransNet_Student_v5(mode='train')
        # weight init 
        # self.Student_net.apply(weight_init_normal)
        self.Student_net.apply(weights_init_kaiming)
        self.Student_net.train()# will change in training stage and validation stage
        self.Student_net = self.Student_net.to(self.device)#all parameter including extra parameter
        self.Student_net.model = self.Student_net.model.to(self.device)# only model parameter

        # optimizer set
        # self.optimizer = optimizer_set_adagrad(self.Student_net.parameters(), args.lr)
        self.optimizer = optimizer_set_adam(self.Student_net.parameters(), args.lr)
        # self.optimizer = optimizer_set_sgd(self.Student_net.parameters(), args.lr)
        
        # lr_scheduler
        # self.scheduler = lr_scheduler_WarmDecayLR(self.optimizer, args.lr, args.epochs, args.warmdecaylr, args.min_lr)
        # self.scheduler = lr_scheduler_WarmConstantDecayLR(self.optimizer, args.lr, args.epochs, args.warmconstantdecaylr, args.min_lr)
        # self.scheduler = lr_scheduler_MultiStepLR(self.optimizer, args.multisteplr)
        self.scheduler = lr_scheduler_CosineAnnealingLR_With_GradualWarmup(self.optimizer, args.epochs, args.min_lr, args.cosineAnnealinglr)

        # loss
        self.Student_net.hard_loss = self.Student_net.hard_loss.to(self.device)
        # self.Student_net.KL_loss = self.Student_net.KL_loss.to(self.device)
        self.Student_net.mask_KL_loss = self.Student_net.mask_KL_loss.to(self.device)
        self.Student_net.Dice_loss = self.Student_net.Dice_loss.to(self.device)
        self.Student_net.RAD_Loss = self.Student_net.RAD_Loss.to(self.device)

        self.Student_net.encoder_loss = self.Student_net.encoder_loss.to(self.device)
        self.Student_net.skip_loss = self.Student_net.skip_loss.to(self.device)
        self.Student_net.decoder_loss = self.Student_net.decoder_loss.to(self.device)

        self.seg = args.seg
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.encoder = args.encoder
        self.skip = args.skip
        self.decoder = args.decoder


        # metric
        self.miou_metric = SigmoidMetric(score_thresh=0.5)
        self.nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5, do_sigmoid=False)
        self.best_miou = 0
        self.best_nIoU = 0


        # hooks for feature extraction
        # Teacher hooks
        self.Teacher_net.model.Conv1.register_forward_hook( get_teacher_multi_output_activation('teacher_encoder_x1') )
        self.Teacher_net.model.Conv2.register_forward_hook( get_teacher_multi_output_activation('teacher_encoder_x2') )
        self.Teacher_net.model.Conv3.register_forward_hook( get_teacher_multi_output_activation('teacher_encoder_x3') )
        self.Teacher_net.model.Conv4.register_forward_hook( get_teacher_multi_output_activation('teacher_encoder_x4') )

        self.Teacher_net.model.mtc.register_forward_hook( get_teacher_multi_output_activation('teacher_skip_x1_x2_x3_x4') )

        self.Teacher_net.model.up_decoder1.register_forward_hook( get_teacher_multi_output_activation('teacher_decoder_d1') )
        self.Teacher_net.model.up_decoder2.register_forward_hook( get_teacher_multi_output_activation('teacher_decoder_d2') )
        self.Teacher_net.model.up_decoder3.register_forward_hook( get_teacher_multi_output_activation('teacher_decoder_d3') )
        self.Teacher_net.model.up_decoder4.register_forward_hook( get_teacher_multi_output_activation('teacher_decoder_d4') )

        # Student hooks
        self.Student_net.model.input_head.register_forward_hook( get_student_multi_output_activation('student_encoder_x1') )
        self.Student_net.model.down_encoder1.register_forward_hook( get_student_multi_output_activation('student_encoder_x2') )
        self.Student_net.model.down_encoder2.register_forward_hook( get_student_multi_output_activation('student_encoder_x3') )
        self.Student_net.model.down_encoder3.register_forward_hook( get_student_multi_output_activation('student_encoder_x4') )

        self.Student_net.model.mtc.register_forward_hook( get_student_multi_output_activation('student_skip_x1_x2_x3_x4') )

        self.Student_net.model.up_decoder1.register_forward_hook( get_student_multi_output_activation('student_decoder_d1') )
        self.Student_net.model.up_decoder2.register_forward_hook( get_student_multi_output_activation('student_decoder_d2') )
        self.Student_net.model.up_decoder3.register_forward_hook( get_student_multi_output_activation('student_decoder_d3') )
        self.Student_net.model.up_decoder4.register_forward_hook( get_student_multi_output_activation('student_decoder_d4') )


    def training(self, epoch):
        self.Student_net.train()

        losses = []

        losses_GT = []
        losses_logit = []
        losses_dice = []
        losses_RAD = []

        losses_encoder = []
        losses_skip = []
        losses_decoder = []

        tbar = tqdm(self.train_data_loader, desc='Iteration...', position=1, leave=True)
        for i, (data, label) in enumerate( tbar ):
            data  = data[:, 1:2, :, :] # for single channel only

            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                feature_t, logit_t = self.Teacher_net(data)
                logit_t = [f.detach() for f in logit_t]

            # logits_val = logit_t[4]
            # print(f"Teacher Logits - Max: {logits_val.max().item():.2f}, Min: {logits_val.min().item():.2f}, Mean Abs: {logits_val.abs().mean().item():.2f}")

            feature_s, logit_s = self.Student_net(data)


            # print("----------------------------------------------")
            # for i in range(len(feature_t)):
            #     print(feature_t[i].shape)
            # print("------------------------")
            # for i in range(len(feature_s)):
            #     print(feature_s[i].shape)
            # print("----------------------------------------------")


            # print("----------------------------------------------")
            # print(features['teacher_encoder_x1'].shape)
            # print(features['teacher_encoder_x2'].shape)
            # print(features['teacher_encoder_x3'].shape)
            # print(features['teacher_encoder_x4'].shape)
            # print(features['teacher_skip_x1_x2_x3_x4'][0].shape)
            # print(features['teacher_skip_x1_x2_x3_x4'][1].shape)
            # print(features['teacher_skip_x1_x2_x3_x4'][2].shape)
            # print(features['teacher_skip_x1_x2_x3_x4'][3].shape)
            # print(len(features['teacher_skip_x1_x2_x3_x4']))
            # print(features['teacher_decoder_d1'].shape)
            # print(features['teacher_decoder_d2'].shape)
            # print(features['teacher_decoder_d3'].shape)
            # print(features['teacher_decoder_d4'].shape)
            # print("-----------------------")
            # print(self.Student_net.model.x1_conv(features['student_encoder_x1']).shape)
            # print(self.Student_net.model.x2_conv(features['student_encoder_x2']).shape)
            # print(self.Student_net.model.x3_conv(features['student_encoder_x3']).shape)
            # print(self.Student_net.model.x4_conv(features['student_encoder_x4']).shape)
            # print(self.Student_net.model.skip_x1_conv(features['student_skip_x1_x2_x3_x4'][0]).shape)
            # print(self.Student_net.model.skip_x2_conv(features['student_skip_x1_x2_x3_x4'][1]).shape)
            # print(self.Student_net.model.skip_x3_conv(features['student_skip_x1_x2_x3_x4'][2]).shape)
            # print(self.Student_net.model.skip_x4_conv(features['student_skip_x1_x2_x3_x4'][3]).shape)
            # print(len(features['student_skip_x1_x2_x3_x4']))
            # print(self.Student_net.model.d1_conv(features['student_decoder_d1']).shape)
            # print(self.Student_net.model.d2_conv(features['student_decoder_d2']).shape)
            # print(self.Student_net.model.d3_conv(features['student_decoder_d3']).shape)
            # print(self.Student_net.model.d4_conv(features['student_decoder_d4']).shape)
            # print("----------------------------------------------")


            loss_GT = self.Student_net.hard_loss_cal(logit_s, label)
            # loss_logit = self.Student_net.KL_loss_cal( logit_s, logit_t )
            loss_logit = self.Student_net.mask_KL_loss_cal( logit_s, logit_t, label )
            loss_dice = self.Student_net.Dice_loss_cal( logit_s, label )
            loss_RAD = self.Student_net.RAD_loss_cal( feature_s, feature_t, label )

            loss_encoder = self.Student_net.encoder_loss_cal( self.Student_net.model.x1_conv( features['student_encoder_x1'] ), features['teacher_encoder_x1'] ) + \
                           self.Student_net.encoder_loss_cal( self.Student_net.model.x2_conv( features['student_encoder_x2'] ), features['teacher_encoder_x2'] ) + \
                           self.Student_net.encoder_loss_cal( self.Student_net.model.x3_conv( features['student_encoder_x3'] ), features['teacher_encoder_x3'] ) + \
                           self.Student_net.encoder_loss_cal( self.Student_net.model.x4_conv( features['student_encoder_x4'] ), features['teacher_encoder_x4'] )

            loss_skip = self.Student_net.skip_loss_cal( self.Student_net.model.skip_x1_conv( features['student_skip_x1_x2_x3_x4'][0] ), features['teacher_skip_x1_x2_x3_x4'][0] ) + \
                        self.Student_net.skip_loss_cal( self.Student_net.model.skip_x2_conv( features['student_skip_x1_x2_x3_x4'][1] ), features['teacher_skip_x1_x2_x3_x4'][1] ) + \
                        self.Student_net.skip_loss_cal( self.Student_net.model.skip_x3_conv( features['student_skip_x1_x2_x3_x4'][2] ), features['teacher_skip_x1_x2_x3_x4'][2] ) + \
                        self.Student_net.skip_loss_cal( self.Student_net.model.skip_x4_conv( features['student_skip_x1_x2_x3_x4'][3] ), features['teacher_skip_x1_x2_x3_x4'][3] )
            
            loss_decoder = self.Student_net.decoder_loss_cal( self.Student_net.model.d1_conv( features['student_decoder_d1'] ), features['teacher_decoder_d1'] ) + \
                           self.Student_net.decoder_loss_cal( self.Student_net.model.d2_conv( features['student_decoder_d2'] ), features['teacher_decoder_d2'] ) + \
                           self.Student_net.decoder_loss_cal( self.Student_net.model.d3_conv( features['student_decoder_d3'] ), features['teacher_decoder_d3'] ) + \
                           self.Student_net.decoder_loss_cal( self.Student_net.model.d4_conv( features['student_decoder_d4'] ), features['teacher_decoder_d4'] )

            loss = self.seg * loss_GT + self.alpha * loss_logit + self.beta * loss_dice + self.gamma * loss_RAD + \
                   self.encoder * loss_encoder + self.skip * loss_skip + self.decoder * loss_decoder

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            losses_GT.append(self.seg * loss_GT.item())
            losses_logit.append(self.alpha * loss_logit.item())
            losses_dice.append(self.beta * loss_dice.item())
            losses_RAD.append(self.gamma * loss_RAD.item())

            losses_encoder.append( self.encoder * loss_encoder.item() )
            losses_skip.append( self.skip * loss_skip.item() )
            losses_decoder.append( self.decoder * loss_decoder.item() )

            # You can't print too frequently, as it will bring down GPU utilization
            # tbar.set_description('Iteration...  Epoch:%3d, lr:%f, train loss:%f'% (epoch, self.optimizer.param_groups[0]['lr'], np.mean(losses)))
        
        self.scheduler.step()

        if epoch > 550:
            self.alpha = 0

        if epoch > 450:
            self.gamma = 0


        if epoch > 550:
            self.encoder = 0
            self.skip = 0
            self.decoder = 0


        self.writer.add_scalar('Losses/train_loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', self.optimizer.param_groups[0]['lr'], epoch)
        self.logger.info('Epoch: %d, train_loss: %.4f, lr: %.6f' % (epoch, np.mean(losses), self.optimizer.param_groups[0]['lr']))
        print( 'loss_GT: %.6f, loss_logit: %.6f, loss_dice: %.6f, loss_RAD: %.6f' % ( np.mean(losses_GT), np.mean(losses_logit), np.mean(losses_dice), np.mean(losses_RAD) ) )
        print( 'loss_encoder: %.6f, loss_skip: %.6f, loss_decoder: %.6f' % ( np.mean(losses_encoder), np.mean(losses_skip), np.mean(losses_decoder) ) )


    def validation(self, epoch):
        self.Student_net.eval()
        self.miou_metric.reset()
        self.nIoU_metric.reset()
        losses = []
        with torch.no_grad():
            tbar = tqdm(self.val_data_loader, desc='Validation...', position=2, leave=True)
            for i, (data, label) in enumerate( tbar ):
                data  = data[:, 1:2, :, :] # for single channel only
                data = data.to(self.device)
                label = label.to(self.device)

                feature_s, logit_s = self.Student_net(data)

                loss = self.Student_net.hard_loss_cal(logit_s, label)
                losses.append(loss.item())

                output = torch.softmax(logit_s[4], dim=1)[:, 1:2, :, :]

                self.miou_metric.update(output, label)
                self.nIoU_metric.update(output, label)
                # print('output max value: %.2f, min value: %.2f' % (torch.max(output[5]), torch.min(output[5])) )
                _, miou = self.miou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                tbar.set_description('Validation...  Epoch:%3d, eval loss:%f, mIoU:%f, nIoU:%f'% (epoch, np.mean(losses), miou, nIoU))

        _, miou = self.miou_metric.get()
        _, nIoU = self.nIoU_metric.get()

        if miou > self.best_miou:
            self.best_miou = miou
            torch.save(self.Student_net.model.state_dict(), os.path.join(self.log_dir, 'best_miou.pth'))
        if nIoU > self.best_nIoU:
            self.best_nIoU = nIoU
            torch.save(self.Student_net.model.state_dict(), os.path.join(self.log_dir, 'best_nIoU.pth'))
        
        self.writer.add_scalar('Losses/val_loss', np.mean(losses), epoch)
        self.writer.add_scalar('Eval/mIoU', miou, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/best_mIoU', self.best_miou, epoch)
        self.writer.add_scalar('Best/best_nIoU', self.best_nIoU, epoch)
        self.logger.info('Epoch: %d, val_loss: %.4f, mIoU: %.4f, best_mIoU: %.4f, nIoU: %.4f, best_nIoU: %.4f' % (epoch, np.mean(losses), miou, self.best_miou, nIoU, self.best_nIoU))


if __name__ == '__main__':

    args = parse_args()

    init_env(args.gpu_ids)

    # set_seed(args.seed)

    trainer = Trainer(args)

    for epoch in  tqdm(range(args.epochs), desc='Epoch...', position=0, leave=True) :
        trainer.training(epoch)
        if epoch<300:
            if epoch % 5 == 0:
                trainer.validation(epoch)
        else:
            trainer.validation(epoch)
