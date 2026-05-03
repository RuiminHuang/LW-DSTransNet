import torch
from torch import nn


from .SCTransNet_Teacher.Tea_SCTrans_02 import get_SCTrans_config
from .SCTransNet_Teacher.Tea_SCTrans_02 import Tea_SCTrans_02

from .SCTransNet_Student.KD_SCTstu01 import get_CTranS_config_Student
from .SCTransNet_Student.KD_SCTstu01 import KD_SCTstu01


from .DSTransNet_Teacher.DSTransNet_Teacher import DSTransNet_Teacher
from .DSTransNet_Teacher.Config import get_DSTransNet_Teacher_config


from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v18
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v19
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v20
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v21
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v22
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v23
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v24
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v25
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v26
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v27
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v28
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v29
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v30
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v31
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v32
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v33
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v34

from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_Adapt_to_HAFNet
from .DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_Adapt_to_SDSNet

from .DSTransNet_Student.Config import get_DSTransNet_Student_config
from .DSTransNet_Student.Config import get_DSTransNet_Student_config_v5
from .DSTransNet_Student.Config import get_DSTransNet_Student_config_v31
from .DSTransNet_Student.Config import get_DSTransNet_Student_config_v32
from .DSTransNet_Student.Config import get_DSTransNet_Student_config_v33
from .DSTransNet_Student.Config import get_DSTransNet_Student_config_v34



from .SCTransNet_Teacher_Github.SCTransNet import SCTransNet as SCTransNet_Github_Teacher
from .SCTransNet_Teacher_Github.Config import get_SCTrans_config as get_SCTrans_config_Github_Teacher


from .SSCFNet_Teacher_Github.SSCFNet import SSCFNet as SSCFNet_Github_Teacher
from .SSCFNet_Teacher_Github.Config import get_config as get_config_SSCFNet_Github_Teacher


from .HAFNet_Teacher_Github.HAFNet import HAFNet as HAFNet_Github_Teacher


from .SDSNet_Teacher_Github.SDSNet import SDSNet as SDSNet_Github_Teacher
from .SDSNet_Teacher_Github.Config import get_config as get_SDS_config_Github_Teacher


from .mkunet_network.mkunet_network import MK_UNet

from .HAFNet.HAFNet import HAFNet




from .model_utils.loss import SoftIoULoss, FocalLoss, DiceLoss, SLSIoULoss, structure_loss, SoftIoULoss_list_tuple, DistillKL, DistillKL1, KL_DistillationLoss, Mask_KL_DistillationLoss, DiceLoss, RADLoss


from thop import profile


class SCTransNet_Teacher(nn.Module):
    
    def __init__(self, mode):
        super(SCTransNet_Teacher, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            config_vit = get_SCTrans_config()
            self.model = Tea_SCTrans_02(config_vit, n_classes=2, mode='train', deepsuper=True)
        elif self.mode == 'test':
            config_vit = get_SCTrans_config()
            self.model = Tea_SCTrans_02(config_vit, n_classes=2, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")

    def forward(self, img):
        return self.model(img)
    

class SCTransNet_Student(nn.Module):
    
    def __init__(self, mode):
        super(SCTransNet_Student, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            config_vit = get_CTranS_config_Student()
            self.model = KD_SCTstu01(config_vit, n_classes=2, mode='train', deepsuper=True)
        elif self.mode == 'test':
            config_vit = get_CTranS_config_Student()
            self.model = KD_SCTstu01(config_vit, n_classes=2, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.hard_loss = nn.CrossEntropyLoss()
        self.KL_loss = DistillKL1(T = 1.0)
        self.criterion_kd_loss = DistillKL(T = 1.0)

    def forward(self, img):
        return self.model(img)
    
    def hard_loss_cal(self, logits, labels):
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT
    
    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)





class SC_TransNet_Github_Teacher(nn.Module):

    def __init__(self, mode):
        super(SC_TransNet_Github_Teacher, self).__init__()

        self.config_vit = get_SCTrans_config_Github_Teacher()
        self.mode = mode


        if self.mode == 'train':
            self.model = SCTransNet_Github_Teacher(config=self.config_vit, n_channels=1, n_classes=2, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = SCTransNet_Github_Teacher(config=self.config_vit, n_channels=1, n_classes=2, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")

        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss() # Part1


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT


class SS_CFNet_Github_Teacher(nn.Module):


    def __init__(self, mode):
        super(SS_CFNet_Github_Teacher, self).__init__()

        self.config_vit = get_config_SSCFNet_Github_Teacher()
        self.mode = mode

        if self.mode == 'train':
            self.model = SSCFNet_Github_Teacher(config=self.config_vit, n_channels=1, n_classes=2, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = SSCFNet_Github_Teacher(config=self.config_vit, n_channels=1, n_classes=2, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss() # Part1


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT


class HA_FNet_Github_Teacher(nn.Module):


    def __init__(self, mode):
        super(HA_FNet_Github_Teacher, self).__init__()


        self.mode = mode

        if self.mode == 'train':
            self.model = HAFNet_Github_Teacher(Train=True)
        elif self.mode == 'test':
            self.model = HAFNet_Github_Teacher(Train=False)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss() # Part1


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            if logits[i] is not None:
                preds = logits[i]
                loss = self.hard_loss(preds, labels)
                loss_all.append( loss )
        # loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4]
        return loss_GT


class SD_SNet_Github_Teacher(nn.Module):

    def __init__(self, mode):
        super(SD_SNet_Github_Teacher, self).__init__()

        self.config_vit = get_SDS_config_Github_Teacher()
        self.mode = mode


        if self.mode == 'train':
            self.model = SDSNet_Github_Teacher(config=self.config_vit, n_channels=1, n_classes=2, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = SDSNet_Github_Teacher(config=self.config_vit, n_channels=1, n_classes=2, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")

        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss() # Part1


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            if logits[i] is not None:
                preds = logits[i]
                loss = self.hard_loss(preds, labels)
                loss_all.append( loss )
        # loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        # loss_GT = loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4]
        return loss_GT



class DS_TransNet_Teacher(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Teacher, self).__init__()

        self.config_vit = get_DSTransNet_Teacher_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Teacher(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Teacher(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss() # Part1


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT


class DS_TransNet_Student(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)



class DS_TransNet_Student_Adapt_to_HAFNet(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_Adapt_to_HAFNet, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_Adapt_to_HAFNet(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_Adapt_to_HAFNet(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)



class DS_TransNet_Student_Adapt_to_SDSNet(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_Adapt_to_SDSNet, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_Adapt_to_SDSNet(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_Adapt_to_SDSNet(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)




class DS_TransNet_Student_v5(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v5, self).__init__()

        self.config_vit = get_DSTransNet_Student_config_v5()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_T4(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_T4, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=4.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 4.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_T8(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_T8, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=8.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 8.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_D7(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_D7, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=7) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_D11(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_D11, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=11) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_S14(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_S14, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_S15(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_S15, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v18(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v18, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v18(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v18(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v19(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v19, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v19(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v19(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v20(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v20, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v20(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v20(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v21(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v21, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v21(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v21(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v22(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v22, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v22(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v22(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v23(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v23, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v23(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v23(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v24(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v24, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v24(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v24(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v25(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v25, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v25(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v25(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v26(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v26, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v26(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v26(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v27(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v27, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v27(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v27(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v28(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v28, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v28(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v28(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v29(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v29, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v29(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v29(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)


class DS_TransNet_Student_v30(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v30, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v30(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v30(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)



class DS_TransNet_Student_v31(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v31, self).__init__()

        self.config_vit = get_DSTransNet_Student_config_v31()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v31(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v31(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)




class DS_TransNet_Student_v32(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v32, self).__init__()

        self.config_vit = get_DSTransNet_Student_config_v32()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v32(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v32(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)




class DS_TransNet_Student_v33(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v33, self).__init__()

        self.config_vit = get_DSTransNet_Student_config_v33()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v33(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v33(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)




class DS_TransNet_Student_v34(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v34, self).__init__()

        self.config_vit = get_DSTransNet_Student_config_v34()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student_v34(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student_v34(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)




class DS_TransNet_Student_v35(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v35, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)



class DS_TransNet_Student_v36(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet_Student_v36, self).__init__()

        self.config_vit = get_DSTransNet_Student_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet_Student(config=self.config_vit, input_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)

        self.hard_loss = nn.CrossEntropyLoss()#reduction='mean' for default # Part1
        # self.KL_loss = DistillKL1(T = 1.0)
        self.KL_loss = KL_DistillationLoss(temperature=6.0, logits_index=4)
        self.mask_KL_loss = Mask_KL_DistillationLoss(temperature = 6.0, logits_index=4) # Part2
        self.Dice_loss = DiceLoss(smooth=1.0, logits_index=4) # Part3
        self.RAD_Loss = RADLoss(num_classes=3, dilation_kernel_size=9) # Part4
        self.encoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.1
        self.skip_loss = nn.MSELoss()#reduction='mean' for default # Part5.2
        self.decoder_loss = nn.MSELoss()#reduction='mean' for default # Part5.3
        self.criterion_kd_loss = DistillKL(T = 1.0)


    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    def hard_loss_cal(self, logits, labels): # Part1
        labels = labels.long().squeeze(1)
        loss_all = []
        for i in range( len(logits) ):
            preds = logits[i]
            loss = self.hard_loss(preds, labels)
            loss_all.append( loss )
        loss_GT = loss_all[0] + loss_all[1] + loss_all[2] + loss_all[3] + loss_all[4] + loss_all[5]
        return loss_GT

    def KL_loss_cal(self, logit_s, logit_t):
        return self.KL_loss(logit_s, logit_t)
    
    def mask_KL_loss_cal(self, logit_s, logit_t, mask): # Part2
        return self.mask_KL_loss(logit_s, logit_t, mask)
    
    def Dice_loss_cal(self, logit_s, labels): # Part3
        return self.Dice_loss(logit_s, labels)
    
    def RAD_loss_cal(self, feature_s, feature_t, gt_masks): # Part4
        loss_all = []
        # loss_all.append( self.RAD_Loss(feature_s[0], feature_t[0], gt_masks) )
        # loss_all.append( self.RAD_Loss(feature_s[1], feature_t[1], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[2], feature_t[2], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[3], feature_t[3], gt_masks) )
        loss_all.append( self.RAD_Loss(feature_s[4], feature_t[4], gt_masks) )
        return sum(loss_all)

    def encoder_loss_cal(self, enc_s, enc_t): # Part5.1
        return self.encoder_loss(enc_s, enc_t)
    
    def skip_loss_cal(self, skip_s, skip_t): # Part5.2
        return self.skip_loss(skip_s, skip_t)
    
    def decoder_loss_cal(self, dec_s, dec_t): # Part5.3
        return self.decoder_loss(dec_s, dec_t)

    def criterion_kd_loss_cal(self, logit_s, logit_t):
        return self.criterion_kd_loss(logit_s, logit_t)





class MK_UNet_Normal(nn.Module):


    def __init__(self, mode):
        super(MK_UNet_Normal, self).__init__()

        self.NET_CONFIGS = {
            'MK_UNet_T': [4, 8, 16, 24, 32],
            'MK_UNet_S': [8, 16, 32, 48, 80],
            'MK_UNet':   [16, 32, 64, 96, 160],
            'MK_UNet_M': [32, 64, 128, 192, 320],
            'MK_UNet_L': [64, 128, 256, 384, 512]
        }
        self.mode = mode

        if self.mode == 'train':
            self.model = MK_UNet(num_classes=1, in_channels=1, channels=self.NET_CONFIGS['MK_UNet'])
        elif self.mode == 'test':
            self.model = MK_UNet(num_classes=1, in_channels=1, channels=self.NET_CONFIGS['MK_UNet'])
        else:
            raise ValueError("Unkown self.mode")

        self.cal_loss = structure_loss()       

    def forward(self, img):
        return self.model(img)[0]
    
    def loss(self, pred, mask):
        return self.cal_loss(pred, mask)



class HAFNet_Normal(nn.Module):


    def __init__(self, mode):
        super(HAFNet_Normal, self).__init__()

        self.mode = mode

        if self.mode == 'train':
            self.model = HAFNet(Train=True)
        elif self.mode == 'test':
            self.model = HAFNet(Train=False)
        else:
            raise ValueError("Unkown self.mode")

        self.cal_loss = SoftIoULoss_list_tuple()

    def forward(self, img):
        return self.model(img)
    
    def loss(self, pred, mask):
        return self.cal_loss(pred, mask)







# Using python -m DSTransNet_Distill.model.model to test the model

if __name__ == '__main__':
    
    # model = DS_TransNet_Teacher(mode='train')
    # input_tensor = torch.randn( (1, 1, 256, 256) )
    # outputs = model(input_tensor)    
    # print( outputs[4].shape)

    # flops, params = profile(model, (input_tensor,))

    # print("-" * 50)
    # print('Params = ' + str(params / 1000 ** 2) + ' M')
    # print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')

    # print("----------------------------------------")


    # model = DS_TransNet_Teacher(mode='test')
    # input_tensor = torch.randn( (1, 1, 256, 256) )
    # outputs = model(input_tensor)    
    # print( outputs[1][4].shape)

    # flops, params = profile(model, (input_tensor,))

    # print("-" * 50)
    # print('Params = ' + str(params / 1000 ** 2) + ' M')
    # print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')

    # print("----------------------------------------")

    # model = DS_TransNet_Student(mode='train')
    # input_tensor = torch.randn( (1, 1, 256, 256) )
    # outputs = model(input_tensor)    
    # print( outputs[1][4].shape)

    # flops, params = profile(model, (input_tensor,))

    # print("-" * 50)
    # print('Params = ' + str(params / 1000 ** 2) + ' M')
    # print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')

    # print("----------------------------------------")

    # model = DS_TransNet_Student(mode='test')
    # input_tensor = torch.randn( (1, 1, 256, 256) )
    # outputs = model(input_tensor)    
    # print( outputs.shape)

    # flops, params = profile(model, (input_tensor,))

    # print("-" * 50)
    # print('Params = ' + str(params / 1000 ** 2) + ' M')
    # print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')


    model = DS_TransNet_Student_v34(mode='train')
    input_tensor = torch.randn( (1, 1, 256, 256) )
    outputs = model(input_tensor)
    
    # for i in  range( len(outputs) ):
    #     if(outputs[i] is not None):
    #         print( outputs[i].shape)


    for i in  range( len(outputs[0]) ):
        if(outputs[0][i] is not None):
            print( outputs[0][i].shape)
    print("----------------------------------------")
    for i in  range( len(outputs[1]) ):
        if(outputs[1][i] is not None):
            print( outputs[1][i].shape)


    flops, params = profile(model, (input_tensor,))

    print("-" * 50)
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')

