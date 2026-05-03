import torch
from torch import nn

from .model_utils.loss import SoftIoULoss, FocalLoss, DiceLoss, SLSIoULoss, structure_loss, SoftIoULoss_list_tuple, DistillKL, DistillKL1, KL_DistillationLoss, Mask_KL_DistillationLoss, DiceLoss, RADLoss


from thop import profile


# Config of Teacher
# ----------------------------------------------------------------------------


# SCTransNet as Teacher
from .SCTransNet_Teacher.SCTransNet import SCTransNet as SCTransNet_Github_Teacher
from .SCTransNet_Teacher.Config import get_SCTrans_config as get_SCTrans_config_Github_Teacher


# DSTransNet as Teacher
from .DSTransNet_Teacher.DSTransNet_Teacher import DSTransNet_Teacher
from .DSTransNet_Teacher.Config import get_DSTransNet_Teacher_config


# Config of Student
# ----------------------------------------------------------------------------

from .LW_DSTransNet_Student.DSTransNet_Student import DSTransNet_Student
# LW-DSTransNet-Nano as Student
from .LW_DSTransNet_Student.Config import get_DSTransNet_Student_config
# LW-DSTransNet-Tiny as Student
from .LW_DSTransNet_Student.Config import get_DSTransNet_Student_config_v5


from .LW_DSTransNet_Student.DSTransNet_Student import DSTransNet_Student_v34
# LW-DSTransNet-Small as Student
from .LW_DSTransNet_Student.Config import get_DSTransNet_Student_config_v34




# Entity of Teacher
# ----------------------------------------------------------------------------
# SCTransNet as Teacher
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

# DSTransNet as Teacher
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



# Entity of Teacher
# ----------------------------------------------------------------------------
# LW-DSTransNet-Nano as Student
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

# LW-DSTransNet-Tiny as Student
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

# LW-DSTransNet-Small as Student
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





# Using python -m Github.model.model to test the model

if __name__ == '__main__':
    

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

