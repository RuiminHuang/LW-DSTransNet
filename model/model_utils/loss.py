import torch
import torch.nn as nn
import torch.nn.functional as F



# SoftIoULoss 的 PyTorch 实现
class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)
        smooth = 1
        intersection = pred * target
        
        # 这里保留了BatchSize维度
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        # 对所有的Batch_Size取平均
        loss = 1 - torch.mean(loss)
        return loss


# Focal Loss 的 PyTorch 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        # 计算 logits 的 sigmoid 概率
        p = torch.sigmoid(inputs)

        # 计算交叉熵损失，直接处理input，所以必须用with_logits
        # 返回的还是[BCHW]
        # 注意，这里inputs和targets的shape必须一致，并且为BCHW，C为1表示对1个类别进行二分类，binary_cross_entropy函数同理
        # 这一点和cross_entropy有明显区别，因为cross_entropy没要求inputs和targets的shape必须一致
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算 Focal Loss
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha * ((1 - p) ** self.gamma) * bce_loss

        # 根据 reduction 参数进行损失归约
        # 直接对一整个[BCHW]的数据进行reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




# 注意，这里的target为0~n_classes-1的整数组成的单通道的Mask，在我们的数据预处理中，0~255被处理成为了0~1，凑巧0代表背景，1(255)代表目标，所以可以直接用。
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes



class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def LLoss(self, pred, target):
            
            loss = torch.tensor(0.0, requires_grad=True).to(pred)

            batch_size = pred.shape[0]
            h = pred.shape[2]
            w = pred.shape[3]        
            x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
            y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
            smooth = 1e-8
            
            for i in range(batch_size):  

                pred_centerx = (x_index*pred[i]).mean()
                pred_centery = (y_index*pred[i]).mean()

                target_centerx = (x_index*target[i]).mean()
                target_centery = (y_index*target[i]).mean()
            
                angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                                - torch.arctan((target_centery) / (target_centerx + smooth))))

                pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
                target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)
                
                length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
            
                loss = loss + (1 - length_loss + angle_loss) / batch_size
            
            return loss

    def forward(self, pred_log, target,warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum  + smooth)       
        lloss = self.LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss



# 注意，这里的target为0~n_classes-1的整数组成的单通道的Mask，在我们的数据预处理中，0~255被处理成为了0~1，凑巧0代表背景，1(255)代表目标，所以可以直接用。此时mask的shape为[B, H, W]
class CrossEntropyLoss_Warp(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_Warp, self).__init__()

    def forward(self, inputs, target):
        return nn.functional.cross_entropy(inputs, target)



class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)



class structure_loss(nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()
        self.w=1

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (self.w * (wbce + wiou)).mean()





class SoftIoULoss_list_tuple(nn.Module):
    def __init__(self):
        super(SoftIoULoss_list_tuple, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            mp= nn.MaxPool2d(2, 2)
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                if i>5:
                    gt_masks=mp(gt_masks)
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss






class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    # def forward(self, y_s, y_t):
    #     p_s = F.log_softmax(y_s/self.T, dim=1)
    #     p_t = F.softmax(y_t/self.T, dim=1)
    #     loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
    #     return loss

    def forward(self, y_s, y_t):
        a = []
        for ii in range(len(y_s)):
            y_ss = y_s[ii]
            y_tt = y_t[ii]
            kl_loss = F.kl_div(F.logsigmoid(y_ss / self.T),
                               torch.sigmoid(y_tt.detach() / self.T)) * self.T ** 2
            a.append(kl_loss)
        loss_kl_loss = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
        return loss_kl_loss


class DistillKL1(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL1, self).__init__()
        self.T = T

    # def forward(self, y_s, y_t):
    #     p_s = F.log_softmax(y_s/self.T, dim=1)
    #     p_t = F.softmax(y_t/self.T, dim=1)
    #     loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
    #     return loss

    def forward(self, y_s, y_t):
        # a = []
        # for ii in range(len(y_s)):
        #     y_ss = y_s[ii]
        #     y_tt = y_t[ii]
        #     kl_loss = F.kl_div(F.logsigmoid(y_ss / self.T),
        #                        torch.sigmoid(y_tt.detach() / self.T)) * self.T ** 2
        #     a.append(kl_loss)
        # loss_kl_loss = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]

        loss_kl_loss = F.kl_div(F.logsigmoid(y_s[4] / self.T),
                                torch.sigmoid(y_t[4].detach() / self.T) ) * self.T ** 2
        # x = y_s[-1]

        # loss_kl_loss = F.kl_div(F.log_softmax(y_s/ self.T, dim=1),
        #                       F.softmax(y_t[-1].detach() / self.T, dim=1))
        # print(loss_kl_loss)
        return loss_kl_loss



class KL_DistillationLoss(nn.Module):
    def __init__(self, temperature=6.0, logits_index=4):
        super(KL_DistillationLoss, self).__init__()
        self.T = temperature
        self.logits_index = logits_index
        # reduction='batchmean' 是 KLDivLoss 在数学上最准确的 reduction 方式
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        """
        输入:
            student_logits: [B, 2, H, W] (未经过 softmax/sigmoid)
            teacher_logits: [B, 2, H, W] (未经过 softmax/sigmoid)
        """

        B, C, H, W = student_logits[self.logits_index].shape

        # 1. 针对学生模型：除以 T 并做 LogSoftmax
        # dim=1 表示在通道维度(背景/前景)上做归一化
        student_log_soft = F.log_softmax(student_logits[self.logits_index] / self.T, dim=1)
        
        # 2. 针对教师模型：除以 T 并做 Softmax
        # 教师模型不需要梯度，所以用 torch.no_grad() 或者 detach()
        with torch.no_grad():
            teacher_soft = F.softmax(teacher_logits[self.logits_index] / self.T, dim=1)
            
        # 3. 计算 KL 散度
        # KLDivLoss 要求输入形状一致。
        # 注意：KL 散度是像素级的，但通常我们会对整个 Batch 求平均
        loss = self.kl_div(student_log_soft, teacher_soft)
        
        # 4. 乘以 T^2 缩放梯度
        loss = ( loss / (H * W) ) * (self.T ** 2)
        
        return loss





class Mask_KL_DistillationLoss(nn.Module):
    def __init__(self, temperature=6.0, logits_index=4):
        super(Mask_KL_DistillationLoss, self).__init__()
        self.T = temperature
        self.logits_index = logits_index
        # 注意：这里改为 'none'，因为我们需要先拿到像素级的 loss，再乘掩码
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, student_logits, teacher_logits, targets):
        """
        输入:
            student_logits: [B, 2, H, W]
            teacher_logits: [B, 2, H, W]
            targets: [B, H, W] (Ground Truth 标签, 0 or 1)
        """
        # 处理 List 输入的情况
        if isinstance(student_logits, (list, tuple)):
            s_in = student_logits[self.logits_index]
            t_in = teacher_logits[self.logits_index]
        else:
            s_in = student_logits
            t_in = teacher_logits

        B, C, H, W = s_in.shape

        # 1. 准备输入
        s_log_prob = F.log_softmax(s_in / self.T, dim=1)
        
        # 教师不需要梯度
        with torch.no_grad():
            t_prob = F.softmax(t_in / self.T, dim=1)
            # 获取教师的硬预测 (Hard Prediction)
            teacher_pred = t_in.argmax(dim=1)  # [B, H, W]

        # 2. 计算像素级 KL 散度
        # 输出 shape: [B, 2, H, W] -> sum(dim=1) -> [B, H, W]
        kl_map = self.kl_div(s_log_prob, t_prob).sum(dim=1)

        # 3. 生成掩码 (Mask)
        # 逻辑：只有当 Teacher 预测正确 (Teacher == GT) 时，mask = 1，否则 = 0
        # targets 需要确保是 [B, H, W]
        if targets.dim() == 4: 
            targets = targets.squeeze(1)
            
        mask = (teacher_pred == targets).float()

        # 4. 计算 Masked Loss
        # 只对 mask 区域求平均
        # 加上 1e-6 防止分母为 0 (虽然极不可能)
        loss = (kl_map * mask).sum() / (mask.sum() + 1e-6)

        # 5. 温度缩放
        loss = loss * (self.T ** 2)

        return loss




class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, logits_index=4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.logits_index = logits_index
    
    def forward(self, logits, targets):
        # logits: [B, 2, H, W]
        # targets: [B, H, W] (0 or 1)
        
        # 只取前景类 (Channel 1) 的概率
        probs = torch.softmax(logits[self.logits_index], dim=1)[:, 1, :, :]
        targets = targets.squeeze(1)  # 如果 targets 是 [B, 1, H, W]，则 squeeze 成 [B, H, W]
        
        # 展平
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        
        # Dice 公式
        loss = 1 - ((2. * intersection + self.smooth) / 
                    (probs_flat.sum() + targets_flat.sum() + self.smooth))
        return loss







class RADLoss(nn.Module):
    """
    Region Affinity Distillation (RAD) Module
    Reference: Efficient Medical Image Segmentation Based on Knowledge Distillation (IEEE TMI 2021)
    """
    def __init__(self, num_classes=3, dilation_kernel_size=5):
        super(RADLoss, self).__init__()
        self.num_classes = num_classes

        # 强制检查 kernel size 为奇数，防止尺寸错位
        if dilation_kernel_size % 2 == 0:
            print(f"[Warning] Even kernel size {dilation_kernel_size} usually causes dimension mismatch.")
            print(f"Auto-adjusting kernel size to {dilation_kernel_size + 1}")
            dilation_kernel_size += 1
        self.dilation_kernel_size = dilation_kernel_size

    def _get_region_prototypes(self, features, masks):
        """
        对应论文公式 (6): 计算每个语义类别的区域特征表示 (Ri)
        
        Args:
            features: (B, C, H, W) - 来自Teacher或Student的特征图
            masks: (B, H, W) - Ground Truth Label，值为 0 到 num_classes-1
        Returns:
            prototypes: (B, num_classes, C) - 每个类别的特征向量
        """
        B, C, H, W = features.shape
        
        # 1. 调整Mask尺寸以匹配Feature Map的空间尺寸 (W x H -> w x h)
        # 使用最近邻插值保持类别标签为整数
        masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
        
        # 2. 将Mask转换为One-Hot编码: (B, H, W) -> (B, num_classes, H, W)
        one_hot_masks = F.one_hot(masks_resized, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # 3. 计算区域特征总和
        # 使用 Einstein Summation: bchw (features) * bkhw (masks) -> bkc (class prototypes)
        # 这相当于 masked features 的 sum
        region_sum = torch.einsum('bchw,bkhw->bkc', features, one_hot_masks)
        
        # 4. 计算每个区域的像素数量 (Area)
        # (B, num_classes, H, W) -> (B, num_classes)
        region_area = one_hot_masks.sum(dim=(2, 3)) 
        
        # 5. 计算平均值得到 Ri (添加 epsilon 防止除以 0)
        # (B, num_classes, C)
        prototypes = region_sum / (region_area.unsqueeze(-1) + 1e-6)
        
        return prototypes

    def _get_region_affinity_matrix(self, prototypes):
        """
        对应论文公式 (7): 计算区域亲和度/对比度矩阵 (Vrc)
        使用余弦相似度
        
        Args:
            prototypes: (B, num_classes, C)
        Returns:
            affinity_matrix: (B, num_classes, num_classes)
        """
        # 1. L2 归一化: Ri / ||Ri||
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # 2. 计算余弦相似度矩阵: R_norm * R_norm^T
        # (B, K, C) @ (B, C, K) -> (B, K, K)
        affinity_matrix = torch.bmm(prototypes_norm, prototypes_norm.transpose(1, 2))
        
        return affinity_matrix


    def _convert_to_3class_mask(self, gt_mask):
        """
        假设 gt_mask 是二分类 (B, H, W)，0为背景，1为前景
        我们需要将其转换为 3 分类：0-远端背景, 1-前景, 2-局部背景
        """

        # 1. 确保输入是 float 以进行 max_pool (模拟膨胀操作)
        mask_float = gt_mask.unsqueeze(1).float() # (B, 1, H, W)
        
        # 2. 膨胀操作：获取包含前景和周围一圈的区域
        dilated = torch.nn.functional.max_pool2d(
            mask_float, 
            kernel_size=self.dilation_kernel_size, 
            stride=1, 
            padding=self.dilation_kernel_size // 2
        ).squeeze(1).long() # (B, H, W)
        
        # 3. 构建新 Mask
        # 初始化全为 0 (远端背景)
        new_mask = torch.zeros_like(gt_mask).long()
        
        # 标记局部背景 (在膨胀范围内，但不是原始前景)
        # 设局部背景为 label 2
        new_mask[(dilated == 1) & (gt_mask == 0)] = 2 
        
        # 标记前景 (保持 label 1)
        new_mask[gt_mask == 1] = 1

        # 剩下的保持为 0 (Global BG)
        
        return new_mask




    def forward(self, student_features, teacher_features, gt_masks):
        """
        Args:
            student_features: (B, C_s, H_s, W_s) 建议使用浅层特征 (Stride=2 或 4)
            teacher_features: (B, C_t, H_t, W_t) 
            gt_masks: (B, H_ori, W_ori) 原始二值掩码
        """

        # 输入维度兼容性处理 (B, 1, H, W) -> (B, H, W)
        if gt_masks.dim() == 4: 
            gt_masks = gt_masks.squeeze(1)

        # 核心修改：动态生成三分类 Mask
        # 注意：这一步是在原始分辨率 Mask 上进行的，保留了小目标的形态结构
        gt_masks_3class = self._convert_to_3class_mask(gt_masks)

        # 论文中提到需要对齐特征通道或尺寸，通常 RAD 模块中，
        # 核心是通过 Affinity Matrix (KxK) 进行比较，所以特征通道 C 可以不同，
        # 因为计算出的相似度矩阵尺寸只与类别数 K 有关。
        
        # 1. 计算 Student 的区域关系矩阵
        s_prototypes = self._get_region_prototypes(student_features, gt_masks_3class)
        s_affinity = self._get_region_affinity_matrix(s_prototypes)
        
        # 2. 计算 Teacher 的区域关系矩阵
        # 教师网络在推理阶段通常不需要计算梯度
        with torch.no_grad():
            t_prototypes = self._get_region_prototypes(teacher_features, gt_masks_3class)
            t_affinity = self._get_region_affinity_matrix(t_prototypes)
        
        # 3. 计算蒸馏损失 (公式 8)
        # 论文中使用 L2 范数 (MSE Loss) 计算两个矩阵的差异
        # 只要类别存在，矩阵就是 K x K，直接计算 MSE
        loss = F.mse_loss(s_affinity, t_affinity)
        
        return loss










if __name__ == '__main__':

    inputs = torch.rand(16, 1, 256, 256, requires_grad=True)
    targets = torch.ones(16, 1, 256, 256, dtype=torch.float32)

    print("input:")
    print(inputs.shape)
    print("target:")
    print(targets.shape)


    criterion = SoftIoULoss()
    loss = criterion(inputs, targets)
    print("SoftIoULoss:")
    print(loss.shape)
    print(loss)


    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    loss = criterion(inputs, targets)
    print("FocalLoss:")
    print(loss.shape)
    print(loss)


    n_classes = 2
    batch_size = 16
    criterion_dice = DiceLoss(n_classes)
    criterion_ce = CrossEntropyLoss_Warp()
    inputs = torch.rand(batch_size, n_classes, 256, 256, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, 256, 256)).float()
    print(inputs.shape)
    print(targets.shape)
    loss = criterion_dice(inputs, targets)
    print(f'Dice Loss: {loss}')
    loss = criterion_ce(inputs, targets[:].long())# same with: loss = criterion_ce(inputs, targets.long())
    print(f'CE Loss: {loss}')
