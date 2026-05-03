import torch
from thop import profile

from model.model import DS_TransNet_Student, DS_TransNet_Student_v5, DS_TransNet_Student_v34

if __name__ == '__main__':
    
    # LW-DSTransNet-Nano
    model = DS_TransNet_Student(mode='train')
    input_tensor = torch.randn( (1, 1, 256, 256) )
    flops, params = profile(model, (input_tensor,))
    print("-" * 50)
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')

    # LW-DSTransNet-Tiny
    model = DS_TransNet_Student_v5(mode='train')
    input_tensor = torch.randn( (1, 1, 256, 256) )
    flops, params = profile(model, (input_tensor,))
    print("-" * 50)
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')

    # LW-DSTransNet-Small
    model = DS_TransNet_Student_v34(mode='train')
    input_tensor = torch.randn( (1, 1, 256, 256) )
    flops, params = profile(model, (input_tensor,))
    print("-" * 50)
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')