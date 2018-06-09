import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.PyTorchSSIM import SSIM as SSIMLoss


class AttackerLoss(nn.Module):
    def __init__(self,
                 gamma=0.9,
                 use_running_mean=False,
                 ssim_weight=10,
                 l2_weight=1):
        
        super().__init__()
        self.gamma = gamma

        self.ssim_loss = SSIMLoss(window_size = 11)
        self.l2_loss =  self.eucl_loss
        
        self.use_running_mean = use_running_mean
        self.ssim_weight=ssim_weight
        self.l2_weight=l2_weight
        
        if self.use_running_mean == True:
            self.register_buffer('running_similarity_loss', torch.zeros(1))
            self.register_buffer('running_distance_loss', torch.zeros(1))
            self.reset_parameters()

    def mse_loss(self,input,target):
        return torch.sum((input - target)**2) / input.size(0)
    
    def eucl_loss(self,input,target):
        return torch.sum(torch.sqrt(torch.sum((input - target)**2,dim=1))) / input.size(0)     
        
    def reset_parameters(self):
        self.running_similarity_loss.zero_()        
        self.running_distance_loss.zero_()

    def forward(self,
                source_imgs,
                mod_imgs,
                avg_target_descriptions,
                mod_descriptions):
        
        assert len(source_imgs.shape) == 4
        assert len(mod_imgs.shape) == 4
        
        assert len(avg_target_descriptions.shape) == 2
        assert len(mod_descriptions.shape) == 2
        
        assert avg_target_descriptions.size(1) == 512
        assert mod_descriptions.size(1) == 512      
        
        assert avg_target_descriptions.size(1) == 512
        assert mod_descriptions.size(1) == 512         

        assert source_imgs.size()[1:] == (3,112,112)
        assert mod_imgs.size()[1:] == (3,112,112)
      
        # target ssim values are 0.95-1, so loss is expected to be 0 - 0.05
        similarity_loss = 1 - self.ssim_loss(source_imgs,mod_imgs).mean()
        
        distance_loss = self.l2_loss(mod_descriptions,avg_target_descriptions)
            
        if self.use_running_mean == True:
            if similarity_loss.data[0]<0.025:
                smw = 0
                dmw = 1
            else:
                self.running_similarity_loss = self.running_similarity_loss * self.gamma + similarity_loss.data * (1 - self.gamma)        
                self.running_distance_loss = self.running_distance_loss * self.gamma + distance_loss.data * (1 - self.gamma)

                sm = float(self.running_similarity_loss)
                dm = float(self.running_distance_loss)

                smf = sm / (sm + dm)
                dmf = dm / (sm + dm)

                smw = 1 - smf
                dmw = 1 - dmf

        else:
            if similarity_loss.data[0]<0.025:
                smw = 0
                dmw = 1
            else:
                smw = self.ssim_weight
                dmw = self.l2_weight
            
        composite_loss = smw * similarity_loss + dmw * distance_loss   
        
        return composite_loss, similarity_loss, distance_loss
    
class EuclLoss(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.l2_loss =  self.eucl_loss

    def mse_loss(self,input,target):
        return torch.sum((input - target)**2) / input.size(0)
    
    def eucl_loss(self,input,target):
        return torch.sum(torch.sqrt(torch.sum((input - target)**2,dim=1))) / input.size(0)     

    def forward(self,
                avg_target_descriptions,
                mod_descriptions):
        
        assert len(avg_target_descriptions.shape) == 2
        assert len(mod_descriptions.shape) == 2
        
        assert avg_target_descriptions.size(1) == 512
        assert mod_descriptions.size(1) == 512      
       
        distance_loss = self.l2_loss(mod_descriptions,avg_target_descriptions)

        return distance_loss    