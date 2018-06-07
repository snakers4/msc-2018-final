import torch
import torch.nn as nn
import torchvision.models as models

def load_model(arch='resnet18',
               pretrained=True):
    if arch.startswith('resnet') :
        model = models.__dict__[arch](pretrained=pretrained)
    elif arch.startswith('densenet'):
        model = models.__dict__[arch](pretrained=True)
    else :
        raise("Finetuning not supported on this architecture yet") 
    return model


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        self.num_classes = num_classes
        
        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(original_model.children())[:-3],
                nn.AvgPool2d(7, stride=1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, num_classes)
            )
            self.modelName = 'resnet'
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)            
        elif arch.startswith('densenet161'):
            self.features = nn.Sequential(
                *list(original_model.features.children())[:-3],
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1)        
             )
            self.classifier = nn.Sequential(
                nn.Linear(2112, num_classes)                   
            )
            self.modelName = 'densenet'
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)   
        else :
            raise("Finetuning not supported on this architecture yet")
    
    def freeze(self):
        print('Features frozen')
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self):
        print('Features unfrozen')
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True            
            
    def forward(self, x):
        f = self.features(x)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        elif self.modelName == 'densenet' :   
            f = f.view(f.size(0), -1)

        y = self.classifier(f) 
        return y