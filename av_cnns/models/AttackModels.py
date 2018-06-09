import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.StudentModels import load_model, FineTuneModel
from models.Decoder import DecoderBlockLinkNetV2 as DecoderBlock

nonlinearity = nn.ReLU

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Attacker34(nn.Module):
    def __init__(self,
                 num_classes=3,
                 num_channels=3,
                 is_deconv = False,
                 decoder_kernel_size=4,
                 checkpoint='weights/resnet18_scale_fold0_best.pth.tar',
                 noise_multiplier=1.0
                ):
        super().__init__()

        filters = [64*2, 128*2, 256*2]
        
        self.noise_multiplier = noise_multiplier

        resnet_student = load_model('resnet18',pretrained=True)
        resnet_student = FineTuneModel(resnet_student,'resnet34',512)
        resnet_student = torch.nn.DataParallel(resnet_student)
        checkpoint = torch.load(checkpoint)
        print('Model loaded from epoch {} checkpoint'.format(checkpoint['epoch']))
        resnet_student.load_state_dict(checkpoint['state_dict'])
        resnet_student = resnet_student.module        
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # self.reverse_mean = [-0.485, -0.456, -0.406]
        # self.reverse_std = [1/0.229, 1/0.224, 1/0.225]

        self.firstconv = resnet_student.features[0]
        self.firstbn = resnet_student.features[1]
        self.firstrelu = resnet_student.features[2]
        self.firstmaxpool = resnet_student.features[3]
        
        self.encoder1 = resnet_student.features[4]
        self.encoder2 = resnet_student.features[5]
        self.encoder3 = resnet_student.features[6]

        self.final_avg = nn.AvgPool2d(7, stride=1)
        self.classifier = resnet_student.classifier[0]
        
        # Decoder
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)                                     
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
    
    def forward(self,
               mode,
               source,
               target,
               target_repeat
               ):
        
        if mode=='attack':
            return self.forward_attack(source,target)
        elif mode=='eval':
            return self.forward_eval(source,target,target_repeat)
        elif mode=='test':
            return self.forward_submit(source,target)
        elif mode=='attack_model_loss':
            return self.attack_model_loss(source,target)
        elif mode=='eval_model_loss':
            return self.eval_model_loss(source,target_repeat)
        
    def freeze_encoder(self):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3]
        
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False

    def forward_noise(self, 
                      source,
                      target):
        
        # just a combination of images
        
        # Encoder
        e1_s,e2_s,e3_s = self.forward_encoder(source)
        e1_t,e2_t,e3_t = self.forward_encoder(target)

        e3_t = self.encoder3(e2_t)   
        
        e1 = torch.cat([e1_s,e1_t],dim=1)
        e2 = torch.cat([e2_s,e2_t],dim=1)
        e3 = torch.cat([e3_s,e3_t],dim=1)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5
    
    def forward_encoder(self,
                        x):
        
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        return e1,e2,e3

    def forward_noise5(self, 
                      source,
                      target):
        
        # using all 5 images to calculate average
        # assuming that inputs are
        # source BxCxWxH
        # target Bx5xCxWxH, i.e. all 5 images
        
        assert len(source.size()) == 4
        assert len(target.size()) == 5
        assert target.size(1) == 5
        
        # Encoder
        e1_s,e2_s,e3_s = self.forward_encoder(source)
        
        # print(e1_s.shape)
        # print(e2_s.shape)
        # print(e3_s.shape)        
        
        e1_ts = []
        e2_ts = []
        e3_ts = []
        
        for i in range(0,5):
            e1_t,e2_t,e3_t = self.forward_encoder(target[:,i,:,:,:])
            e1_ts.append(e1_t)
            e2_ts.append(e2_t)
            e3_ts.append(e3_t)
            
        e1_t = torch.stack(e1_ts,dim=1).mean(dim=1)
        e2_t = torch.stack(e2_ts,dim=1).mean(dim=1)
        e3_t = torch.stack(e3_ts,dim=1).mean(dim=1)
        
        # print(torch.stack(e1_ts,dim=1).shape)
        # print(torch.stack(e2_ts,dim=1).shape)
        # print(torch.stack(e3_ts,dim=1).shape)        
        
        # print(e1_t.shape)
        # print(e2_t.shape)
        # print(e3_t.shape)

        e1 = torch.cat([e1_s,e1_t],dim=1)
        e2 = torch.cat([e2_s,e2_t],dim=1)
        e3 = torch.cat([e3_s,e3_t],dim=1)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5    
    
    def forward_descriptor(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        e3 = self.final_avg(e3) 
             
        return self.classifier(e3.view(e3.size(0), -1))    
    
    def attack_model_loss(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==6
        assert target.size(1)== 5
        assert target.size(2)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
                             
        source_mod = self.forward_noise5(source,
                                    target)
        
        return source_mod
    
    def eval_model_loss(self,
                        source,
                        target_repeat):
        
        # input tensors should be Bx25xCxWxH
        assert len(source.size())==5
        
        assert (source.size(1))==25
        
        assert len(target_repeat.size())==6
        assert target_repeat.size(1)==25
        assert target_repeat.size(2)==5
        
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target_repeat = target_repeat.view(-1,
                                           target_repeat.size(2),
                                           target_repeat.size(3),
                                           target_repeat.size(4),
                                           target_repeat.size(5))

        # we use all of the 5 target imgs to calculate noise        
        source_mod = self.forward_noise5(source,
                                         target_repeat)    

        return source_mod
    
    def forward_attack(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==6
        assert target.size(1)== 5
        assert target.size(2)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source + noise * self.noise_multiplier
        
        # print(source.shape)
        # print(noise.shape)
        
        modified_descriptions = self.forward_descriptor(source_mod)
     
        avg_target_descriptions = []
        
        for i in range(0,5):
            avg_target_descriptions.append(self.forward_descriptor(target[:,i,:,:,:])) 
            
        avg_target_descriptions = torch.stack(avg_target_descriptions,dim=1).mean(dim=1)
        
        return source_mod,modified_descriptions,avg_target_descriptions
    
    def forward_eval(self,
                     source,
                     target,
                     target_repeat):
        
        # input tensors should be Bx25xCxWxH
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert (source.size(1))==25
        assert (target.size(1))==25
        
        assert len(target_repeat.size())==6
        assert target_repeat.size(1)==25
        assert target_repeat.size(2)==5
        
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4))
        target_repeat = target_repeat.view(-1,
                                           target_repeat.size(2),
                                           target_repeat.size(3),
                                           target_repeat.size(4),
                                           target_repeat.size(5))

        # we use all of the 5 target imgs to calculate noise        
        noise = self.forward_noise5(source,
                                    target_repeat)    
        
        source_mod = source + noise * self.noise_multiplier
        
        # we use only image pairs to calculate descriptors
        source_descriptions = self.forward_descriptor(source_mod)
        target_descriptions = self.forward_descriptor(target)        

        # we use all of the 5 target imgs to calculate noise
        
        return source_mod,source_descriptions,target_descriptions
    
    def forward_submit(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert source.size(1)== 1
        assert target.size(1)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source + noise * self.noise_multiplier

        # assuming x and y are Batch x 3 x H x W and mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
        source_mod_renorm = source_mod.clone()
        source_mod_renorm[:, 0, :, :] = source_mod[:, 0, :, :] * self.std[0] + self.mean[0]
        source_mod_renorm[:, 1, :, :] = source_mod[:, 1, :, :] * self.std[1] + self.mean[1]
        source_mod_renorm[:, 2, :, :] = source_mod[:, 2, :, :] * self.std[2] + self.mean[2]
        
        source_mod_renorm[source_mod_renorm > 1] = 1
        source_mod_renorm[source_mod_renorm < 0] = 0   
        
        source_mod_renorm = source_mod_renorm.squeeze(0)

        return source_mod_renorm
    
class Attacker34_noskip(nn.Module):
    def __init__(self,
                 num_classes=3,
                 num_channels=3,
                 is_deconv = False,
                 decoder_kernel_size=4,
                 checkpoint='weights/resnet34_scale_fold0_best.pth.tar',
                 noise_multiplier=1.0
                ):
        super().__init__()

        filters = [64*2, 128*2, 256*2]
        
        self.noise_multiplier = noise_multiplier

        resnet_student = load_model('resnet34',pretrained=True)
        resnet_student = FineTuneModel(resnet_student,'resnet34',512)
        resnet_student = torch.nn.DataParallel(resnet_student)
        checkpoint = torch.load(checkpoint)
        print('Model loaded from epoch {} checkpoint'.format(checkpoint['epoch']))
        resnet_student.load_state_dict(checkpoint['state_dict'])
        resnet_student = resnet_student.module        
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # self.reverse_mean = [-0.485, -0.456, -0.406]
        # self.reverse_std = [1/0.229, 1/0.224, 1/0.225]

        self.firstconv = resnet_student.features[0]
        self.firstbn = resnet_student.features[1]
        self.firstrelu = resnet_student.features[2]
        self.firstmaxpool = resnet_student.features[3]
        
        self.encoder1 = resnet_student.features[4]
        self.encoder2 = resnet_student.features[5]
        self.encoder3 = resnet_student.features[6]

        self.final_avg = nn.AvgPool2d(7, stride=1)
        self.classifier = resnet_student.classifier[0]
        
        # Decoder
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)                                     
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
    
    def forward(self,
               mode,
               source,
               target,
               target_repeat
               ):
        
        if mode=='attack':
            return self.forward_attack(source,target)
        elif mode=='eval':
            return self.forward_eval(source,target,target_repeat)
        elif mode=='test':
            return self.forward_submit(source,target)
    
    def freeze_encoder(self):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3]
        
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
    
    def forward_encoder(self,
                        x):
        
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        return e3

    def forward_noise5(self, 
                      source,
                      target):
        
        # using all 5 images to calculate average
        # assuming that inputs are
        # source BxCxWxH
        # target Bx5xCxWxH, i.e. all 5 images
        
        assert len(source.size()) == 4
        assert len(target.size()) == 5
        assert target.size(1) == 5
        
        # Encoder
        e3_s = self.forward_encoder(source)

        e3_ts = []
        
        for i in range(0,5):
            e3_t = self.forward_encoder(target[:,i,:,:,:])
            e3_ts.append(e3_t)
            
        e3_t = torch.stack(e3_ts,dim=1).mean(dim=1)
        e3 = torch.cat([e3_s,e3_t],dim=1)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5    
    
    def forward_descriptor(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        e3 = self.final_avg(e3) 
             
        return self.classifier(e3.view(e3.size(0), -1))    
    
    def forward_attack(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==6
        assert target.size(1)== 5
        assert target.size(2)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source + noise * self.noise_multiplier
        
        # print(source.shape)
        # print(noise.shape)
        
        modified_descriptions = self.forward_descriptor(source_mod)
     
        avg_target_descriptions = []
        
        for i in range(0,5):
            avg_target_descriptions.append(self.forward_descriptor(target[:,i,:,:,:])) 
            
        avg_target_descriptions = torch.stack(avg_target_descriptions,dim=1).mean(dim=1)
        
        return source_mod,modified_descriptions,avg_target_descriptions
    
    def forward_eval(self,
                     source,
                     target,
                     target_repeat):
        
        # input tensors should be Bx25xCxWxH
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert (source.size(1))==25
        assert (target.size(1))==25
        
        assert len(target_repeat.size())==6
        assert target_repeat.size(1)==25
        assert target_repeat.size(2)==5
        
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4))
        target_repeat = target_repeat.view(-1,
                                           target_repeat.size(2),
                                           target_repeat.size(3),
                                           target_repeat.size(4),
                                           target_repeat.size(5))

        # we use all of the 5 target imgs to calculate noise        
        noise = self.forward_noise5(source,
                                    target_repeat)    
        
        source_mod = source + noise * self.noise_multiplier
        
        # we use only image pairs to calculate descriptors
        source_descriptions = self.forward_descriptor(source_mod)
        target_descriptions = self.forward_descriptor(target)        

        # we use all of the 5 target imgs to calculate noise
        
        return source_mod,source_descriptions,target_descriptions
    
    def forward_submit(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert source.size(1)== 1
        assert target.size(1)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source + noise * self.noise_multiplier

        # assuming x and y are Batch x 3 x H x W and mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
        source_mod_renorm = source_mod.clone()
        source_mod_renorm[:, 0, :, :] = source_mod[:, 0, :, :] * self.std[0] + self.mean[0]
        source_mod_renorm[:, 1, :, :] = source_mod[:, 1, :, :] * self.std[1] + self.mean[1]
        source_mod_renorm[:, 2, :, :] = source_mod[:, 2, :, :] * self.std[2] + self.mean[2]
        
        source_mod_renorm[source_mod_renorm > 1] = 1
        source_mod_renorm[source_mod_renorm < 0] = 0   
        
        source_mod_renorm = source_mod_renorm.squeeze(0)

        return source_mod_renorm

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.norm = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)    
    
class NaiveDecoder(nn.Module):
    def __init__(self):
        super(NaiveDecoder, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(1024, 2352),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ) 
        
        self.sdecoder = nn.Sequential(
            
            ConvRelu(48, 48),
            nn.ConvTranspose2d(48,
                               24,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU(inplace=True),
            
            ConvRelu(24, 24),
            nn.ConvTranspose2d(24,
                               12,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU(inplace=True),
            
            ConvRelu(12, 12),
            nn.ConvTranspose2d(12,
                               6,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU(inplace=True),
            
            ConvRelu(6, 6),
            nn.ConvTranspose2d(6,
                               3,
                               kernel_size=4, stride=2,
                               padding=1),
            torch.nn.Sigmoid()
        ) 

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1,48,7,7)
        x = self.sdecoder(x)
        return x
        
class Attacker34_only_vector(nn.Module):
    def __init__(self,
                 num_classes=3,
                 num_channels=3,
                 is_deconv = False,
                 decoder_kernel_size=4,
                 checkpoint='weights/resnet34_scale_fold0_best.pth.tar',
                 noise_multiplier=1.0
                ):
        super().__init__()

        filters = [64*2, 128*2, 256*2]
        
        self.noise_multiplier = noise_multiplier

        resnet_student = load_model('resnet34',pretrained=True)
        resnet_student = FineTuneModel(resnet_student,'resnet34',512)
        resnet_student = torch.nn.DataParallel(resnet_student)
        checkpoint = torch.load(checkpoint)
        print('Model loaded from epoch {} checkpoint'.format(checkpoint['epoch']))
        resnet_student.load_state_dict(checkpoint['state_dict'])
        resnet_student = resnet_student.module        
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # self.reverse_mean = [-0.485, -0.456, -0.406]
        # self.reverse_std = [1/0.229, 1/0.224, 1/0.225]

        self.firstconv = resnet_student.features[0]
        self.firstbn = resnet_student.features[1]
        self.firstrelu = resnet_student.features[2]
        self.firstmaxpool = resnet_student.features[3]
        
        self.encoder1 = resnet_student.features[4]
        self.encoder2 = resnet_student.features[5]
        self.encoder3 = resnet_student.features[6]

        self.final_avg = nn.AvgPool2d(7, stride=1)
        self.classifier = resnet_student.classifier[0]
        
        # Decoder
        self.decoder = NaiveDecoder()
    
    def forward(self,
               mode,
               source,
               target,
               target_repeat
               ):
        
        if mode=='attack':
            return self.forward_attack(source,target)
        elif mode=='eval':
            return self.forward_eval(source,target,target_repeat)
        elif mode=='test':
            return self.forward_submit(source,target)
    
    def freeze_encoder(self):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3]
        
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
    
    def forward_noise5(self, 
                      source,
                      target):
        
        # using all 5 images to calculate average
        # assuming that inputs are
        # source BxCxWxH
        # target Bx5xCxWxH, i.e. all 5 images
        
        assert len(source.size()) == 4
        assert len(target.size()) == 5
        assert target.size(1) == 5
        
        # Encoder
        d_s = self.forward_descriptor(source)

        d_ts = []
        
        for i in range(0,5):
            d_t = self.forward_descriptor(target[:,i,:,:,:])
            d_ts.append(d_t)
            
        d_t = torch.stack(d_ts,dim=2).mean(dim=2)
        d = torch.cat([d_s,d_t],dim=1)

        # Decoder with Skip Connections
        out = self.decoder(d)

        return out
    
    def forward_descriptor(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        e3 = self.final_avg(e3) 
             
        return self.classifier(e3.view(e3.size(0), -1))    
    
    def forward_attack(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==6
        assert target.size(1)== 5
        assert target.size(2)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
                             
        noise = self.forward_noise5(source,
                                    target)

        
        source_mod = source * noise * self.noise_multiplier
        
        # print(source.shape)
        # print(noise.shape)
        
        modified_descriptions = self.forward_descriptor(source_mod)
     
        avg_target_descriptions = []
        
        for i in range(0,5):
            avg_target_descriptions.append(self.forward_descriptor(target[:,i,:,:,:])) 
            
        avg_target_descriptions = torch.stack(avg_target_descriptions,dim=1).mean(dim=1)
        
        return source_mod,modified_descriptions,avg_target_descriptions
    
    def forward_eval(self,
                     source,
                     target,
                     target_repeat):
        
        # input tensors should be Bx25xCxWxH
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert (source.size(1))==25
        assert (target.size(1))==25
        
        assert len(target_repeat.size())==6
        assert target_repeat.size(1)==25
        assert target_repeat.size(2)==5
        
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4))
        target_repeat = target_repeat.view(-1,
                                           target_repeat.size(2),
                                           target_repeat.size(3),
                                           target_repeat.size(4),
                                           target_repeat.size(5))

        # we use all of the 5 target imgs to calculate noise        
        noise = self.forward_noise5(source,
                                    target_repeat)    
        
        source_mod = source * noise * self.noise_multiplier
        
        # we use only image pairs to calculate descriptors
        source_descriptions = self.forward_descriptor(source_mod)
        target_descriptions = self.forward_descriptor(target)        

        # we use all of the 5 target imgs to calculate noise
        
        return source_mod,source_descriptions,target_descriptions
    
    def forward_submit(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert source.size(1)== 1
        assert target.size(1)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source * noise * self.noise_multiplier

        # assuming x and y are Batch x 3 x H x W and mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
        source_mod_renorm = source_mod.clone()
        source_mod_renorm[:, 0, :, :] = source_mod[:, 0, :, :] * self.std[0] + self.mean[0]
        source_mod_renorm[:, 1, :, :] = source_mod[:, 1, :, :] * self.std[1] + self.mean[1]
        source_mod_renorm[:, 2, :, :] = source_mod[:, 2, :, :] * self.std[2] + self.mean[2]
        
        source_mod_renorm[source_mod_renorm > 1] = 1
        source_mod_renorm[source_mod_renorm < 0] = 0   
        
        source_mod_renorm = source_mod_renorm.squeeze(0)

        return source_mod_renorm    
    
class Attacker34E2E(nn.Module):
    def __init__(self,
                 num_classes=3,
                 num_channels=3,
                 is_deconv = False,
                 decoder_kernel_size=4,
                 checkpoint='weights/resnet18_scale_fold0_best.pth.tar',
                 noise_multiplier=1.0
                ):
        super().__init__()

        filters = [64*2, 128*2, 256*2]
        
        self.noise_multiplier = noise_multiplier

        resnet_student = load_model('resnet18',pretrained=True)
        resnet_student = FineTuneModel(resnet_student,'resnet34',512)
        resnet_student = torch.nn.DataParallel(resnet_student)
        checkpoint = torch.load(checkpoint)
        print('Model loaded from epoch {} checkpoint'.format(checkpoint['epoch']))
        resnet_student.load_state_dict(checkpoint['state_dict'])
        resnet_student = resnet_student.module        
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # self.reverse_mean = [-0.485, -0.456, -0.406]
        # self.reverse_std = [1/0.229, 1/0.224, 1/0.225]

        self.firstconv = resnet_student.features[0]
        self.firstbn = resnet_student.features[1]
        self.firstrelu = resnet_student.features[2]
        self.firstmaxpool = resnet_student.features[3]
        
        self.encoder1 = resnet_student.features[4]
        self.encoder2 = resnet_student.features[5]
        self.encoder3 = resnet_student.features[6]

        self.final_avg = nn.AvgPool2d(7, stride=1)
        self.classifier = resnet_student.classifier[0]
        
        # Decoder
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)                                     
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        # self.finalconv4 = nn.Conv2d(num_classes*2, num_classes, 3, padding=1)
    
    def forward(self,
               mode,
               source,
               target,
               target_repeat
               ):
        
        if mode=='attack':
            return self.forward_attack(source,target)
        elif mode=='eval':
            return self.forward_eval(source,target,target_repeat)
        elif mode=='test':
            return self.forward_submit(source,target)
        elif mode=='attack_model_loss':
            return self.attack_model_loss(source,target)
        elif mode=='eval_model_loss':
            return self.eval_model_loss(source,target_repeat)
        
    def freeze_encoder(self):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3]
        
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False

    def forward_noise(self, 
                      source,
                      target):
        
        # just a combination of images
        
        # Encoder
        e1_s,e2_s,e3_s = self.forward_encoder(source)
        e1_t,e2_t,e3_t = self.forward_encoder(target)

        e3_t = self.encoder3(e2_t)   
        
        e1 = torch.cat([e1_s,e1_t],dim=1)
        e2 = torch.cat([e2_s,e2_t],dim=1)
        e3 = torch.cat([e3_s,e3_t],dim=1)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5
    
    def forward_encoder(self,
                        x):
        
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        return e1,e2,e3

    def forward_noise5(self, 
                      source,
                      target):
        
        # using all 5 images to calculate average
        # assuming that inputs are
        # source BxCxWxH
        # target Bx5xCxWxH, i.e. all 5 images
        
        assert len(source.size()) == 4
        assert len(target.size()) == 5
        assert target.size(1) == 5
        
        # Encoder
        e1_s,e2_s,e3_s = self.forward_encoder(source)
        
        # print(e1_s.shape)
        # print(e2_s.shape)
        # print(e3_s.shape)        
        
        e1_ts = []
        e2_ts = []
        e3_ts = []
        
        for i in range(0,5):
            e1_t,e2_t,e3_t = self.forward_encoder(target[:,i,:,:,:])
            e1_ts.append(e1_t)
            e2_ts.append(e2_t)
            e3_ts.append(e3_t)
            
        e1_t = torch.stack(e1_ts,dim=1).mean(dim=1)
        e2_t = torch.stack(e2_ts,dim=1).mean(dim=1)
        e3_t = torch.stack(e3_ts,dim=1).mean(dim=1)
        
        # print(torch.stack(e1_ts,dim=1).shape)
        # print(torch.stack(e2_ts,dim=1).shape)
        # print(torch.stack(e3_ts,dim=1).shape)        
        
        # print(e1_t.shape)
        # print(e2_t.shape)
        # print(e3_t.shape)

        e1 = torch.cat([e1_s,e1_t],dim=1)
        e2 = torch.cat([e2_s,e2_t],dim=1)
        e3 = torch.cat([e3_s,e3_t],dim=1)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        
        out = self.noise_multiplier * f5 + source

        return out    
    
    def forward_descriptor(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        e3 = self.final_avg(e3) 
             
        return self.classifier(e3.view(e3.size(0), -1))    
    
    def attack_model_loss(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==6
        assert target.size(1)== 5
        assert target.size(2)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
                             
        source_mod = self.forward_noise5(source,
                                    target)
        
        return source_mod
    
    def eval_model_loss(self,
                        source,
                        target_repeat):
        
        # input tensors should be Bx25xCxWxH
        assert len(source.size())==5
        
        assert (source.size(1))==25
        
        assert len(target_repeat.size())==6
        assert target_repeat.size(1)==25
        assert target_repeat.size(2)==5
        
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target_repeat = target_repeat.view(-1,
                                           target_repeat.size(2),
                                           target_repeat.size(3),
                                           target_repeat.size(4),
                                           target_repeat.size(5))

        # we use all of the 5 target imgs to calculate noise        
        source_mod = self.forward_noise5(source,
                                         target_repeat)    

        return source_mod
    
    def forward_attack(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==6
        assert target.size(1)== 5
        assert target.size(2)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source + noise * self.noise_multiplier
        
        # print(source.shape)
        # print(noise.shape)
        
        modified_descriptions = self.forward_descriptor(source_mod)
     
        avg_target_descriptions = []
        
        for i in range(0,5):
            avg_target_descriptions.append(self.forward_descriptor(target[:,i,:,:,:])) 
            
        avg_target_descriptions = torch.stack(avg_target_descriptions,dim=1).mean(dim=1)
        
        return source_mod,modified_descriptions,avg_target_descriptions
    
    def forward_eval(self,
                     source,
                     target,
                     target_repeat):
        
        # input tensors should be Bx25xCxWxH
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert (source.size(1))==25
        assert (target.size(1))==25
        
        assert len(target_repeat.size())==6
        assert target_repeat.size(1)==25
        assert target_repeat.size(2)==5
        
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
        target = target.view(-1,target.size(2),target.size(3),target.size(4))
        target_repeat = target_repeat.view(-1,
                                           target_repeat.size(2),
                                           target_repeat.size(3),
                                           target_repeat.size(4),
                                           target_repeat.size(5))

        # we use all of the 5 target imgs to calculate noise        
        noise = self.forward_noise5(source,
                                    target_repeat)    
        
        source_mod = source + noise * self.noise_multiplier
        
        # we use only image pairs to calculate descriptors
        source_descriptions = self.forward_descriptor(source_mod)
        target_descriptions = self.forward_descriptor(target)

        # we use all of the 5 target imgs to calculate noise
        
        return source_mod,source_descriptions,target_descriptions
    
    def forward_submit(self,
                      source,
                      target):
        
        assert len(source.size())==5
        assert len(target.size())==5
        
        assert source.size(1)== 1
        assert target.size(1)== 5
        
        # stack all source images into one column
        source = source.view(-1,source.size(2),source.size(3),source.size(4))
                             
        noise = self.forward_noise5(source,
                                    target)
        
        source_mod = source + noise * self.noise_multiplier

        # assuming x and y are Batch x 3 x H x W and mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
        source_mod_renorm = source_mod.clone()
        source_mod_renorm[:, 0, :, :] = source_mod[:, 0, :, :] * self.std[0] + self.mean[0]
        source_mod_renorm[:, 1, :, :] = source_mod[:, 1, :, :] * self.std[1] + self.mean[1]
        source_mod_renorm[:, 2, :, :] = source_mod[:, 2, :, :] * self.std[2] + self.mean[2]
        
        source_mod_renorm[source_mod_renorm > 1] = 1
        source_mod_renorm[source_mod_renorm < 0] = 0   
        
        source_mod_renorm = source_mod_renorm.squeeze(0)

        return source_mod_renorm    