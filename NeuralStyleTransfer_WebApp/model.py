import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self ):
        super(Model , self ).__init__()

        self.vgg = models.vgg19(pretrained = True).features[:29]
        self.needed_layers = [0 , 5 , 10 , 19 , 28]
    

    def forward(self , x):
        features = []

        for layer_idx , layer in enumerate(self.vgg):
            x = layer(x)

            if layer_idx in self.needed_layers:
                features.append(x)
        
        return features


