from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleTransfer(torch.nn.Module):
    """
        # StyleTransfer

        paper: https://arxiv.org/pdf/1508.06576.pdf
    """
    def __init__(self, model: torch.nn.Sequential, 
        style_layers=['conv_1','conv_2','conv_3','conv_4','conv_5'], 
        content_layers=['conv_4'], 
        loss_factor={"content":1,"style":1000000}
    ):
        super().__init__()
        self.model = self.layer_naming(model)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.loss_factor = loss_factor

    def layer_naming(self, model):
        layers = []
        i=1
        for n in model.children():
            if isinstance(n, nn.Conv2d):
                layers.append((f"conv_{i}", n))
            elif isinstance(n,nn.ReLU):
                layers.append((f"relu_{i}",torch.nn.ReLU()))
                i+=1
            else:
                layers.append((f"_i", n))
        return nn.Sequential(OrderedDict(layers))

    def get_model(model, style_target, content_target, **kwargs):
        obj = StyleTransfer(model, **kwargs)
        obj.style_target = style_target.detach()
        obj.content_target = content_target.detach()
        return obj

    def gram_metrix(self, x):
        a,b,c,d = x.shape
        x = x.view(a*b, c*d)
        return torch.matmul(x,x.T)/(a*b*c*d)

    def loss(self, x):
        content_loss = []
        style_loss = []
        y_c = self.content_target.clone()
        y_s = self.style_target.clone()
        for name, layer in self.model.named_children():
            x = layer(x)
            y_c = layer(y_c)
            y_s = layer(y_s)
            if name in self.content_layers:
                content_loss.append(F.mse_loss(x,y_c))
            if name in self.style_layers:
                style_loss.append(F.mse_loss(self.gram_metrix(x), self.gram_metrix(y_s)))
            if len(content_loss) == self.content_layers and len(style_loss) == self.style_layers:
                break

        return self.loss_factor['content']*torch.sum(torch.stack(content_loss)) + self.loss_factor['style']*torch.sum(torch.stack(style_loss))

    def forward(self, x):
        return x