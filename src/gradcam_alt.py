# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

import numpy as np
import torch
import cv2

def activations_hook(self, grad):
    self.gradients = grad

def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((1, 1920))
        x = self.classifier(x)
        return x

def get_activations_gradient(self):
        return self.gradients
    
def get_activations(self, x):
    return self.features_conv(x)

pred = vgg(img).argmax(dim=1)
# get the gradient of the output with respect to the parameters of the model
pred[:, 386].backward()
# pull the gradients out of the model
gradients = vgg.get_activations_gradient()
# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
# get the activations of the last convolutional layer
activations = vgg.get_activations(img).detach()
# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()
# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)
# normalize the heatmap
heatmap /= torch.max(heatmap)
# draw the heatmap
plt.matshow(heatmap.squeeze())


img = cv2.imread('./data/Elephant/data/05fig34.jpg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./map.jpg', superimposed_img)