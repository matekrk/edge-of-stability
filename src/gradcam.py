import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utilities import iterate_dataset

def get_target_layers(arch_id, network):
    if arch_id == "lenet2":
        target_layers =  [network.conv2]
    elif arch_id == "resnet9":
        target_layers = [network.conv4]
    elif arch_id == "resnet32":
        target_layers = [network.layer3[4].conv2]
    return target_layers

def do_gradcam(model, dataset, batch_size=64, targets = None, standardized=True, target_layers=None):
    if target_layers is None:
        target_layers = [model.gradcam]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(targets)] if targets is not None else None
    some_visualizations = []

    for (X, y) in iterate_dataset(dataset, batch_size):
        grayscale_cam = cam(input_tensor=X, targets=targets)
        print(grayscale_cam.shape, grayscale_cam)
        model_outputs = cam.outputs
        for img in X:
            img = img.permute((1,2,0)).cpu().numpy()
            if standardized: #FIXME
                img = np.clip((img+1)/2, a_max=1, a_min=0)
            v = model_outputs[0]
            some_visualizations.append((img, show_cam_on_image(img, grayscale_cam[0], use_rgb=True, image_weight=0.6), v.detach().cpu()))
            break # limit to one img per batch! FIXME

    return some_visualizations, model_outputs

def do_gradcam_particular(model, img, standardize):
    #TODO
    pass
