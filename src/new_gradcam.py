import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from new_data_utils import iterate_dataset

def do_gradcam(model, dataset, batch_size=64, counter=10, targets = None, rescale=True, target_layers=None):
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(targets)] if targets is not None else None
    some_visualizations = []

    for (X, y) in iterate_dataset(dataset, batch_size):
        grayscale_cam = cam(input_tensor=X, targets=targets)
        model_outputs = cam.outputs
        for i, img in enumerate(X):
            img = img.permute((1,2,0)).cpu().numpy()
            if rescale:
                img = np.clip((img+1)/2, a_max=1, a_min=0)
            v = model_outputs[i]
            some_visualizations.append((img, show_cam_on_image(img, grayscale_cam[i], use_rgb=True, image_weight=0.6), v.detach().cpu()))
            if len(some_visualizations) >= counter:
                return some_visualizations, model_outputs
