import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layer):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def forward(self, x):
        for name, module in self.model.named_children():
            if name == 'transformer':
                for transformer_layer in module.encoder.layers:
                    x, _ = transformer_layer(x)
            else:
                x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                features = x
        return features

def get_gradcam(image_path, model, target_layer):
    # Load the image
    image = Image.open(image_path)

    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transforms to the image
    image_tensor = transform(image).unsqueeze(0)

    # Extract the features from the target layer
    feature_extractor = FeatureExtractor(model, target_layer)
    features = feature_extractor(image_tensor)

    # Forward pass
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()

    # Backward pass
    model.zero_grad()
    one_hot_output = torch.zeros((1, output.size()[-1]), device=output.device)
    one_hot_output[0][prediction] = 1
    output.backward(gradient=one_hot_output)

    # Calculate the weights
    gradients = feature_extractor.gradients[0]
    weights = torch.mean(gradients, axis=(1, 2))
    weights = weights.view(-1, 1, 1)
    activation = features

    # Multiply the weights with the activation maps
    cam = torch.sum(weights * activation, axis=0)
    
    # Apply the ReLU function to obtain the final map
    cam = nn.functional.relu(cam)

    # Normalize the map
    cam = cam / torch.max(cam)

    return cam

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True, num_classes=2)
model.eval()

img_path = '/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/cocoapi/multi_rgb/test2017/1.tif'
result = get_gradcam(img_path, model, 'backbone.conv1')
print(result)