import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np

def load_model():
    # load model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
    checkpoint = torch.load('/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/output/multi_rgb/checkpoint0099.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model

def get_gradcam(model, img_path, target_class):
    # Load the image
    img = cv2.imread(img_path)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    # Get the feature map of the last convolutional layer and the final output from the model
    features, output = model.backbone(img_tensor)
    output = model.class_embed(output)

    # Compute the gradient of the target class output with respect to the feature map
    one_hot = F.one_hot(torch.tensor([target_class]), num_classes=output.shape[-1])
    one_hot = torch.sum(one_hot.float() * output)

    model.zero_grad()
    features.zero_grad()
    one_hot.backward(retain_graph=True)

    # Extract the gradients and compute the weights using global average pooling
    gradients = model.backbone.grad.cpu().squeeze(0)
    weights = F.adaptive_avg_pool2d(gradients, 1)

    # Compute the GradCAM heatmap
    cam = GradCAM(model=model.backbone, target_layer_names=["7"], use_cuda=torch.cuda.is_available())
    heatmap = cam(input_tensor=img_tensor, target_category=target_class, gradients=weights)[0]

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Normalize the heatmap and show it on the original image
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    cam_image = show_cam_on_image(img, heatmap)

    return cam_image



model = load_model()
cam = get_gradcam(model, '/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/cocoapi/multi_rgb/test2017/1.tif', 1)