from PIL import Image
import cv2
import io
import numpy as np

from PIL import Image
import requests
import matplotlib.pyplot as plt


import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

# COCO classes
CLASSES = [
    'N/A', 'polyp']
	  
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
checkpoint = torch.load('/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/output/multi_rgb/checkpoint0099.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

img = '/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/cocoapi/multi_rgb/test2017/1.tif'
im = Image.open(img)
original_h, original_w  = im.size[0], im.size[1]

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []

hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]

# propagate through the model
outputs = model(img)

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0]
dec_attn_weights = dec_attn_weights[0]	

# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]

#fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=1, figsize=(7, 7))
fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=1, figsize=(original_h/100, original_w/100), squeeze=False)
axs = axs.flatten()
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(None)

for ax in axs[len(bboxes_scaled):]:
    ax.axis('off')

fig.tight_layout()
fig.savefig('teste.png')

#fig.canvas.draw()
#img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
#img.save('teste.png', format='PNG')



# ... faça o que precisar com os objetos de imagem ...

# Feche a figura
#plt.close(fig)