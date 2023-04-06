#model.py
import io
import torch
from torch import nn
from torchvision.transforms import functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from . import general

import torchvision.transforms as T

import matplotlib.pyplot as plt

from PIL import ImageFont, ImageDraw, Image

# COCO classes
CLASSES = [
    'N/A', 'polyp'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    # load model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
    checkpoint = torch.load('/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/output/multi_rgb/checkpoint0099.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model

def get_activation_map(model, frame_num):
    
    img = '/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/cocoapi/multi_rgb/test2017/1.tif'
    im = Image.open(img)
    original_h, original_w  = im.size[0], im.size[1]

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    print('outputs1', outputs)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    print('im.size', im.size)
    print('outputs', outputs['pred_boxes'])
    bboxes_scaled = general.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

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

    print('len: ', len(bboxes_scaled))

    #fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=1, figsize=(7, 7))
    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=1, figsize=(700/100, 700/100), squeeze=False)
    axs = axs.flatten()
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i
        ax.imshow(dec_attn_weights[0, idx].view(h, w).detach().numpy())
        ax.axis('off')
        ax.set_title(None)

    for ax in axs[len(bboxes_scaled):]:
        ax.axis('off')

    fig.tight_layout()
    buffer = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buffer)
    image_data = buffer.getvalue()

    return image_data


def draw_frame(im, probas, bboxes):
    draw = ImageDraw.Draw(im)
    colors = COLORS * 100
    if probas is not None and bboxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(probas, bboxes.tolist(), colors):
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='red', width=2)
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            font = ImageFont.truetype(r'/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', 20)
            draw.text((xmin + 100, ymin + 100), text=text, font=font, fill='red')
    return im