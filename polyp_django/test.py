import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from skimage import io, transform

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

def load_model():
    # load model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
    checkpoint = torch.load('/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/output/multi_rgb/checkpoint0099.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model

model = load_model()

url = '/home/viplabgpu/Documentos/alan/DETR/fabebook_DETR/detr/cocoapi/multi_rgb/test2017/1.tif'
im = Image.open(url)

img = transform(im).unsqueeze(0)

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

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9
# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0]
dec_attn_weights = dec_attn_weights[0]

# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]

fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))

# downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
fact = 32

# let's select 4 reference points for visualization
idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]

# here we create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# and we add one plot per reference point
gs = fig.add_gridspec(2, 4)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[0, -1]),
    fig.add_subplot(gs[1, -1]),
]
f_map = conv_features['0']
shape = f_map.tensors.shape[-2:]
# and reshape the self-attention to a more interpretable shape
sattn = enc_attn_weights[0].reshape(shape + shape)

# for each one of the reference points, let's plot the self-attention
# for that point
for idx_o, ax in zip(idxs, axs):
    idx = (idx_o[0] // fact, idx_o[1] // fact)
    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(im)
for (y, x) in idxs:
    scale = im.height / img.shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')

fig.savefig('final_image.png', dpi=300)

# Cria uma nova imagem com o tamanho da imagem plotada
width, height = fig.get_size_inches() * fig.get_dpi()
new_image = Image.new('RGB', (int(width), int(height)))

# Desenha a imagem plotada na nova imagem
fig.canvas.draw()
new_image.paste(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()), (0, 0))

# Salva a nova imagem
new_image.save('final_image.png')

'''
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(None)
fig.tight_layout()
fig.savefig('mapas_de_ativacao.png', dpi=150)
'''

'''
activation_maps = []
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(4, 4))
axs = axs.flatten()

for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i
    ax.imshow(dec_attn_weights[0, idx].detach().numpy().reshape(h, w))
    ax.axis('off')
    ax.set_title(None)
    activation_maps.append(dec_attn_weights[0, idx].detach().numpy())
    plt.imsave(f'mapa_ativacao_{idx.item()}.png', dec_attn_weights[0, idx].detach().numpy().reshape(h, w), dpi=1000)

for ax in axs[len(bboxes_scaled):]:
    ax.axis('off')

fig.tight_layout()
fig.savefig('mapas_de_ativacao.png', dpi=150)
'''


'''
activation_maps = []

fig, axs = plt.subplots(ncols=min(len(bboxes_scaled), 4), nrows=2, figsize=(16, 8))
axs = axs.flatten()

for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs, bboxes_scaled):
    ax = ax_i
    ax.imshow(dec_attn_weights[0, idx].detach().numpy().reshape(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    activation_maps.append(dec_attn_weights[0, idx].detach().numpy())
    if len(activation_maps) == 8:
        break

#fig.tight_layout()
fig.savefig('mapas_de_ativacao.png', dpi=150)
'''

'''
fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(300, 300))
axs = axs.flatten()


for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i
    ax.imshow(dec_attn_weights[0, idx].detach().numpy().reshape(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    activation_maps.append(dec_attn_weights[0, idx].detach().numpy())
    #plt.imsave(f'mapa_ativacao_{idx.item()}.png', dec_attn_weights[0, idx].detach().numpy().reshape(h, w), dpi=1000)
    
print(len(activation_maps))
fig.tight_layout()
fig.savefig('mapas_de_ativacao.png')

buffer = BytesIO()
fig.canvas.draw()
pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
pil_image.save('mapa_de_ativacao.png')
'''

# Retorna a lista de mapas de ativação
#activation_maps