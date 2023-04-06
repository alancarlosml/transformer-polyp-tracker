import os
import html
import time
import json
import base64
import datetime
import numpy as np

import torch
import torchvision.transforms as T

import cv2
from PIL import Image
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from . import general
from . import models
from . import model

torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

detr = None
def get_detr_model():
    global detr
    if detr is None:
        detr = model.load_model()
    return detr

def index(request):
    return render(request, 'polyp_django/index.html')

@csrf_exempt
def video_feed(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        # Salvar o arquivo de vídeo no disco
        with open('media/video.mp4', 'wb') as f:
            f.write(video_file.read())
        # Definir o número inicial do frame
        request.session['frame_num'] = 0
        
        # Criar um novo objeto VideoProcessing
        video_processing = models.VideoProcessing(start_time=datetime.datetime.now())
        video_processing.file_type = 1 #video
        video_processing.save()
        request.session['id'] = video_processing.id
        
        return HttpResponse('')
    else:
        # Obter o número atual do frame
        frame_num = request.session.get('frame_num', 0)

        # Obter o objeto VideoProcessing correspondente ao ID da sessão atual
        id_obj = request.session.get('id')
        video_processing = models.VideoProcessing.objects.get(id=id_obj)

        # Abrir o arquivo de vídeo
        #cap = get_video()
        cap = cv2.VideoCapture('media/video.mp4')

        # Obter o total de frames do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Verificar se a última imagem foi processada
        if frame_num >= total_frames:
            now = datetime.datetime.now()
            filename = now.strftime("%Y-%m-%d_%H-%M-%S.mp4")
            output_video_path = os.path.join('media', filename)
            input_images_path = os.path.join('media', 'output')
            # Criar o vídeo completo
            clip = ImageSequenceClip(input_images_path, fps=20)
            clip.write_videofile(output_video_path)
            video_processing.filename = filename
            video_processing.end_time = now
            video_processing.status = 'success'
            video_processing.save()
            #remover arquivos pasta temp media/output
            for filename in os.listdir(input_images_path):
                file_path = os.path.join(input_images_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print('Falha ao excluir %s. Razão: %s' % (file_path, e))
            
            return HttpResponse('')
        
        # Obter o frame atual
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            video_processing.status = 'error'
            video_processing.save()
            # Retornar uma resposta vazia se não houver mais frames
            return HttpResponse('')

        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # apply transforms to prepare input image
        input_tensor = transform(im).unsqueeze(0)

        # load model
        detr = get_detr_model()

        # get model outputs
        start_time = time.time()
        outputs = detr(input_tensor)
        end_time = time.time()
        frame_time = (end_time - start_time) * 1000  # tempo de processamento em milissegundos

        # filter boxes based on confidence threshold
        probas, bboxes = general.filter_bboxes_from_outputs(im, outputs, threshold=0.5)

        image_original = os.path.join('media', 'output', f'original_{frame_num:d}.png')
        im.save(image_original)
        
        activation_map = model.get_activation_map(detr, frame_num)

        im = model.draw_frame(im, probas, bboxes)

        # Salvar a imagem processada em disco
        image_path = os.path.join('media', 'output', f'{frame_num:06d}.png')
        im.save(image_path)

        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', im)
        #_, buffer_map = cv2.imencode('.jpg', activation_map)
        img_str = base64.b64encode(buffer).decode()
        activation_map = base64.b64encode(activation_map).decode()

        data = {'img_str': img_str, 'activation_map': activation_map, 'frame_time': int(frame_time), 'frame_num': frame_num}
        data = json.dumps(data)
        data = html.unescape(data)
        video_processing.status = 'processing'
        video_processing.save()
        # Salvar o número do próximo frame
        request.session['frame_num'] = frame_num + 1

        # Retornar a imagem codificada em base64
        return HttpResponse(data, content_type='application/json')
