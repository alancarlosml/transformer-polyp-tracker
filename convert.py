import cv2
import os
import ffmpeg

# Defina o caminho para a pasta com as imagens
img_folder = '/home/viplabgpu/Downloads/Test/Images/101'

# Defina o nome do arquivo de vídeo
video_name = 'polyp.mp4'

# Obtenha a lista de arquivos na pasta
files = os.listdir(img_folder)

# Ordenar os arquivos por nome
files.sort()

# Obtenha as informações da primeira imagem
frame = cv2.imread(os.path.join(img_folder, files[0]))
height, width, channels = frame.shape

# Crie um objeto de gravação de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 25, (width, height))

# Escreva as imagens no arquivo de vídeo
for filename in files:
    image = cv2.imread(os.path.join(img_folder, filename))
    video.write(image)

# Libere os recursos
video.release()

# Converter o arquivo para mp4 usando ffmpeg
input_video = ffmpeg.input(video_name)
ffmpeg.output(input_video, 'final_' + video_name, vcodec='libx264', acodec='aac').run()
