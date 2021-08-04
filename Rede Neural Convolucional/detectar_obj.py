from IPython.display import Image, display
import os
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


imagens = ["apple.jpg",
           "apples.jpg",
           "car.jpg",
           "car1.jpg",
           "car2.jpg",
           "car3.jpg",
           "clock.jpg",
           "clock2.jpg",
           "clock3.jpg",
           "fruits.jpg",
           "oranges.jpg"
           ]

for i in imagens:
    print(f"\nDisplay Imagem: {i}")
    display(Image(filename=f"imagens/{i}"))


def detectar_prever(arq_nome, modelo="yolov3-tiny", confianca=0.2,
                    diretorio_save=r"C:\Users\icaro\PycharmProjects\Projeto\PYTHONML\Rede Neural Convolucional\img_predict"):

    img_path = f"imagens/{arq_nome}"
    img = cv2.imread(img_path)

    bbox, label, conf = cv.detect_common_objects(img, confidence=confianca, model=modelo)

    print(f'Processando Imagem -> {arq_nome}')

    for l,c in zip(label, conf):
        print(f'Objeto Detectado: {l} com Confiança: {c}\n')

    output = draw_bbox(img, bbox, label, conf)

    diretorio = diretorio_save
    if not os.path.exists(diretorio):
        os.mkdir(diretorio)
    cv2.imwrite(f"{diretorio}/{arq_nome}", output)

    display(Image(f"{diretorio}/{arq_nome}"))


for i in imagens:
    detectar_prever(i)

