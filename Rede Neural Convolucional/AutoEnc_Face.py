from tensorflow.keras.layers import Input, Add, Dense, Activation, \
    BatchNormalization, Conv2D, AveragePooling2D, Flatten, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
import cv2


"""
Inspirado em:
[Sandler et al. 2019, MobileNetV2: Inverted Residuals and Linear Bottlenecks]
https://arxiv.org/abs/1801.04381
"""


def blotteneck_resnet_s1(X, filtros, camada, bloco, s):
    nome_conv = 'res' + str(camada) + bloco
    nome_bn = 'bn' + str(camada) + bloco
    # Corta Caminho do X
    X_shortcut = X
    # Expansão
    X = Conv2D(filtros, (1, 1), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1a')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1a')(X)
    X = Activation('relu')(X)
    # DepthWise
    X = DepthwiseConv2D((3, 3), (s, s), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1b')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1b')(X)
    X = Activation('relu')(X)
    # Projecão
    X = Conv2D(filtros, (1, 1), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1c')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1c')(X)
    # Concluindo Res
    X = Add()([X, X_shortcut])

    return X


def AutoEnc(input_shape=(224, 224, 3), enc_dims=(128, 64, 32), dec_dims=(64, 128, 784)):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3), (2, 2), 'valid', name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = blotteneck_resnet_s1(X, 16, 1, 'A', s=1)
    X = blotteneck_resnet_s1(X, 24, 2, 'B', s=2)
    X = blotteneck_resnet_s1(X, 32, 3, 'C', s=2)
    X = blotteneck_resnet_s1(X, 64, 4, 'D', s=2)
    X = blotteneck_resnet_s1(X, 96, 5, 'E', s=1)
    X = blotteneck_resnet_s1(X, 160, 6, 'F', s=2)
    X = blotteneck_resnet_s1(X, 320, 7, 'G', s=1)
    X = Conv2D(1280, (1, 1), (1, 1), 'same', name='conv2', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn_conv2')(X)
    X = AveragePooling2D((7, 7), strides=(1, 1))(X)

    enc1, enc2, enc3 = enc_dims
    X = Flatten()(X)
    encoded = Dense(enc1, activation='relu', name='fc_relu' + str(enc1),
                    kernel_initializer=glorot_uniform(seed=0))(X)
    encoded = Dense(enc2, activation='relu', name='fc_relu' + str(enc2),
                    kernel_initializer=glorot_uniform(seed=0))(encoded)
    encoded = Dense(enc3, activation='relu', name='fc_relu' + str(enc3),
                    kernel_initializer=glorot_uniform(seed=0))(encoded)

    dec1, dec2, dec3 = dec_dims
    decoded = Dense(dec1, activation='relu', name='fc_relu' + str(dec1),
                    kernel_initializer=glorot_uniform(seed=0))(encoded)
    decoded = Dense(dec2, activation='relu', name='fc_relu' + str(dec2),
                    kernel_initializer=glorot_uniform(seed=0))(decoded)
    decoded = Dense(dec3, activation='sigmoid', name='fc_relu' + str(dec3),
                    kernel_initializer=glorot_uniform(seed=0))(decoded)

    autoencoder = Model(input_shape, decoded, name='AE')

    return autoencoder


def img_to_encoding(image_path, modelo):
    img1 = cv2.imread(image_path, 1)
    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = modelo.predict_on_batch(x_train)
    return embedding


def face_verificar(image_path, nome: str, dados: dict, modelo):
    enconding = img_to_encoding(image_path, modelo)
    verif = np.linalg.norm(enconding - dados[nome])

    if verif < 0.7:
        print("É você mesmo " + str(nome))
    else:
        print("Não é " + str(nome))

    return verif


def face_reconhecimento(image_path, dados: dict, modelo):
    enconding = img_to_encoding(image_path, modelo)
    min_verif = 100
    identidade = None
    for nome, dados_enc in dados.items():
        verif = np.linalg.norm(enconding - dados_enc)
        if verif < min_verif:
            min_verif = verif
            identidade = nome

    if min_verif < 0.7:
        print("É você mesmo " + str(identidade))
    else:
        print("Não esta no Banco de Dados!")


modelo = AutoEnc()
dados = {}
# img_to_enconding para cada img em dados
# E Fazer o Reconhecimento ou Verificação
