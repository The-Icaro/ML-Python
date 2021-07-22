from tensorflow.keras.layers import Input, Add, Dense, Activation, \
    BatchNormalization, Conv2D, AveragePooling2D, Flatten, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

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


def MobileNet(input_shape=(224, 224, 3), classes=6):
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
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='out_softmax' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    modelo = Model(input_shape=input_shape, output=X, name='MobileNet')

    return modelo


modelo = MobileNet()
modelo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar = modelo.fit(X_treino, Y_treino, N de treinos, Tamanho dos lotes)
modelo.fit()

# Testar o modelo = modelo.evaluate(X_teste, Y_teste)
previsao = modelo.evaluate()
print('Loss = ' + str(previsao[0]))
print('Teste Precisão = ' + str(previsao[1]))

