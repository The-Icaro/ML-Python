from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
    BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

"""
Infos:
X_treino, X_teste .shape = (N de treinos/teste, Altura(n_H), Largura(n_W), Camadas(n_C))
                 exemplo = Treino - (1080, 64, 64, 3); Teste - (620, 64, 64, 3)
Y_treino, Y_teste .shape = (N de treinos/Teste, Camadas(n_Y))
                 exemplo = Treino - (1080, 6); Teste - (620, 6)
Estrutura do Modelo: 6 Camadas Principais, 50 Layers
https://www.researchgate.net/publication/341581939/figure/fig1/AS:894155496648705@1590194691064/Overview-of-original-ResNet50-architecture-42-At-stage-1-the-feature-map-size-is.ppm
1 Camada = Pad -> Conv -> Normalização -> AT Relu -> MaxPool (1)
2 Camada = Bloco Conv (4) -> Bloco Id (7) -> Bloco Id (10)
3 Camada = Bloco Conv (13) -> Bloco Id (16) -> Bloco Id (19) -> Bloco Id (22)
4 Camada = Bloco Conv (25) -> Bloco Id (28) -> Bloco Id (31) -> Bloco Id (34) -> Bloco Id (37) -> Bloco Id (40)
5 Camada = Bloco Conv (43) -> Bloco Id (46) -> Bloco Id (49)
6 Camada = AvgPool -> AT Softmax -> Y (50)
"""


def bloco_de_identidade(X, f, filtros, camada, bloco):
    nome_conv = 'res' + str(camada) + bloco
    nome_bn = 'bn' + str(camada) + bloco
    f_1, f_2, f_3 = filtros
    # Corta Caminho do X
    X_shortcut = X
    # Camada 1
    X = Conv2D(f_1, (1, 1), (1, 1), 'valid', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1a')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1a')(X)
    X = Activation('relu')(X)
    # Camada 2
    X = Conv2D(f_2, (f, f), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1b')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1b')(X)
    X = Activation('relu')(X)
    # Camada 3
    X = Conv2D(f_3, (1, 1), (1, 1), 'valid', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1c')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def bloco_convolucional(X, f, filtros, camada, bloco, s=2):
    nome_conv = 'res' + str(camada) + bloco
    nome_bn = 'bn' + str(camada) + bloco
    f_1, f_2, f_3 = filtros
    # Corta Caminho do X
    X_shortcut = X
    # Camada 1
    X = Conv2D(f_1, (1, 1), (s, s), 'valid', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1a')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1a')(X)
    X = Activation('relu')(X)
    # Camada 2
    X = Conv2D(f_2, (f, f), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1b')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1b')(X)
    X = Activation('relu')(X)
    # Camada 3
    X = Conv2D(f_3, (1, 1), (1, 1), 'valid', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1c')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1c')(X)

    X_shortcut = Conv2D(f_3, kernel_size=(1, 1), strides=(s, s),
                        padding='valid', name=nome_conv + 'shortcut')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=nome_bn + 'shortcut')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def Resnet_50(input_shape=(224, 224, 3), classes=6):
    X_input = Input(input_shape)
    # Camada 1
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X - MaxPooling2D((3, 3), strides=(2, 2))(X)
    # Camada 2
    X = bloco_convolucional(X, 3, [64, 64, 256], 2, 'A', 2)
    X = bloco_de_identidade(X, 3, [64, 64, 256], 2, 'B')
    X = bloco_de_identidade(X, 3, [64, 64, 256], 2, 'C')
    # Camada 3
    X = bloco_convolucional(X, 3, [128, 128, 512], 3, 'A', 2)
    X = bloco_de_identidade(X, 3, [128, 128, 512], 3, 'B')
    X = bloco_de_identidade(X, 3, [128, 128, 512], 3, 'C')
    X = bloco_de_identidade(X, 3, [128, 128, 512], 3, 'D')
    # Camada 4
    X = bloco_convolucional(X, 3, [512, 512, 1024], 4, 'A', 2)
    X = bloco_de_identidade(X, 3, [512, 512, 1024], 4, 'B')
    X = bloco_de_identidade(X, 3, [512, 512, 1024], 4, 'C')
    X = bloco_de_identidade(X, 3, [512, 512, 1024], 4, 'D')
    X = bloco_de_identidade(X, 3, [512, 512, 1024], 4, 'E')
    X = bloco_de_identidade(X, 3, [512, 512, 1024], 4, 'F')
    # Camada 5
    X = bloco_convolucional(X, 3, [512, 512, 2048], 5, 'A', 2)
    X = bloco_de_identidade(X, 3, [512, 512, 2048], 5, 'B')
    X = bloco_de_identidade(X, 3, [512, 512, 2048], 5, 'C')
    # Camada 6
    X = AveragePooling2D(pool_size=(2, 2), name='pool_media')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax',
              name='out_softmax' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    modelo = Model(input_shape=input_shape, output=X, name='ResNet50')
    return modelo


modelo = Resnet_50()
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar = modelo.fit(X_treino, Y_treino, N de treinos, Tamanho dos lotes)
modelo.fit()

# Testar o modelo = modelo.evaluate(X_teste, Y_teste)
previsao = modelo.evaluate()
print('Loss = ' + str(previsao[0]))
print('Teste Precisão = ' + str(previsao[1]))
