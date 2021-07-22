from tensorflow.keras.layers import Input, Add, Dense, Activation, \
    BatchNormalization, Conv2D, MaxPooling2D, Flatten, DepthwiseConv2D, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


"""
Inpirado em:
https://arxiv.org/abs/1801.04381.pdf
https://arxiv.org/pdf/1409.4842.pdf
https://arxiv.org/pdf/1707.07012.pdf
"""


def mobile_structure(X, filtros, s, camada, bloco):
    nome_conv = 'mob_conv' + str(camada) + bloco
    nome_bn = 'mob_bn' + str(camada) + bloco
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


def inception_module(X, filtros, f, camada, bloco):
    nome_conv = 'inc_conv' + str(camada) + bloco
    nome_bn = 'inc_bn' + str(camada) + bloco
    filtro1, filtro2 = filtros
    # 1x1 Conv
    X = Conv2D(filtro1, (1, 1), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1a')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1a')(X)
    X = Activation('relu')(X)
    if f > 1:
        # FxF Conv
        X = Conv2D(filtro2, (f, f), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1b')(X)
        X = BatchNormalization(axis=3, name=nome_bn + '_1b')(X)
        X = Activation('relu')(X)

    return X


def inception_module_pool(X, filtro, camada, bloco):
    nome_conv = 'inc_conv' + str(camada) + bloco
    nome_bn = 'inc_bn' + str(camada) + bloco
    X = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
    X = Conv2D(filtro, (1, 1), (1, 1), 'same', kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_MP_Conv1x1')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_MP_Conv1x1')(X)
    X = Activation('relu')(X)

    return X


def bloco_de_identidade(X, f, filtros, camada, bloco):
    nome_conv = 'id_conv' + str(camada) + bloco
    nome_bn = 'id_bn' + str(camada) + bloco
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


def nas_reduction_cell(X_ant, X_pos, filtros, f_identidade, camada, bloco):
    nome_conv = 'nas_conv' + str(camada) + bloco
    nome_bn = 'nas_bn' + str(camada) + bloco

    filtros_1 = filtros[0]
    filtros_2 = filtros[1]

    f_1, f_2, f_3, f_4 = filtros_1
    # P1
    X_1 = Conv2D(f_1, (7, 7), (1, 1), 'valid',
                 kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1a')(X_ant)
    X_1 = BatchNormalization(axis=3, name=nome_bn + '_1a')(X_1)
    X_2 = Conv2D(f_2, (5, 5), (1, 1), 'valid',
                 kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1b')(X_pos)
    X_2 = BatchNormalization(axis=3, name=nome_bn + '_1b')(X_2)
    X_add1 = Add()([X_1, X_2])

    X_3 = MaxPooling2D((3, 3))(X_ant)
    X_add2 = Add()([X_1, X_3])

    X_4 = AveragePooling2D((3, 3))(X_pos)
    X_5 = Conv2D(f_3, (5, 5), (1, 1), 'valid',
                 kernel_initializer=glorot_uniform(seed=0), name=nome_conv + '_1c')(X_ant)
    X_5 = BatchNormalization(axis=3, name=nome_bn + '_1c')(X_5)
    X_add3 = Add()([X_4, X_5])

    # P2
    X_1_2 = Conv2D(f_4, (3, 3), (1, 1), 'valid', kernel_initializer=glorot_uniform(seed=0),
                   name=nome_conv + '_2a')(X_add1)
    X_add1_2 = Add()([X_3, X_1_2])

    X_2_2 = AveragePooling2D((3, 3))(X_add1)
    X_3_2 = bloco_de_identidade(X_add2, f_identidade, filtros_2, 2, 'Add2')
    X_add2_2 = Add()([X_2_2, X_3_2])

    # P3
    X = Concatenate(axis=3)([X_add1_2, X_add2_2, X_add3])

    return X


def NN(input_shape=(224, 224, 3), classes=25):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3), (2, 2), 'valid', name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    # Inception Block
    X_1 = inception_module(X, [64, 64], 1, 2, 'A')
    X_2 = inception_module(X, [96, 128], 5, 2, 'B')
    X_3 = inception_module(X, [16, 32], 3, 2, 'C')
    X_4 = inception_module_pool(X, 32, 2, 'D')
    X_ant = Concatenate(axis=3)([X_4, X_3, X_2, X_1])
    # MobileNetV2
    X = mobile_structure(X_ant, 512, 1, 1, 'A')
    X = mobile_structure(X, 512, 2, 2, 'B')
    X = mobile_structure(X, 512, 2, 3, 'C')
    X = mobile_structure(X, 1024, 2, 4, 'D')
    X = mobile_structure(X, 1024, 1, 5, 'E')
    X = mobile_structure(X, 1024, 2, 6, 'F')
    X_pos = mobile_structure(X, 2048, 1, 7, 'G')
    # NasNet - Reduction
    X = nas_reduction_cell(X_ant, X_pos, [[2048, 2048, 4096, 4096], [2048, 2048, 4096]], 2, 8, 'H')

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='out_softmax' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    modelo = Model(input_shape=input_shape, output=X, name='IMN_Net')

    return modelo


modelo = NN()
modelo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar = modelo.fit(X_treino, Y_treino, N de treinos, Tamanho dos lotes)
modelo.fit()

# Testar o modelo = modelo.evaluate(X_teste, Y_teste)
previsao = modelo.evaluate()
print('Loss = ' + str(previsao[0]))
print('Teste Precisão = ' + str(previsao[1]))