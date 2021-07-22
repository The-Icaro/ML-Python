from tensorflow.keras.layers import Input, Add, Dense, Activation, \
    BatchNormalization, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


def bloco_down(X, filtros, f, camada):
    nome_cn = 'conv_' + str(camada)
    nome_bn = 'bn_' + str(camada)
    f_1, f_2 = filtros

    X = Conv2D(f_1, (f, f), padding='valid', kernel_initializer=glorot_uniform(seed=0), name=nome_cn + '_1')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1')(X)
    X = Activation('relu')(X)

    X = Conv2D(f_2, (f, f), padding='valid', kernel_initializer=glorot_uniform(seed=0), name=nome_cn + '_2')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_2')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X_add = X

    return X, X_add


def bloco_up(X, filtros, f, camada):
    nome_cn = 'conv_' + str(camada)
    nome_bn = 'bn_' + str(camada)
    f_1, f_2 = filtros

    X = Conv2D(f_1, (f, f), padding='valid', kernel_initializer=glorot_uniform(seed=0), name=nome_cn + '_1')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_1')(X)
    X = Activation('relu')(X)

    X = Conv2D(f_2, (f, f), padding='valid', kernel_initializer=glorot_uniform(seed=0), name=nome_cn + '_2')(X)
    X = BatchNormalization(axis=3, name=nome_bn + '_2')(X)
    X = Activation('relu')(X)

    X = Conv2DTranspose(f_2, (2, 2), padding='valid', kernel_initializer=glorot_uniform(seed=0), name=nome_cn + '_T')(X)

    return X


def U_net(input_shape=(256, 256, 3)):
    X_input = Input(input_shape)
    X, X_add_1 = bloco_down(X_input, [32, 32], 3, 1)
    X, X_add_2 = bloco_down(X, [64, 64], 3, 2)
    X, X_add_3 = bloco_down(X, [128, 128], 3, 3)
    X, X_add_4 = bloco_down(X, [256, 256], 3, 4)
    X, X_add_5 = bloco_down(X, [512, 512], 3, 5)
    X = bloco_up(X, [1024, 1024], 3, 1)
    X = Add()([X, X_add_5])
    X = bloco_up(X, [512, 512], 3, 2)
    X = Add()([X, X_add_4])
    X = bloco_up(X, [256, 256], 3, 3)
    X = Add()([X, X_add_3])
    X = bloco_up(X, [128, 128], 3, 4)
    X = Add()([X, X_add_2])
    X = bloco_up(X, [64, 64], 3, 5)
    X = Add()([X, X_add_1])
    X = Conv2D(32, (3, 3), kernel_initializer=glorot_uniform(seed=0), name='conv_final_1')(X)
    X = Conv2D(32, (3, 3), kernel_initializer=glorot_uniform(seed=0), name='conv_final_2')(X)
    X = Conv2D(32, (1, 1), kernel_initializer=glorot_uniform(seed=0), name='conv_final_3')(X)

    modelo = Model(input_shape=input_shape, output=X, name='U_net')

    return modelo



