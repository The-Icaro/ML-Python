from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


def bloco_tipo1(X, filtros:int, f, camada, max=True):
    X = Conv2D(filtros, (f, f), strides=(2, 2),
               kernel_initializer=glorot_uniform(seed=0), name='conv' + str(camada))(X)
    X = BatchNormalization(axis=3, name='bn' + str(camada))(X)
    if max:
        X = MaxPooling2D((2, 2), strides=(2, 2), name='max' + str(camada))(X)

    return X


def bloco_tipo2(X, filtros: list, rep=False, int_rep=4):
    f_1, f_2, f_3, f_4 = filtros
    if rep:
        if int_rep == 4:
            for _ in range(int_rep):
                X = Conv2D(f_1, (1, 1), strides=(1, 1),
                           kernel_initializer=glorot_uniform(seed=0), name='conv4_A')(X)
                X = Conv2D(f_2, (3, 3), strides=(1, 1),
                           kernel_initializer=glorot_uniform(seed=0), name='conv4_B')(X)
            X = Conv2D(f_3, (1, 1), strides=(1, 1),
                       kernel_initializer=glorot_uniform(seed=0), name='conv4_C')(X)
            X = Conv2D(f_4, (3, 3), strides=(1, 1),
                       kernel_initializer=glorot_uniform(seed=0), name='conv4_D')(X)
        if int_rep == 2:
            for _ in range(int_rep):
                X = Conv2D(f_1, (1, 1), strides=(1, 1),
                           kernel_initializer=glorot_uniform(seed=0), name='conv5_A')(X)
                X = Conv2D(f_2, (3, 3), strides=(1, 1),
                           kernel_initializer=glorot_uniform(seed=0), name='conv5_B')(X)
            X = Conv2D(f_3, (3, 3), strides=(1, 1),
                       kernel_initializer=glorot_uniform(seed=0), name='conv5_C')(X)
            X = Conv2D(f_4, (3, 3), strides=(1, 1),
                       kernel_initializer=glorot_uniform(seed=0), name='conv5_D')(X)
            return X
    else:
        X = Conv2D(f_1, (1, 1), strides=(1, 1),
                   kernel_initializer=glorot_uniform(seed=0), name='conv3_A')(X)
        X = Conv2D(f_2, (3, 3), strides=(1, 1),
                   kernel_initializer=glorot_uniform(seed=0), name='conv3_B')(X)
        X = Conv2D(f_3, (1, 1), strides=(1, 1),
                   kernel_initializer=glorot_uniform(seed=0), name='conv3_C')(X)
        X = Conv2D(f_4, (3, 3), strides=(1, 1),
                   kernel_initializer=glorot_uniform(seed=0), name='conv3_D')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    return X


def DarkNet(input_shape=(448, 448, 3)):
    X_input = Input(input_shape)
    X = Conv2D(64, (7, 7), strides=(2, 2),  name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max1')(X)

    X = bloco_tipo1(X, 192, 3, 2, max=True)
    X = bloco_tipo2(X, [128, 256, 256, 512], rep=False)
    X = bloco_tipo2(X, [256, 512, 512, 1024], rep=True, int_rep=4)
    X = bloco_tipo2(X, [512, 1024, 1024, 1024], rep=True, int_rep=2)
    for _ in range(2):
        X = bloco_tipo1(X, 1024, 3, 6, max=False)
    X = Flatten()(X)
