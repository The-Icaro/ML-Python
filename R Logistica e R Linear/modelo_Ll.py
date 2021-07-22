import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.max(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def l_relu(Z):
    A = np.max(0.01 * Z, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def sig_prop_tras(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu_prop_tras(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def l_relu_prop_tras(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0.01] = 0
    assert dZ.shape == Z.shape
    return dZ


def ini(n_l: list):
    para = {}
    L = len(n_l)
    for i in range(1, L):
        para['W' + str(i)] = np.random.randn(n_l[i], n_l[i - 1]) * 0.01
        para['b' + str(i)] = np.zeros((n_l[i], 1))
    return para


def linear_frente(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_frente_at(A, W, b, at: str):
    global at_cache
    Z, cache_linear = linear_frente(A, W, b)
    if at == 'sigmoid':
        A, at_cache = sigmoid(Z)
    elif at == 'relu':
        A, at_cache = relu(Z)
    elif at == 'lrelu':
        A, at_cache = l_relu(Z)
    cache = (cache_linear, at_cache)
    return A, cache


def prop_frente(X, par: dict):
    caches = []
    A = X
    L = len(par) // 2
    for i in range(1, L):
        A_ant = A
        A, cache = linear_frente_at(A_ant, par['W' + str(i)], par['b' + str(i)], 'relu')
    AL, cache = linear_frente_at(A, par['W'] + str(L), par['b'] + str(L), 'lrelu')
    caches.append(cache)
    return AL, caches


def f_custo(AL, Y):
    m = Y.shape[1]
    custo = (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    custo = np.squeeze(custo)
    return custo


def linear_tras(dZ, cache):
    A_ant, W, b = cache
    m = A_ant.shape[1]
    dW = 1 / m * np.dot(dZ, A_ant.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_tras_at(dA, cache, at: str):
    global dZ
    linear_cache, at_cache = cache
    if at == 'sigmoid':
        dZ = sig_prop_tras(dA, at_cache)
    if at == 'relu':
        dZ = relu_prop_tras(dA, at_cache)
    if at == 'lrelu':
        dZ = l_relu_prop_tras(dA, at_cache)
    dA_ant, dW, db = linear_tras(dZ, linear_cache)
    return dA_ant, dW, db


def prop_tras(AL, caches, Y):
    derivadas = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cache_atual = caches[L - 1]
    dA_ant_t, dW_t, db_t = linear_tras_at(dAL, cache_atual, 'lrelu')
    derivadas['dA' + str(L - 1)] = dA_ant_t
    derivadas['dW' + str(L)] = dW_t
    derivadas['dB' + str(L)] = db_t
    for i in reversed(range(L - 1)):
        cache_atual = caches[i]
        dA_ant_t, dW_t, db_t = linear_tras_at(derivadas['dA' + str(i + 1)], cache_atual, 'relu')
        derivadas['dA' + str(i)] = dA_ant_t
        derivadas['dW' + str(i + 1)] = dW_t
        derivadas['dB' + str(i + 1)] = db_t
    return derivadas


def otimizar(par, derivadas, l_r):
    par = par.copy()
    L = len(par) // 2
    for i in range(L):
        par['W' + str(i + 1)] = par['W' + str(i + 1)] - l_r * derivadas['dW' + str(i + 1)]
        par['b' + str(i + 1)] = par['b' + str(i + 1)] - l_r * derivadas['db' + str(i + 1)]
    return par


def modelo(X, Y, layers_dims: list, l_r = 0.05, n_it = 3000, p_custo = False):
    custos = []
    par = ini(layers_dims)
    for i in range(0, n_it):
        AL, caches = prop_frente(X, par)
        custo = f_custo(AL, Y)
        derivadas = prop_tras(AL, caches, Y)
        par = otimizar(par, derivadas, l_r)
        if p_custo and i % 150 == 0:
            print(f'Custo com {i} iterações = {np.squeeze(custos)}')
        if i % 150 == 0:
            custos.append(custo)
        return par, custos


def prever(X_teste, par, p_previsao = False):
    AL, caches = prop_frente(X_teste, par)
    previsao = AL > 0.5
    if p_previsao:
        if previsao:
            print(f'Com o dado {X_teste}, É um/o ......')
        else:
            print(f'Com o dado {X_teste}, Não é um/o .......')
    return previsao
