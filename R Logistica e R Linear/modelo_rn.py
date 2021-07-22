import numpy as np

#####
# Rede Neural
# Hl pode ser variavel em n_h = hiperparametro
# Funcao de ativacao -> A1 = tangente hiperbolica, A2 = sigma
# Alteracao de fun de ativacao -> tanh = 39, 64; sigma = 41
# v em ini_param = float, quanto menor melhor para otimizacao
# n_it = quantidade de treino = hiperparametro
# l_r = learning rate = hiperparametro
######


def sigm(z):
    sig = 1 / (1 + np.exp(z))
    return sig


def camadas(X_treino, Y_treino):
    n_x = X_treino.shape[0]
    n_y = Y_treino.shape[0]
    return n_x, n_y


def ini_param(n_x, n_y, n_h, v: float):
    W1 = np.random.randn(n_h, n_x) * v
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_x) * v
    b2 = np.zeros((n_y, 1))
    parametros = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parametros


def prop_frente(X_treino, W1, W2, b1, b2):
    Z1 = np.dot(W1, X_treino) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigm(Z2)
    assert A2.shape == (1, X_treino.shape[1])
    resultado = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }
    return resultado


def f_custo(A2, Y):
    m = Y.shape[1]
    custo = (- 1 / m) * np.sum(Y + np.log(A2) + (1 - Y) * np.log(1 - A2))
    custo = float(np.squeeze(custo))
    return custo


def prop_tras(X_treino, Y_treino, A1, A2, W2):
    m = X_treino.shape[1]
    dZ2 = A2 - Y_treino
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X_treino.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    derivadas = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    return derivadas


def otimizar(W1, W2, b1, b2, dW1, dW2, db1, db2, l_r=1):
    W1 = W1 - l_r * dW1
    W2 = W2 - l_r * dW2
    b1 = b1 - l_r * db1
    b2 = b2 - l_r * db2
    parametros = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parametros


def modelo_rn(X_treino, Y_treino, n_h, v, n_it=10000, l_r=1, print_c=False):
    n_x, n_y = camadas(X_treino, Y_treino)
    p = ini_param(n_x, n_y, n_h, v)
    for i in range(0, n_it):
        r = prop_frente(X_treino, p['W1'], p['W2'], p['b1'], p['b2'])
        custo = f_custo(r['A2'], Y_treino)
        d = prop_tras(X_treino, Y_treino, r['A1'], r['A2'], p['W2'])
        p = otimizar(p['W1'], p['W2'], p['b1'], p['b2'],
                     d['dW1'], d['dW2'], d['db1'], d['db2'], l_r)
        if print_c and i % 500 == 0:
            print(f'Custo da {i} it: {custo}')
        return p


def prever(X_teste, W1, W2, b1, b2):
    r = prop_frente(X_teste, W1, W2, b1, b2)
    previsao = r['A2'] > 0.5
    return previsao


