import numpy as np
import copy
from PIL import Image
import glob


def imagem_rec(path: str, num_px: int):  # Pegar a imagem e transformar em um vetor - X-Treino e Y-treino
    Y_set = []
    X_set = []
    for i in range(131):
        i += 1
        path = path + str(f'{i}')
        for arq in glob.glob(path):
            im = np.array(Image.open(arq).resize((num_px, num_px)))
            im = im / 255
            X_set.append(im)
        if i <= 12:
            Y_set.append(1)
        else:
            Y_set.append(0)
    X_set = np.asarray(X_set)
    X_set = X_set.reshape((1, num_px * num_px * 3)).T
    m_treino = X_set.shape[1]
    Y_set = np.asarray(Y_set)
    Y_set = Y_set.reshape((1, m_treino))

    return X_set, Y_set


def sig(z):
    s = 1 / (1 + np.exp(z))
    return s


def iniciar_zeros(x):
    w = np.zeros([x, 1])
    b = float(0)
    return w, b


def propagacao(w, b, X, Y):  # Propagação
    m = X.shape[1]  # O primeiro dado de X = (a, b, c) = A quantidade de treinos
    A = sig(np.dot(w.T, X) + b)  # Função de Loss
    cost = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # Função de Custo/ Cost Func
    dz = A - Y  # Derivadas z, w, b
    dw = (np.dot(X, dz.T)) / m
    db = (1 / m) * np.sum(dz)
    cost = np.squeeze(np.array(cost))  # Remove os eixo/lista de valor 1

    grads = {'dw': dw,  # Armazenando os valores das derivadas para otimizaçõa
             'db': db}
    return grads, cost


def otimizacao(w, b, X, Y, num_int=100, l_rate=0.009, print_cost=False):
    global dw, db  # evitar probelma
    w = copy.deepcopy(w)  # O pq, docs.python.org/pt-br/3/library/copy.html
    b = copy.deepcopy(b)
    costs = []  # armazenando todos os resultados da função de custo - analise
    for i in range(num_int):  # num_int = quantas vezes para treinar w e b
        grads, cost = propagacao(w, b, X, Y)  # Fazendo a Propagação e obtendo os valores das derivadas d, w
        dw = grads['dw']
        db = grads['db']
        w = w - l_rate * dw  # Fazendo a otimização w e b
        b = w - l_rate * db
        if i % 50 == 0:  # A cada 50 interações
            costs.append(cost)  # Armazena o resultado da função de custo
            if print_cost:
                print(f'Função de Custo ({i} interações) : {cost}')
    params = {'w': w,
              'b': b}  # Armazenamento
    grads = {'dw': dw,
             'db': db}
    return params, grads, costs


def prever(w, b, X):
    m = X.shape[1]
    Y_previsao = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)  # O primeiro dado de X = (a, b, c) = A quantidade de treinos, colocando assim para w
    A = sig(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):  # 1 representa a posição daquela imagem, pode alterar
        if A[0, i] > 0.5:  # Para ele ser um gato > 0,5
            Y_previsao[0, i] = 1
        else:
            Y_previsao[0, i] = 0
    return Y_previsao


def modelo(Xtr, Ytr, Xte, Yte, num_int=2000, l_rate=0.5, print_cost=False):
    w, b = iniciar_zeros(Xtr.shape[0])
    params, grads, costs = otimizacao(w, b, Xtr, Ytr, num_int, l_rate, print_cost)
    w = params['w']
    b = params['b']
    Ytr_previsao = prever(w, b, Xtr)
    Yte_previsao = prever(w, b, Xte)
    if print_cost:
        print(f'Precisõ do Treino: {100 - np.mean(np.abs(Ytr_previsao - Ytr) * 100)}')
        print(f'Precisõ do Teste: {100 - np.mean(np.abs(Yte_previsao - Yte) * 100)}')
    resultados = {'custo': costs,
                  'previsao_treino': Ytr_previsao,
                  'previsao_teste': Yte_previsao,
                  'w': w,
                  'b': b,
                  'aprendizado': l_rate,
                  'numint': num_int}
    return resultados
