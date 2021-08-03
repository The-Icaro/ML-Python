import numpy as np


dados = open("dino.txt", "r").read()
dados = dados.lower()
letras = list(set(dados))
dados_tamanho, vocab_tamanho = len(dados), len(letras)

letra_index = {l: i for i, l in enumerate(sorted(letras))}
index_letra = {i: l for i, l in enumerate(sorted(letras))}

print(letra_index)
print(index_letra)
print("")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def loss_inicial(vocab_tamanho, seq_tamanho):
    return - np.log(1.0/vocab_tamanho) * seq_tamanho


def suavizar(loss, loss2):
    s = (loss * 0.999) + (loss2 * 0.001)
    return s


def ini(n_a, n_x, n_y, v: float):
    Wax = np.random.randn(n_a, n_x) * v
    Waa = np.random.randn(n_a, n_a) * v
    Wya = np.random.randn(n_y, n_a) * v

    ba = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))

    params = {
        "Wax": Wax,
        "Waa": Waa,
        "Wya": Wya,
        "ba": ba,
        "by": by
    }

    return params


def rnn_celula_frente(x, a_ant, params):
    Wax = params["Wax"]
    Waa = params["Waa"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]

    a_prox = np.tanh(np.dot(Waa, a_ant) + np.dot(Wax, x) + ba)
    y_previsao = softmax(np.dot(Wya, a_prox) + by)

    return a_prox, y_previsao


def rnn_frente(X, Y, a0, params, vocab_tamanho:int):
    x = {}
    a = {}
    y_prev = {}

    a[-1] = np.copy(a0)
    loss = 0

    for i in range(len(X)):
        # One-Hot
        x[i] = np.zeros((vocab_tamanho, 1))
        if X[i] is not None:
            x[i][X[i]] = 1

        a[i], y_prev[i] = rnn_celula_frente(x[i], a[i-1], params)

        loss -= np.sum(y_prev[i] * np.log(Y[i]))

    cache = (y_prev, a, x)

    return loss, cache


def rnn_celula_retro(x, dy, a_prox,  a_ant, params, gradientes):
    Wax = params["Wax"]
    Waa = params["Waa"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]

    dWaa = gradientes["dWaa"]
    dWax = gradientes["dWax"]
    dWya = gradientes["dWya"]
    dba = gradientes["dba"]
    dby = gradientes["dby"]
    da_prox = gradientes["da_prox"]

    da = np.dot(Wya.T, dy) + da_prox
    dtanh = (1 - a_prox * a_prox) * da

    dWax += np.dot(dtanh, x.T)
    dWaa += np.dot(dtanh, a_ant.T)
    dWya += np.dot(dy, a_prox.T)
    dba += dtanh
    dby += dy
    da_prox = np.dot(Waa.T, dtanh)

    gradientes = {
        "da_prox": da_prox,
        "dWax": dWax,
        "dWaa": dWaa,
        "dWya": dWya,
        "dba": dba,
        "dby": dby,
    }
    return gradientes


def rnn_retropropagacao(X, Y, params, cache):
    (y_prev, a, x) = cache

    Wax = params["Wax"]
    Waa = params["Waa"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]

    dWax = np.zeros_like(Wax)
    dWaa = np.zeros_like(Waa)
    dWya = np.zeros_like(Wya)
    dba = np.zeros_like(ba)
    dby = np.zeros_like(by)
    da_prox = np.zeros_like(a[0])

    gradientes = {
        "da_prox": da_prox,
        "dWax": dWax,
        "dWaa": dWaa,
        "dWya": dWya,
        "dba": dba,
        "dby": dby,
    }

    for i in reversed(range(len(X))):
        dy = np.copy(y_prev[i])
        dy[Y[i]] -= 1
        gradientes = rnn_celula_retro(x[i], dy, a[i], a[i - 1], params, gradientes)

    return gradientes, a


def clip(gradientes, max_valor):
    dWaa = gradientes["dWaa"]
    dWax = gradientes["dWax"]
    dWya = gradientes["dWya"]
    dba = gradientes["dba"]
    dby = gradientes["dby"]

    for g in [dWaa, dWax, dWya, dba, dby]:
        np.clip(g, -max_valor, max_valor, out=g)

        gradientes = {
            "dWaa": dWaa,
            "dWax": dWax,
            "dWya": dWya,
            "dba": dba,
            "dby": dby,
        }

    return gradientes


def sample(params, letra_index):
    Waa = params["Waa"]
    Wax = params["Wax"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]

    vocab_tamanho = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_tamanho, 1))
    a_ant = np.zeros((n_a, 1))

    indices = []
    index = -1
    contador = 0
    nova_linha = letra_index["\n"]

    while index != nova_linha and contador != 50:
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_ant) + ba)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        index = np.random.choice(list(range(vocab_tamanho)), p=y.ravel())
        indices.append(index)

        # Resetando Valores / x = One-Hot
        x = np.zeros((vocab_tamanho, 1))
        x[index] = 1
        a_ant = a

    if contador == 50:
        indices.append(letra_index["\n"])

    return indices


def print_sample(samples, index_letras):
    texto = "".join(index_letras[ix] for ix in samples)
    texto = texto[0].upper() + texto[1:]

    print(f'{texto}', end='')


def up_params(params, gradientes, l_r=0.05):
    Wax = params["Wax"]
    Waa = params["Waa"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]

    dWax = gradientes["dWax"]
    dWaa = gradientes["dWaa"]
    dWya = gradientes["dWya"]
    dba = gradientes["dba"]
    dby = gradientes["dby"]

    Wax = Wax - l_r * dWax
    Waa = Waa - l_r * dWaa
    Wya = Wya - l_r * dWya
    ba = ba - l_r * dba
    by = by - l_r * dby

    params = {
        "Wax": Wax,
        "Waa": Waa,
        "Wya": Wya,
        "ba": ba,
        "by": by
    }

    return params


def otimizar(X, Y, a_ant, params, l_r=0.01, vocab_tamanho=27):
    loss, cache = rnn_frente(X, Y, a_ant, params, vocab_tamanho)

    gradientes, a = rnn_retropropagacao(X, Y, params, cache)
    gradientes = clip(gradientes, 5)

    params = up_params(params, gradientes, l_r)

    return loss, gradientes, a[len(X)-1], params


def modelo(index_letra, letra_index, num_it=32000, n_a=50, nomes=7, vocab_tamanho=27, l_r=0.01):
    n_x, n_y = vocab_tamanho, vocab_tamanho

    params = ini(n_a, n_x, n_y, 0.01)
    loss = loss_inicial(vocab_tamanho, nomes)

    with open("dino.txt") as f:
        exemplos = f.readlines()

    exemplos = [x.lower().strip() for x in exemplos]
    np.random.shuffle(exemplos)

    a_ant = np.zeros((n_a, 1))

    for i in range(num_it):
        index = i % len(exemplos)

        X = [None] + [letra_index[l] for l in exemplos[index]]
        Y = X[1:] + [letra_index["\n"]]

        loss_atual, gradientes, a_ant, params = otimizar(X, Y, a_ant, params, l_r)
        loss = suavizar(loss, loss_atual)

        if i % 1000 == 0:
            print(f'Loss: {loss}; Iteração: {i}')

            for nome in range(nomes):
                indices = sample(params, letra_index)
                print_sample(indices, index_letra)
            print("\n")

    return params


parametros = modelo(index_letra, letra_index, num_it=50000, vocab_tamanho=vocab_tamanho)
print(parametros)