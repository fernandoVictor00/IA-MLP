import os
import numpy as np

def ler_dados(diretorio):
    dir_atual = os.getcwd()
    dir_completo = dir_atual + '/' + diretorio
    os.chdir(dir_completo)
    t_target = np.loadtxt('respostas.csv', delimiter=',', skiprows=0)
    vanterior = np.loadtxt('vnovo.csv', delimiter=';', skiprows=0)
    v0_anterior = np.loadtxt('v0novo.csv', delimiter=';', skiprows=0)
    wanterior = np.loadtxt('wnovo.csv', delimiter=';', skiprows=0)
    w0_anterior = np.loadtxt('w0novo.csv', delimiter=';', skiprows=0)
    return t_target, vanterior, v0_anterior, wanterior, w0_anterior

def calcular_taxa_acerto(t_target, vanterior, v0_anterior, wanterior, w0_anterior):
    num_entradas, neur = vanterior.shape
    num_saida, num_classes = t_target.shape
    limiar = 0
    zin = np.zeros((1, neur))
    target = np.zeros((num_saida, 1))
    aminicial = 80
    amtestedigitos = 9
    yteste = np.zeros((num_saida, 1))
    k2 = '_'
    k4 = '.txt'
    cont = 0
    contcerto = 0

    for m in range(10):
        k1 = str(m)
        for n in range(amtestedigitos):
            k3a = n + aminicial
            k3 = str(k3a)
            nome = k1 + k2 + k3 + k4
            xteste = np.loadtxt(nome)
            for m2 in range(num_saida):
                for n2 in range(neur):
                    zin[0][n2] = np.dot(xteste, vanterior[:, n2]) + v0_anterior[n2][0]
                z = np.tanh(zin)
                yin = np.dot(z, wanterior) + w0_anterior
                y = np.tanh(yin)
            for j in range(num_saida):
                if yin[0][j] >= limiar:
                    y[0][j] = 1.0
                else:
                    y[0][j] = -1.0
            for j in range(num_saida):
                yteste[j][0] = y[0][j]

            for j in range(num_saida):
                target[j][0] = t_target[j][m]
            soma = np.sum(y - target)
            if soma == 0:
                contcerto = contcerto + 1
            cont = cont + 1
    taxa = contcerto / cont
    return taxa

def main():
    print("\x1b[2J\x1b[1;1H")
    diretorio = 'digitos'
    t_target, vanterior, v0_anterior, wanterior, w0_anterior = ler_dados(diretorio)
    taxa = calcular_taxa_acerto(t_target, vanterior, v0_anterior, wanterior, w0_anterior)
    print('taxa:', taxa)

if __name__ == "__main__":
    main()
