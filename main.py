import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os

dir_atual = os.getcwd()
dir_completo = dir_atual + '/digitos'

os.chdir(dir_completo)

amp_digitos = 80
vsai = 10
amostras = amp_digitos * vsai
entradas = 256
neur = 200
limiar = 0.0
alfa = 0.005
erro_tolerado = 0.01
lista_ciclo = []
lista_erro = []

x_entrada = np.zeros((amostras, entradas))
k2 = '_'
k4 = '.txt'
contador = 0
ordem_amostras = np.zeros(amostras)

for m in range(vsai):
    k1 = str(m)

    for n in range(amp_digitos):
        k3a = n + 1
        k3 = str(k3a)
        nome_arquivo = k1 + k2 + k3 + k4
        entrada = np.loadtxt(nome_arquivo)
        x_entrada[contador, :] = entrada[:]
        ordem_amostras[contador] = m
        contador = contador + 1

ordem_amostras = ordem_amostras.astype('int')

t_target = np.loadtxt('respostas.csv', delimiter=',', skiprows=0)

pesos_vanterior = np.zeros((entradas, neur))
aleatorio = 0.2

for i in range(entradas):
    for j in range(neur):
        pesos_vanterior[i][j] = rd.uniform(-aleatorio, aleatorio)

pesos_v0_anterior = np.zeros((1, neur))

for j in range(neur):
    pesos_v0_anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

pesos_w_anterior = np.zeros((neur, vsai))
aleatorio = 0.2

for i in range(neur):
    for j in range(vsai):
        pesos_w_anterior[i][j] = rd.uniform(-aleatorio, aleatorio)

pesos_w0_anterior = np.zeros((1, vsai))

for j in range(vsai):
    pesos_w0_anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

pesos_v_novo = np.zeros((entradas, neur))
pesos_v0_novo = np.zeros((1, neur))
pesos_w_novo = np.zeros((neur, vsai))
pesos_w0_novo = np.zeros((1, vsai))
zin = np.zeros((1, neur))
z = np.zeros((1, neur))
deltinha_k = np.zeros((vsai, 1))
deltaw0 = np.zeros((vsai, 1))
deltinha = np.zeros((1, neur))
xaux = np.zeros((1, entradas))
h = np.zeros((vsai, 1))
target = np.zeros((vsai, 1))
deltinha2 = np.zeros((neur, 1))
ciclo = 0
erro_total = 100000

while erro_tolerado < erro_total:

    erro_total = 0

    for padrao in range(amostras):
        for j in range(neur):
            zin[0][j] = np.dot(x_entrada[padrao, :], pesos_vanterior[:, j]) + pesos_v0_anterior[0][j]
        z = np.tanh(zin)
        yin = np.dot(z, pesos_w_anterior) + pesos_w0_anterior
        y = np.tanh(yin)

        for m in range(vsai):
            h[m][0] = y[0][m]
        for m in range(vsai):
            target[m][0] = t_target[m][ordem_amostras[padrao]]

        erro_total = erro_total + np.sum(0.5 * ((target - h) ** 2))

        deltinha_k = (target - h) * (1 + h) * (1 - h)
        deltaw = alfa * (np.dot(deltinha_k, z))
        deltaw0 = alfa * deltinha_k
        deltinhain = np.dot(np.transpose(deltinha_k), np.transpose(pesos_w_anterior))
        deltinha = deltinhain * (1 + z) * (1 - z)

        for m in range(neur):
            deltinha2[m][0] = deltinha[0][m]
        for k in range(entradas):
            xaux[0][k] = x_entrada[padrao][k]

        deltav = alfa * np.dot(deltinha2, xaux)
        deltav0 = alfa * deltinha

        pesos_v_novo = pesos_vanterior + np.transpose(deltav)
        pesos_v0_novo = pesos_v0_anterior + np.transpose(deltav0)
        pesos_w_novo = pesos_w_anterior + np.transpose(deltaw)
        pesos_w0_novo = pesos_w0_anterior + np.transpose(deltaw0)
        pesos_vanterior = pesos_v_novo
        pesos_v0_anterior = pesos_v0_novo
        pesos_w_anterior = pesos_w_novo
        pesos_w0_anterior = pesos_w0_novo

    ciclo = ciclo + 1
    lista_ciclo.append(ciclo)
    lista_erro.append(erro_total)
    print('Ciclo\t Erro')
    print(ciclo, '\t', erro_total)

plt.plot(lista_ciclo, lista_erro)
plt.xlabel('Ciclo')
plt.ylabel('Erro')
plt.show()

print("pesos_v_novo", pesos_v_novo)
print("pesos_v0_novo", pesos_v0_novo)
print("pesos_w_novo", pesos_w_novo)
print("pesos_w0_novo", pesos_w0_novo)

np.savetxt("pesos_v_novo.csv", pesos_v_novo, delimiter=';')
np.savetxt("pesos_v0_novo.csv", pesos_v0_novo, delimiter=';')
np.savetxt("pesos_w_novo.csv", pesos_w_novo, delimiter=';')
np.savetxt("pesos_w0_novo.csv", pesos_w0_novo, delimiter=';')
