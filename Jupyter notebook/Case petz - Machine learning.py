#!/usr/bin/env python
# coding: utf-8

# # Case Petz vaga de cientista de dados Jr

# # Análise de Internações no Sistema de Saúde Brasileiro

# **Introdução**
# 
# Você foi contratado(a) para fazer uma análise apurada do número de internações no
# sistema de saúde brasileiro. Esta análise é de extrema importância para tomada de
# decisões que deverão contribuir para melhorias no sistema e planejamento estratégico.
# Os dados em anexo (case_internacao_SUS.xls) são referentes às internações que
# ocorreram no país durante o período de dezembro de 2017 a julho de 2019, separados
# por região e unidade de federação (Fonte: Ministério da Saúde - Sistema de Informações
# Hospitalares do SUS (SIH/SUS)).

# **Base de dados**
# 
# - Link: http://tabnet.datasus.gov.br/cgi/sih/sxdescr.htm 

# **Tratamento dos dados**
# 
# 
# - 1: Muitas vezes, cerca de 70% do tempo de um projeto é despendido na coleta e tratamento dos dados. Sabendo disso, leia o arquivo e o transforme de modo a ter mais facilidade em analisar os dados. Lembre-se que essa etapa poderá te dar bons insumos.
# 
# 
# 
# **Análise**
# 
# 
# - 2: Dados tratados, bora explorá-los? Faça uma boa EDA e não esqueça de anotar todos os insights que você obter. Gráficos e informações sem uma boa interpretação não valem, ok?
# 
# 
# 
# **Modelagem**
# 
# 
# - 3: Agora que já tem certa intimidade com os dados, cite pelo menos 2 métodos possíveis para estimar os dados para os meses faltantes. Tente não se complicar aqui. Utilize os métodos mais simples e mais funcionais possíveis. Neste tópico, é importante que argumente o porquê dos métodos recomendados.Escolha um desses métodos e estime. 
# 
# 
# - a) o número de Internações.
# 
# - b) o Valor Total das internações nos períodos faltantes.
# 
# 
# 
# **Crie um modelo que preveja** 
# 
# 
# - a) As Internações.
# 
# 
# - b) O número de Óbitos.
# 
# 
# - c) O Valor Médio de AIH pelos próximos 6 meses. 
# 
# Explique a escolha do modelo e quais parâmetros utilizou para serem input no modelo.
# 
# 
# 
# **Planejamento estratégico**
# 
# - Com base nos dados e nas suas análises, que tipo de estratégia você sugeriria para diminuir o número de internações em hospitais do SUS? E para o Estado de São Paulo? Quais especificidades deveriam ser levadas em conta?
# 

# # 0 - Importação das bibliotecas

# In[111]:


# Versão do python
from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[112]:


# Importação das bibliotecas 

import pandas as pd # Pandas carregamento csv
import numpy as np # Numpy para carregamento cálculos em arrays multidimensionais

# Visualização de dados
import seaborn as sns
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")


# In[113]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# In[114]:


# Configuração para os gráficos largura e layout dos graficos

plt.rcParams["figure.figsize"] = (25, 20)

plt.style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

m.rcParams['axes.labelsize'] = 25
m.rcParams['xtick.labelsize'] = 25
m.rcParams['ytick.labelsize'] = 25
m.rcParams['text.color'] = 'k'


# # 0.1) Base de dados

# In[115]:


# Carregando a base de dados
base = pd.read_excel('case_internacao_SUS.xls', sheet_name=None)


# In[116]:


base_1.head()


# # 0.2) Descrição dados
# 
# - Verificação de linhas colunas informaçãos dos dados e tipos de variáveis. Valores das colunas verficando dados nulos ou vazios.

# In[117]:


# Exibido 5 primeiros dados
base_1.head()


# In[118]:


# Exibido 5 últimos dados 
base_1.tail()


# In[119]:


# Número de linhas e colunas
base_1.shape


# In[120]:


# Verificando informações das variaveis
base_1.info()


# In[121]:


# Exibido tipos de dados
base_1.dtypes


# In[122]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(base_1.shape[0]))
print("Números de colunas: {}" .format(base_1.shape[1]))


# In[123]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", base_1.isnull().sum().values.sum())
print("\nUnique values :  \n",base_1.nunique())


# # 0.3) Verificação dos dados
# 
# 

# In[124]:


# Cópia de segurança dos dados
data_1 = base_1.copy()

# Renomeando colunas
n_1 = data_1.columns
n_2 = lambda x: x.lower()
base = list(map(n_2, n_1))

data_1.columns = base
data_1.columns = ["Região",
                  "Internações",
                  "AIH_aprovadas",
                  "Valor_total",
                  "Valor_serviços_hospitalares",
                  "Val_serv_hosp_compl_federal",
                  "Val_serv_hosp_compl_gestor",
                  "Valor_serviços_profissionais",
                  "Val_serv_prof_compl_federal",
                  "Val_serv_prof_compl_gestor",
                  "Valor_médio_AIH",
                  "Valor_médio_intern",
                  "Dias_permanência",
                  "Média_permanência",
                  "Óbitos",
                  "Taxa_mortalidade",
                  "Data"]

data_1.head()


# In[125]:


# Dados faltantes coluna óbitos

data = data_1[data_1["Óbitos"].notnull()]
data.isna().sum()


# In[126]:


# Dados faltantes colunas internacoes

data = data_1[data_1["Internações"].notnull()]
data.isna().sum()


# In[127]:


# Removendo dados ausentes do dataset 

data_1 = data_1.dropna()
data_1.head()


# In[128]:


# Sum() Retorna a soma dos valores sobre o eixo solicitado
# Isna() Detecta valores ausentes

data_1.isna().sum()


# In[129]:


# Retorna a soma dos valores sobre o eixo solicitado
# Detecta valores não ausentes para um objeto semelhante a uma matriz.

data_1.notnull().sum()


# In[130]:


# Total de número duplicados

data_1.duplicated()


# In[131]:


# Renomeando estados por região 

data_1["Região"].unique()


# In[132]:


# Regiãoes que têm pontos(.) antes dos nomes 

data_1 = data_1[data_1['Região'].str.contains('.', regex=False)]
data_1['Região'].unique()


# In[133]:


# Estados vazios

data_1[data_1["Região"].isnull()]


# In[134]:


# Estados vazios um filtro de estados não nulos

data_1 = data_1[data_1['Região'].notnull()]
data_1.head()


# # 0.4) Informação e remoção texto nas colunas 
# 
# **AIH - Aprovadas no período sem considerar prorrogação**
# - Uma parte importante para internação hospitalar.

# In[135]:


# Remoção de pontos 

data_1 = data_1[data_1['Região'].str.contains('.', regex=False)]
data_1['Região'].unique()


# In[136]:


# Uma limpeza na coluna "Região"

data_1['Região'] = data_1['Região'].apply(lambda x: x.replace('.',''))
data_1['Região'] = data_1['Região'].apply(lambda x: x.lstrip())
data_1['Região'] = data_1['Região'].apply(lambda x: x.rstrip())

estados_df = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}

df_estados = {v: k for k, v in estados_df.items()}
data_1['Região'] = data_1['Região'].map(df_estados)

for i in data_1.columns:
    data_1[data_1[i] == '-'] = data_1[data_1[i] == '-'].apply(lambda x: x.replace('-', np.NaN))

# Separando mês e ano nos dados

data_1['mes'] = data_1['Data'].apply(lambda x: x[0:3])
data_1['ano'] = data_1['Data'].apply(lambda x: x[-2:])

# Nessa etapa substituindo meses extensos

meses = {'jan':'1', 
         'fev':'2', 
         'mar':'3', 
         'abr':'4', 
         'mai':'5', 
         'jun':'6', 
         'jul':'7', 
         'ago':'8', 
         'set':'9', 
         'out':'10', 
         'nov':'11', 
         'dez':'12'}

for k,v in meses.items():
    data_1['mes'] = data_1['mes'].apply(lambda x: x.replace(k,v))
    
# Transformando dados ano para 4 dígitos

data_1['ano'] = data_1['ano'].apply(lambda x: '20'+x)

# Os dados em datas

data_1["data"] = data_1["ano"] + "-" + data_1["mes"]

# Visualizando o dataset completo

data_1.head()


# # 0.5) - Limpeza da base de dados
# 
# - Alguns dados tinha dados ausentes e nulos dentro do dataset.

# In[137]:


# Limpando a base de dados

data_1.drop(columns=["Val_serv_hosp_compl_federal", 
                     "Val_serv_hosp_compl_gestor", 
                     "Val_serv_prof_compl_federal",
                     "Val_serv_prof_compl_gestor",
                     "Data"], inplace = True)
data_1.head()


# In[138]:


# Salvando o dataset para modelo 2

data_1.to_csv('data1.csv', index=False)


# In[139]:


# Convertendo os dados para tipo datetime

data_1['data'] = pd.to_datetime(data_1['data'], format='%Y-%m')
data_1.info()


# In[140]:


# Dados faltantes

data_1.fillna(0, inplace=True)
data_1.head()


# In[141]:


# Períodos faltantes

sorted(data_1['data'].unique())


# # 0.6) Estatística descritiva

# In[142]:


# Exibindo estatísticas descritivas visualizar alguns detalhes estatísticos básicos como percentil, média, padrão, etc. 
# De um quadro de dados ou uma série de valores numéricos.

data_1.describe().T


# # 6.1) Gráfico de distribuição normal

# In[143]:


# Gráfico distribuição normal
plt.figure(figsize=(18.2, 8))

ax = sns.distplot(data_1['Taxa_mortalidade']);
plt.title("Distribuição normal", fontsize=20)
plt.xlabel("Total de mortalidade")
plt.ylabel("Total")
plt.axvline(data_1['Taxa_mortalidade'].mean(), color='b')
plt.axvline(data_1['Taxa_mortalidade'].median(), color='r')
plt.axvline(data_1['Taxa_mortalidade'].mode()[0], color='g');
plt.legend(["Media", "Mediana", "Moda"])
plt.show()


# In[144]:


# Verificando os dados no boxplot região valor total verificando possíveis outliers

ax = sns.boxplot(x="Região", y="Valor_total", data = data_1)
plt.title("Gráfico de boxplot - Região o valor total")
plt.xlabel("Total")
plt.ylabel("Valor total")


# In[145]:


# Cálculo da média de internações e óbitos

media_internações = data_1[['data', 'Internações']].groupby('data').mean()
media_obitos = data_1[["data", "Óbitos"]].groupby('data').mean()

print("Média de média internações", media_internações)
print()
print("Média de média óbitos", media_obitos)


# In[146]:


# Verificação média móvel de internações e óbitos

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50.5, 25));
plt.rcParams['font.size'] = '25'

ax1.plot(media_internações, marker='o', color = 'blue', markersize = 15);
ax1.set(title="Média móvel - Internações", xlabel = "Anos", ylabel = "Total de internações")
ax2.plot(media_obitos, marker='o', color = 'blue', markersize = 15);
ax2.set(title="Média móvel - Óbitos", xlabel="Anos", ylabel="Total de óbitos")


# # 6.2) Matriz de correlação dos dados

# In[147]:


# Matriz correlação de pares de colunas, excluindo NA / valores nulos.

corr = data_1.corr()
corr


# In[148]:


# Gráfico da matriz de correlação

plt.figure(figsize=(20,11))
ax = sns.heatmap(data_1.corr(), annot=True, cmap='YlGnBu');
plt.title("Matriz de correlação")


# # 6.3) Análise de dados
# 
# - 2.1 - Análise 

# In[149]:


# Verificando óbitos por ano com gráfico interativo 
fig = px.bar(data_1, x='ano', y='Óbitos', title='Óbitos por ano')
fig.show()


# In[150]:


# Observando total de internações

sns.barplot(x='Valor_total', y='Região', data=data_1)
plt.title("Total de internações do SUS por região")
plt.xlabel("Região os estados")
plt.ylabel("Total")


# In[151]:


# Observando total de internações

sns.histplot(data_1["Internações"])
plt.title("Internações na UTI")
plt.xlabel("Internações")
plt.ylabel("Total")


# In[152]:


# Observando total de óbitos

sns.histplot(data_1["Óbitos"])
plt.title("Total de óbitos")
plt.xlabel("Óbitos")
plt.ylabel("Total")


# In[153]:


# Observando média do valor do AIH

sns.histplot(data_1["Valor_médio_AIH"])
plt.title("Valor médio AIH")
plt.xlabel("Médio AIH")
plt.ylabel("Total")


# In[154]:


# Observando média de internações

sns.histplot(data_1["Valor_médio_intern"])
plt.title("Valor total de internações")
plt.xlabel("Médio internações")
plt.ylabel("Total")


# In[155]:


# Observando total da taxa de mortalidade

sns.histplot(data_1["Taxa_mortalidade"])
plt.title("Valor total taxa mortalidade")
plt.xlabel("Médio internações")
plt.ylabel("Total")


# In[156]:


# Comparando permanência média da UTI

sns.histplot(data_1["Média_permanência"])
plt.title("Valor total da média permanência na UTI")
plt.xlabel("Médio média permanência")
plt.ylabel("Total")


# In[157]:


# Comparando o AIH de aprovados

sns.histplot(data_1["AIH_aprovadas"])
plt.title("Valor total da AIH aprovados")
plt.xlabel("AIH aprovados")
plt.ylabel("Total")


# # 6.4) Análise de dados = Univariada

# In[158]:


# Fazendo um comparativo dos dados 

data_1.hist(bins = 40, figsize=(50.2, 20))
plt.title("Gráfico de histograma")
plt.show()


# # 6.5) Data Processing
# 
# **O processamento de dados começa com os dados em sua forma bruta e os converte em um formato mais legível (gráficos, documentos, etc.), dando-lhes a forma e o contexto necessários para serem interpretados por computadores e utilizados.**
# 
# - Exemplo: Uma letra, um valor numérico. Quando os dados são vistos dentro de um contexto e transmite algum significado, tornam-se informações.

# In[159]:


# Limpeza dos dados
data_1.drop(columns=["Região", "data"], inplace = True)
data_1.head()


# In[160]:


# Mundando os tipo de dados de object para inteiros 

data_1['Óbitos'] = data_1['Óbitos'].astype(int)
data_1['Taxa_mortalidade'] = data_1['Taxa_mortalidade'].astype(int)
data_1['Internações'] = data_1['Internações'].astype(int)
data_1.info()


# # 6.6) Feature Engineering
# 
# - Praticamente todos os algoritmos de Aprendizado de Máquina possuem entradas e saídas. As entradas são formadas por colunas de dados estruturados, onde cada coluna recebe o nome de feature, também conhecido como variáveis independentes ou atributos. Essas features podem ser palavras, pedaços de informação de uma imagem, etc. Os modelos de aprendizado de máquina utilizam esses recursos para classificar as informações. 
# 
# **Por exemplo, sedentarismo e fator hereditário são variáveis independentes para quando se quer prever se alguém vai ter câncer ou não**  
# 
# - As saídas, por sua vez, são chamadas de variáveis dependentes ou classe, e essa é a variável que estamos tentando prever. O nosso resultado pode ser 0 e 1 correspondendo a 'Não' e 'Sim' respectivamente, que responde a uma pergunta como: "Fulano é bom pagador?" ou a probabilidade de alguém comprar um produto ou não.

# In[161]:


# Importando a biblioteca para pré-processamento 

from sklearn.preprocessing import LabelEncoder

for i in data_1.columns:
    if data_1[i].dtype==np.number:
        continue
    data_1[i]= LabelEncoder().fit_transform(data_1[i])
    
data_1.head(4)


# # 6.7) Treino e Teste
# 
# - Treino e teste da base de dados da coluna Internações

# In[162]:


y = data_1['Internações'] # Variável para teste
x = data_1.drop('Internações', axis=1) # Variável para treino


# In[163]:


# Total de linhas e colunas dados variável x
x.shape


# In[164]:


# Total de linhas e colunas dados variável y
y.shape


# # 6.8) - Escalonamento
# 
# - Escalonamento uma forma de contornar os problemas relacionados à escala, mantendo a informação estatística dos dados. O procedimento consiste em realizar uma transformação sobre o conjunto original dos dados de modo que cada variável apresente média zero e variância unitária.

# In[165]:


# Importando a biblioteca sklearn para o escalonamneto dos dados

from sklearn.preprocessing import StandardScaler 

scaler_pre = StandardScaler() # Inicializando o escalonamento
scaler_pre_fit_train = scaler_pre.fit_transform(x) # Treinamento com a função fit_transform com a variável x
scaler_pre_fit_train # Imprimindo o valor do escalonamento


# # 6.9) Modelo treinado para x, y valor
# 
# - 20% para os dados de treino
# - 80% para teste
# - Random state igual a zero

# In[166]:


# Importação da biblioteca sklearn para treino e teste do modelo

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, # Variável x
                                                    y, # Variável y
                                                    test_size=0.2, # Divivindo os dados em 20% para treino e 80% para teste
                                                    random_state = 0) # Random state igual a zero


# In[167]:


# Total de linhas e colunas e linhas dos dados de treino x

x_train.shape


# In[168]:


# Total de linhas dos dados de treino y

y_train.shape


# In[169]:


# Total de linhas e colunas dos dados de treino x teste 

x_test.shape


# In[170]:


# Total de linhas e colunas dos dados de treino y teste 

y_test.shape


# # 7.0) Modelo machine learning 
# 
# Eu utlizei modelo de regressão linear para prever internações, óbitos e valor médio de AIH.

# # 7.1) Modelo 01 - Regressão linear
# 
# - Nesse modelo estamos prevendo o número de internações utilizando modelo de regressão linear.

# In[171]:


# Modelo regressão linear - 1
# Importação da biblioteca sklearn o modelo regressão linear

from sklearn.linear_model import LinearRegression 

# Nome do algoritmo M.L
model_linear = LinearRegression() 

# Treinamento do modelo
model_linear_fit = model_linear.fit(x_train, y_train)

# Score do modelo
model_linear_score_1 = model_linear.score(x_train, y_train)

print("Modelo - Regressão linear: %.2f" % (model_linear_score_1 * 100))


# In[172]:


# Previsão do modelo

model_linear_pred = model_linear.predict(x_test)
model_linear_pred


# In[173]:


# O intercepto representa o efeito médio em tendo todas as variáveis explicativas excluídas do modelo. 
# De forma mais simples, o intercepto representa o efeito médio em são iguais a zero.

model_linear.intercept_


# In[174]:


# Os coeficientes de regressão  𝛽2 ,  𝛽3  e  𝛽4  são conhecidos como coeficientes parciais de regressão ou coeficientes parciais angulares. 
# Considerando o número de variáveis explicativas de nosso modelo, seu significado seria o seguinte

model_linear.coef_


# In[175]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(model_linear.score(x_train, y_train).round(2)))


# In[176]:


# Previsão do modelo 
pred = model_linear.predict(x_train)
pred2 = y_train - pred
pred2


# # Gráfico de regressão linear

# In[177]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, y_train)
plt.plot(pred, model_linear.predict(x_train), color = "red")
plt.title("Grafico de regressão linear", fontsize = 20)
plt.xlabel("Total")
plt.ylabel("Totoal de internações")
plt.legend(["Valor", "Internações"])


# # 7.2) Distribuição de Frequências dos Resíduos

# In[178]:


# Gráfico de distribuição Frequências

ax = sns.distplot(pred)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Internações', fontsize=14)
ax


# # 7.3) Métricas para o modelo de regressão linear

# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[179]:


# Importando bibliotecas verificações das métricas 

from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, model_linear_pred))
mae = mean_absolute_error(y_test, model_linear_pred)
mape = mean_absolute_percentage_error(y_test, model_linear_pred)
mse = mean_squared_error(y_test, model_linear_pred)
r2 = r2_score(y_test, model_linear_pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[222]:


# Previsão de internações

prev = x_test[0:25]
model_pred = model_linear.predict(prev)[0]
print("Previsão de internações", model_pred)
prev


# # 7.4) Modelo 02
# 
# - Nesse segundo modelo estamos prevendo o número de óbitos utilizando modelo de regressão linear.

# In[181]:


# Criando uma Series (pandas) para armazenar números de óbitos

test = data_1['Óbitos'] # Variável para teste
train = data_1.drop('Óbitos', axis=1) # Variável para treino


# In[182]:


# Total de linhas e colunas dados variável train

train.shape


# In[183]:


# Total de linhas e colunas dados variável test

test.shape


# # 7.5) Escalonamento dos dados

# In[184]:


# Importando a biblioteca sklearn para o escalonamneto dos dados

from sklearn.preprocessing import StandardScaler

scaler_pre = StandardScaler() # Inicializando o escalonamento
scaler_pre_fit_train = scaler_pre.fit_transform(train) # Treinamento com a função fit_transform com a variável x
scaler_pre_fit_train # Imprimindo o valor do escalonamento


# # 7.6) Modelo treinado para x, y valor
# 
# - 20% para os dados de treino
# - 80% para teste
# - Random state igual a zero

# In[185]:


# Importação da biblioteca sklearn para treino e teste do modelo

from sklearn.model_selection import train_test_split 

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(train, # Variável train
                                                    test, # Variável test
                                                    test_size=0.2, # Divivindo os dados em 20% para treino e 80% para teste
                                                    random_state = 0) # Random state igual a zero


# In[186]:


# Total de linhas e colunas e linhas dos dados de treino x

x_train_1.shape


# In[187]:


# Total de linhas dos dados de treino y

y_train_1.shape


# In[188]:


# Total de linhas e colunas dos dados de treino x teste 

x_test_1.shape


# In[189]:


# Total de linhas e colunas dos dados de treino y teste 

y_test_1.shape


# # 7.7) Modelo regressão linear - 2

# In[190]:


# Modelo regressão linear - 2 Óbitos
# Importação da biblioteca sklearn o modelo regressão linear

from sklearn.linear_model import LinearRegression

# Nome do algoritmo M.L
model_linear_2 = LinearRegression() 

# Treinamento do modelo
model_linear_fit = model_linear_2.fit(x_train_1, y_train_1)

# Score do modelo
model_linear_score_2 = model_linear_2.score(x_train_1, y_train_1)

print("Modelo - Regressão linear: %.2f" % (model_linear_score_2 * 100))


# In[191]:


# Previsão do modelo

model_linear_pred_2 = model_linear_2.predict(x_test_1)
model_linear_pred_2


# In[192]:


# O intercepto representa o efeito médio em tendo todas as variáveis explicativas excluídas do modelo. 
# De forma mais simples, o intercepto representa o efeito médio em são iguais a zero.

model_linear_2.intercept_


# In[193]:


# Os coeficientes de regressão  𝛽2 ,  𝛽3  e  𝛽4  são conhecidos como coeficientes parciais de regressão ou coeficientes parciais angulares. 
# Considerando o número de variáveis explicativas de nosso modelo, seu significado seria o seguinte

model_linear_2.coef_


# In[194]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(model_linear_2.score(x_train_1, y_train_1).round(2)))


# In[195]:


# Previsão do modelo 

pred_2 = model_linear_2.predict(x_train_1)
pred_2 = y_train - pred_2
pred_2


# In[196]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, y_train_1)
plt.title("Grafico de regressão linear - Óbitos", fontsize = 20)
plt.xlabel("Total")
plt.ylabel("Totoal de óbitos")
plt.legend(["Óbitos", "Valor"])


# In[197]:


# Distribuição de Frequências dos Resíduos

ax = sns.distplot(pred_2)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Internações', fontsize=14)
ax


# # 7.8) Métricas para o modelo 2 regressão linear
# 
# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[198]:


# Verificações das métricas 

rmse = np.sqrt(mean_squared_error(y_test, model_linear_pred_2))
mae = mean_absolute_error(y_test, model_linear_pred_2)
mape = mean_absolute_percentage_error(y_test, model_linear_pred_2)
mse = mean_squared_error(y_test, model_linear_pred_2)
r2 = r2_score(y_test, model_linear_pred_2)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[221]:


# Previsão de obitos

prev_2 = x_test_1[0:25]
model_pred_2 = model_linear_2.predict(prev_2)[0]
print("Previsão de óbitos", model_pred_2)
prev_2


# # 7.9) Modelo 03: Regressão linear
# 
# - Nesse modelo estamos prevendo o valor Médio de AIH pelos próximos 6 meses utilizando modelo de regressão linear.

# # Treino e Teste
# 
# - Treino e teste da base de dados da coluna Internações

# In[224]:


y2 = data_1['Valor_médio_AIH'] # Variável para y2
x1 = data_1.drop('Valor_médio_AIH', axis=1) # Variável para x1


# In[225]:


# Total de linhas e colunas dados variável x

x1.shape


# In[226]:


# Total de linhas e colunas dados variável y

y2.shape


# # 8.0) Escalonamento dos dados

# In[227]:


# Importando a biblioteca sklearn para o escalonamneto dos dados

from sklearn.preprocessing import StandardScaler 

scaler_pre = StandardScaler() # Inicializando o escalonamento
scaler_pre_fit_train = scaler_pre.fit_transform(x1) # Treinamento com a função fit_transform com a variável x1
scaler_pre_fit_train # Imprimindo o valor do escalonamento


# # 9.0) Modelo treinado para x, y valor
# 
# - 20% para os dados de treino
# - 80% para teste
# - Random state igual a zero

# In[228]:


# Importação da biblioteca sklearn para treino e teste do modelo

from sklearn.model_selection import train_test_split 

train_x, test_x, train_y, test_y = train_test_split(train, # Variável x1
                                                    test, # Variável y2
                                                    test_size=0.2, # Divivindo os dados em 20% para treino e 80% para teste
                                                    random_state = 0) # Random state igual a zero


# In[229]:


# Total de linhas e colunas e linhas dos dados de treino x

train_x.shape


# In[230]:


# Total de linhas dos dados de treino y

train_y.shape


# In[231]:


# Total de linhas e colunas dos dados de treino x teste 

test_x.shape


# In[232]:


# Total de linhas e colunas dos dados de treino y teste 

test_y.shape


# In[233]:


# Modelo regressão linear - 3 Valor Médio de AIH
# Importação da biblioteca sklearn o modelo regressão linear

from sklearn.linear_model import LinearRegression

# Nome do algoritmo M.L
model_linear_3 = LinearRegression() 

# Treinamento do modelo
model_linear_fit = model_linear_3.fit(train_x, train_y)

# Score do modelo
model_linear_score_3 = model_linear_3.score(x_train_1, y_train_1)

print("Modelo - Regressão linear: %.2f" % (model_linear_score_3 * 100))


# In[234]:


# Previsão do modelo

model_linear_pred_3 = model_linear_3.predict(x_test_1)
model_linear_pred_3


# In[235]:


# O intercepto representa o efeito médio em tendo todas as variáveis explicativas excluídas do modelo. 
# De forma mais simples, o intercepto representa o efeito médio em são iguais a zero.

model_linear_3.intercept_


# In[236]:


# Os coeficientes de regressão  𝛽2 ,  𝛽3  e  𝛽4  são conhecidos como coeficientes parciais de regressão ou coeficientes parciais angulares. 
# Considerando o número de variáveis explicativas de nosso modelo, seu significado seria o seguinte.

model_linear_3.coef_


# In[237]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(model_linear_3.score(x_train_1, y_train_1).round(2)))


# In[238]:


# Previsão do modelo

pred_2 = model_linear_3.predict(train_x)
pred_2 = y_train - pred_2
pred_2


# In[239]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, train_y)
plt.title("Grafico de regressão linear", fontsize = 20)
plt.xlabel("Total")
plt.ylabel("Totoal de Valor Médio de AIH")
plt.legend(["Valor Médio de AIH", "Valor"])


# In[240]:


# Distribuição de Frequências dos Resíduos

ax = sns.distplot(pred_2)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Internações', fontsize=14)
ax


# # 1.0) Métricas para o modelo 3 regressão linear 
# 
# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[241]:


# Verificações das métricas 

rmse = np.sqrt(mean_squared_error(y_test, model_linear_pred_3))
mae = mean_absolute_error(y_test, model_linear_pred_3)
mape = mean_absolute_percentage_error(y_test, model_linear_pred_3)
mse = mean_squared_error(y_test, model_linear_pred_3)
r2 = r2_score(y_test, model_linear_pred_3)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultados'])


# In[242]:


# Previsão valor Médio de AIH

prev_3 = x_test_1[0:25]
model_pred_3 = model_linear_3.predict(prev_3)[0]
print("Previsão total valor Médio de AIH:", model_pred_3)
prev_3


# # 1.1) Resultados final dos modelos

# In[243]:


# Exibindo um comparativo dos modelos de regressão linear

modelos = pd.DataFrame({
    
    "Modelos" :[ "Modelo regressão linear 1", 
                "Modelo regressão linear 2", 
                "Modelo regressão linear 3"],

    "Acurácia" :[model_linear_score_1, 
                 model_linear_score_2, 
                 model_linear_score_3]})

modelos.sort_values(by = "Acurácia", ascending = True)


# In[244]:


# Salvando modelo de regressão linear

import pickle

with open('model_linear_pred.pkl', 'wb') as file:
    pickle.dump(model_linear_pred, file)
    
with open('model_linear_pred_2.pkl', 'wb') as file:
    pickle.dump(model_linear_pred_2, file)
    
with open('model_linear_pred_3.pkl', 'wb') as file:
    pickle.dump(model_linear_pred_3, file)


# # Conclusão do modelo machine learning

# Pela análise dos modelos, modelo 1 teve melhor resultado que os demais, atigindo uma acurácia de 97.18% ou seja capaz de acertar as previsões de internações, óbitos, valor do AIH. De acordo com análise realizada.  
