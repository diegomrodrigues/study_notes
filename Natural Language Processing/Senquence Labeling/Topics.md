## Capítulo 7: Rotulação de Sequências

**7.1 Rotulação de Sequências como Classificação**

- 
- **Rotulação de Sequências como um Problema de Classificação:**  Formulação da rotulação de sequências como um problema de classificação, onde cada elemento da sequência é classificado individualmente. Contexto: Abordagem simples para rotular sequências, tratando cada token de forma independente.
- **Função de Característica Simples:**  Uso da própria palavra como característica para a classificação. Contexto:  Feature simples baseada apenas na palavra em si.
- **Incorporando Contexto:**  Incorporação de palavras vizinhas como características para capturar contexto. Contexto:  Melhoria na função de característica para considerar o contexto da palavra.
- **Relações entre Rótulos:**  Necessidade de modelar relações entre rótulos adjacentes para lidar com ambiguidades gramaticais. Contexto:  Limitações da abordagem de classificação independente e a necessidade de modelar dependências entre rótulos.

**7.2 Rotulação de Sequências como Predição de Estrutura**

- 
- **Rotulação de Sequências como Predição de Estrutura:**  Consideração da sequência inteira de rótulos como uma única estrutura. Contexto:  Abordagem que considera as dependências entre os rótulos da sequência.
- **Função de Pontuação:**  Definição de uma função de pontuação para avaliar a qualidade de uma sequência de rótulos. Contexto:  Funcionalidade para avaliar a sequência de rótulos.
- **Inferência:**  O desafio de realizar a inferência em um espaço de rótulos exponencialmente grande. Contexto:  Desafio computacional de buscar a melhor sequência de rótulos.
- **Decomposição da Função de Pontuação:**  Decomposição da função de pontuação em partes locais para tornar a inferência tratável. Contexto:  Técnicas para tornar o problema computacionalmente viável.
- **Modelo Linear:**  Definição da função de pontuação como um produto escalar entre pesos e características. Contexto:  Modelo matemático para a função de pontuação.
- **Características de Emissão e Transição:**  Incorporação de características para pares palavra-rótulo e rótulo-rótulo. Contexto:  Features para modelar as dependências entre palavras e rótulos.

**7.3 O Algoritmo de Viterbi**

- 
- **Algoritmo de Viterbi:**  Um algoritmo de programação dinâmica para encontrar a sequência de rótulos com maior pontuação. Contexto:  Algoritmo eficiente para inferência em modelos com decomposição local da função de pontuação.
- **Variáveis de Viterbi:**  Variáveis auxiliares que representam a pontuação da melhor sequência de rótulos terminando em um determinado rótulo em uma determinada posição. Contexto:  Variáveis auxiliares para o algoritmo Viterbi.
- **Recorrência de Viterbi:**  Recorrência para calcular as variáveis de Viterbi. Contexto:  Equação de recorrência para calcular as variáveis Viterbi.
- **Condição Inicial:**  Condição inicial para a recorrência de Viterbi. Contexto:  Valor inicial para o cálculo recursivo.
- **Trellis:**  Representação gráfica do algoritmo de Viterbi. Contexto:  Representação visual do cálculo.
- **Ponteiros para Trás:**  Armazenamento dos ponteiros para trás para recuperar a sequência de rótulos ótima. Contexto:  Armazenamento de informação para recuperação da sequência ótima.
- **Complexidade:**  Complexidade temporal e espacial do algoritmo de Viterbi. Contexto:  Eficiência computacional do algoritmo.

**7.3.1 Exemplo**

- 
- **Exemplo de Aplicação do Algoritmo de Viterbi:**  Exemplo numérico detalhado da aplicação do algoritmo de Viterbi. Contexto:  Demonstração prática do algoritmo com valores numéricos.

**7.3.2 Características de Ordem Superior**

- 
- **Características de Ordem Superior:**  Generalização do algoritmo de Viterbi para incluir características de ordem superior (trigramas, etc.). Contexto:  Incorporação de mais contexto na função de pontuação.

**7.4 Modelos de Markov Ocultos (HMMs)**

- 
- **Modelos de Markov Ocultos (HMMs):**  Modelos probabilísticos para rotulação de sequências, baseados em probabilidades de emissão e transição. Contexto:  Modelo probabilístico para rotular sequências.
- **Suposições de Independência:**  Suposições de independência nos HMMs. Contexto:  Simplificações para tornar o problema computacionalmente tratável.
- **Processo Gerativo:**  Processo gerativo para HMMs. Contexto:  Descrição da geração probabilística das sequências.
- **Estimação de Parâmetros:**  Estimação de parâmetros de emissão e transição em HMMs. Contexto:  Como estimar os parâmetros do modelo.
- **Inferência em HMMs:**  Inferência em HMMs usando o algoritmo de Viterbi. Contexto:  Como encontrar a sequência de rótulos mais provável.
- **Algoritmo de Viterbi para HMMs:**  Interpretação probabilística do algoritmo de Viterbi para HMMs. Contexto:  Relação entre o algoritmo Viterbi e o modelo probabilístico HMM.

**7.5 Rotulação de Sequências Discriminativas com Características**

- 
- **Limitações dos HMMs:**  Limitações dos HMMs em relação à representação de informações contextuais ricas. Contexto:  Motivação para usar métodos mais gerais.
- **Incorporação de Informações Adicionais:**  Incorporação de características adicionais (afixos, contexto de grão fino, etc.) para melhorar a precisão. Contexto:  Incorporação de features para capturar informações mais detalhadas.
- **Função de Característica Global:**  Construção de uma função de característica global como soma de vetores de característica locais. Contexto:  Agrupamento das features em um único vetor.

**7.5.1 Perceptron Estruturado**

- 
- **Perceptron Estruturado:**  Extensão do algoritmo perceptron para predição de estruturas. Contexto:  Algoritmo para treinar o modelo.
- **Atualização do Perceptron Estruturado:**  Atualização dos pesos em perceptron estruturado. Contexto:  Como atualizar os pesos do modelo.

**7.5.2 Máquinas de Vetores de Suporte Estruturadas (SVMs)**

- 
- **Máquinas de Vetores de Suporte Estruturadas (SVMs):**  Extensão das SVMs para predição de estruturas, utilizando restrições de margem para melhorar a generalização. Contexto:  Algoritmo para treinar o modelo, buscando margens maiores.
- **Restrição de Margem Aumentada por Custo:**  Incorporação de uma função de custo para ponderar diferentes tipos de erros. Contexto:  Como lidar com erros de diferentes magnitudes.
- **Decodificação Aumentada por Custo:**  Algoritmo para encontrar a predição que mais viola a restrição de margem. Contexto:  Método para encontrar o ponto mais crítico para atualização do modelo.
- **Otimização para SVMs Estruturadas:**  Algoritmos de otimização para SVMs estruturadas. Contexto:  Métodos para otimizar os pesos do modelo.

**7.5.3 Campos Aleatórios Condicionais (CRFs)**

- 
- **Campos Aleatórios Condicionais (CRFs):**  Um modelo probabilístico condicional para rotulação de sequências. Contexto:  Modelo probabilístico para rotular sequências.
- **Decodificação em CRFs:**  Decodificação em CRFs usando o algoritmo de Viterbi. Contexto:  Método para encontrar a sequência mais provável.
- **Aprendizado em CRFs:**  Aprendizado em CRFs minimizando a verossimilhança negativa logarítmica regularizada. Contexto:  Método para treinar o modelo.
- **Algoritmo Forward:**  Algoritmo para computar a função de partição em CRFs. Contexto:  Algoritmo para calcular a normalização da probabilidade.
- **Algoritmo Forward-Backward:**  Algoritmo para computar probabilidades marginais em CRFs. Contexto:  Método para calcular as probabilidades marginais dos rótulos.

**7.6 Rotulação de Sequências Neurais**

**7.6.1 Redes Neurais Recorrentes (RNNs)**

- 
- **Redes Neurais Recorrentes (RNNs) para Rotulação de Sequências:**  Uso de RNNs para modelar a dependência entre rótulos em sequências. Contexto:  Aplicação de RNNs para rotulação de sequências.
- **Redes Neurais Recorrentes Bidirecionais:**  Uso de duas RNNs, uma para frente e outra para trás, para capturar contexto em ambas as direções. Contexto:  Modelo para capturar o contexto completo da palavra.
- **Predição de Estrutura Neural:**  Combinação de RNNs com o algoritmo de Viterbi para predição conjunta de estruturas. Contexto:  Combinação de RNN com Viterbi para considerar todas as dependências na sequência.

**7.6.2 Modelos de Nível de Caractere**

- 
- **Modelos de Nível de Caractere:**  Construção de representações de palavras a partir de suas grafias para lidar com palavras raras ou desconhecidas. Contexto:  Tratamento de palavras fora do vocabulário do modelo.

**7.6.3 Redes Neurais Convolucionais para Rotulação de Sequências**

- 
- **Redes Neurais Convolucionais (CNNs) para Rotulação de Sequências:**  Uso de CNNs para rotulação de sequências, explorando o paralelismo computacional. Contexto:  Uso de CNNs para obter melhor desempenho computacional.

**7.7 Rotulação de Sequências Não Supervisionada**

- 
- **Rotulação de Sequências Não Supervisionada:**  Indução de um modelo HMM a partir de um corpus de texto não anotado. Contexto:  Aprendizado de modelos sem dados rotulados.
- **Algoritmo Baum-Welch:**  Algoritmo para aprendizado não supervisionado em HMMs usando EM. Contexto:  Algoritmo para aprender os parâmetros do modelo HMM sem dados rotulados.
- **Sistemas Dinâmicos Lineares:**  Extensão do algoritmo forward-backward para espaços de estados contínuos. Contexto:  Generalização para dados contínuos.
- **Métodos Alternativos de Aprendizado Não Supervisionado:**  Métodos alternativos como MCMC e aprendizado espectral. Contexto:  Métodos alternativos para aprendizado não supervisionado.
- **Notação Semiring e Algoritmo de Viterbi Generalizado:**  Expressão dos algoritmos de Viterbi e Forward em uma notação semiring generalizada. Contexto:  Abstração matemática que unifica os diferentes algoritmos.

Esta lista abrange os conceitos-chave de cada seção do Capítulo 7, focando em uma perspectiva teórica e avançada para um cientista de dados.  Não inclui passos práticos ou implementações de código.