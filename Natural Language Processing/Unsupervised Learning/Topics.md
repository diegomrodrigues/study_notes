## Capítulo 5: Aprendizado Sem Supervisão

**5.1 Aprendizado Sem Supervisão**

- 
- **Desambiguização de Sentido de Palavras como Problema de Aprendizado Sem Supervisão:**  A dificuldade de obter dados rotulados suficientes para desambiguização e a proposta de aprender a estrutura subjacente dos dados sem rótulos. Contexto: Introdução ao problema e motivação para o uso de aprendizado não supervisionado.
- **Agrupamento de Dados em Espaço de Alta Dimensionalidade:**  Identificação de estruturas subjacentes em dados de alta dimensionalidade através do agrupamento (clustering). Contexto:  Extensão da ideia de agrupamento visual em gráficos bidimensionais para espaços de alta dimensionalidade.
- **Algoritmos de Agrupamento:**  A capacidade dos algoritmos de agrupamento em identificar clusters coerentes internamente em dados não rotulados. Contexto:  Introdução à necessidade e funcionalidade dos algoritmos de clustering.

**5.1.1 Agrupamento K-means**

- 
- **Algoritmo K-means:**  Descrição do algoritmo iterativo K-means, alternando entre atribuição de instâncias a clusters e recálculo dos centróides. Contexto:  Algoritmo básico para clustering, com explicação passo a passo.
- **K-means Suave (Soft K-means):**  Atribuição de probabilidades a clusters para cada instância, em vez de atribuições discretas. Contexto:  Variante do K-means que adiciona incerteza na atribuição de clusters.

**5.1.2 Expectation-Maximization (EM)**

- 
- **EM como uma combinação de K-means Suave e Naïve Bayes:**  Combinação da ideia de atribuição suave de clusters com o modelo probabilístico Naïve Bayes. Contexto:  Derivação da metodologia EM com base em modelos probabilísticos.
- **Maximização da Verossimilhança Marginal:**  O objetivo de maximizar a verossimilhança marginal dos dados observados (sem rótulos). Contexto:  Justificativa teórica para a escolha do método de maximização.
- **Introdução de uma Variável Auxiliar q(i):**  Uso de uma distribuição auxiliar q(i) sobre o conjunto de rótulos para aproximar a verossimilhança marginal. Contexto:  Técnica para tornar a otimização mais tratável.
- **Desigualdade de Jensen:**  Uso da desigualdade de Jensen para obter um limite inferior para a verossimilhança marginal. Contexto:  Método para obter um limite inferior para otimizar.
- **Passo E (E-step):**  Atualização da distribuição q(i) como a probabilidade posterior p(z | x(i); φ, μ). Contexto:  Passo de otimização que atualiza as probabilidades de pertencimento aos clusters.
- **Passo M (M-step):**  Atualização dos parâmetros φ e μ usando contagens esperadas em vez de contagens observadas. Contexto:  Passo de otimização que atualiza os parâmetros do modelo.
- **Suavização das Contagens:**  Suavização das contagens esperadas para evitar problemas de zero-frequência. Contexto:  Técnica para melhorar a estabilidade do modelo.
- **Outras Distribuições de Probabilidade:**  Possibilidade de usar outras distribuições de probabilidade além da multinomial (ex: Gaussiana). Contexto:  Extensibilidade do modelo para diferentes tipos de dados.

**5.1.3 EM como um Algoritmo de Otimização**

- 
- **EM como um algoritmo de subida de coordenadas:**  A garantia de que cada passo do EM não diminui o limite inferior J. Contexto:  Propriedades de convergência do algoritmo.
- **Ótimos Locais:**  A sensibilidade da solução EM à inicialização e a possibilidade de convergência para um ótimo local. Contexto:  Limitações do algoritmo EM.
- **EM Hard:**  Variante do EM onde cada distribuição q(i) atribui probabilidade 1 a um único rótulo. Contexto:  Variante que simplifica o algoritmo.
- **EM Incremental/Online:**  Variante do EM que incorpora o descenso de gradiente estocástico. Contexto:  Variante que permite processamento de dados em tempo real.

**5.1.4 Quantos Clusters?**

- 
- **Escolha do Número de Clusters K:**  Métodos para escolher o número de clusters K, incluindo o uso de métricas de qualidade de agrupamento, critério de informação de Akaike (AIC) e a verossimilhança preditiva em dados de teste. Contexto:  Abordagem para determinar o número ideal de clusters.
- **Métodos Bayesianos Não Paramétricos:**  Tratamento do número de clusters como uma variável latente, utilizando inferência estatística em um conjunto de modelos com número variável de clusters. Contexto:  Abordagem mais sofisticada para a determinação do número de clusters.

**5.2 Aplicações de Expectation-Maximization**

**5.2.1 Indução de Sentido de Palavras**

- 
- **EM para Indução de Sentido de Palavras:**  Aplicação do EM para induzir sentidos de palavras a partir de dados não rotulados, tratando cada cluster como um sentido. Contexto:  Aplicação prática da metodologia EM.
- **Decomposição em Valores Singulares (SVD):**  Uso de SVD para obter representações de baixa dimensionalidade dos vetores de contagem de palavras. Contexto:  Técnica para reduzir a dimensionalidade dos dados.

**5.2.2 Aprendizado Semi-Supervisionado**

- 
- **EM para Aprendizado Semi-Supervisionado:**  Aplicação do EM para aprender com dados rotulados e não rotulados, maximizando um limite inferior na verossimilhança conjunta. Contexto:  Combinação de dados rotulados e não rotulados para melhorar a precisão.
- **Balanceamento entre Dados Rotulados e Não Rotulados:**  A importância de balancear o impacto dos dados rotulados e não rotulados na estimação dos parâmetros. Contexto:  Desafio para evitar que os dados não rotulados dominem o modelo.

**5.2.3 Modelagem de Múltiplos Componentes**

- 
- **Modelagem de Múltiplos Componentes para Classificação Supervisionada:**  Introdução de variáveis latentes para modelar componentes dentro de uma mesma classe. Contexto:  Abordagem para lidar com classes com sub-estruturas.

**5.3 Aprendizado Semi-Supervisionado**

- 
- **Auto-Treinamento (Self-Training):**  Uma abordagem para aprendizado semi-supervisionado onde um modelo treinado com dados rotulados é usado para rotular dados não rotulados, que são então usados para treinar o modelo novamente. Contexto:  Abordagem iterativa para aprendizado semi-supervisionado.
- **Aprendizado Multi-Visão (Multi-view Learning):**  Decomposição das características em múltiplas visões condicionalmente independentes, dado o rótulo. Contexto:  Técnica para melhorar a robustez do aprendizado semi-supervisionado.
- **Co-Treinamento (Co-training):**  Um algoritmo de aprendizado multi-visão iterativo, onde classificadores separados para cada visão são treinados e usados para rotular dados não rotulados. Contexto:  Algoritmo específico de aprendizado multi-visão.
- **Algoritmos Baseados em Grafos:**  Construção de um grafo onde pares de instâncias são conectados com pesos simétricos, para propagar rótulos de instâncias rotuladas para instâncias não rotuladas. Contexto:  Abordagem que usa grafos para propagar informação entre instâncias.
- **Propagação de Rótulos:**  Um algoritmo baseado em grafos que propaga rótulos através de operações matriciais. Contexto:  Algoritmo específico baseado em grafos.

**5.4 Adaptação de Domínio**

- 
- **Adaptação de Domínio:**  Aprendizado de modelos que generalizam bem para dados em domínios diferentes do domínio de treinamento. Contexto:  Problema de generalizar modelos para diferentes distribuições de dados.
- **Transferência Direta:**  Treinar um classificador em um domínio e aplicá-lo diretamente a outro domínio. Contexto:  Abordagem simples, mas que pode ter baixo desempenho.
- **Adaptação de Domínio Supervisionada:**  Existência de uma pequena quantidade de dados rotulados no domínio alvo. Contexto:  Cenário onde há dados rotulados no domínio alvo, além dos dados do domínio fonte.
- **Interpolação, Predição e Priores:**  Métodos para combinar previsões de classificadores treinados em diferentes domínios. Contexto:  Métodos para combinar modelos treinados em diferentes domínios.
- **EASYADAPT:**  Criação de cópias de cada característica para cada domínio e para um cenário entre domínios, permitindo que o classificador aloque peso entre as características do domínio específico e entre domínios. Contexto:  Método para lidar com diferenças entre domínios.
- **Adaptação de Domínio Não Supervisionada:**  Ausência de dados rotulados no domínio alvo. Contexto:  Cenário onde não há dados rotulados no domínio alvo.
- **Projeção Linear:**  Aprendizado de uma função de projeção para colocar os dados de diferentes domínios em um espaço compartilhado. Contexto:  Método para alinhar as distribuições dos dados de diferentes domínios.
- **Projeção Não Linear:**  Uso de redes neurais profundas para realizar transformações não lineares dos dados. Contexto:  Método para lidar com relações não lineares entre os domínios.
- **Objetivos de Desruidificação:**  Treinamento de uma função de projeção para reconstruir uma versão corrompida da entrada original. Contexto:  Método para aumentar a robustez do modelo.
- **Objetivos Adversariais:**  Formulação de uma tarefa de classificação auxiliar onde o domínio é tratado como um rótulo. Contexto:  Método para aprender uma representação que seja invariante ao domínio.

**5.5 Outras Abordagens para Aprendizado com Variáveis Latentes**

- 
- **Amostragem (Sampling):**  Abordagem para aprender com variáveis latentes através da amostragem de variáveis latentes a partir de sua distribuição posterior. Contexto:  Método alternativo ao EM.
- **Markov Chain Monte Carlo (MCMC):**  Métodos de amostragem que constroem uma cadeia de Markov para obter amostras da distribuição posterior. Contexto:  Framework para amostragem.
- **Amostragem de Gibbs:**  Um algoritmo MCMC onde cada variável latente é amostrada de sua distribuição posterior condicional às outras variáveis. Contexto:  Algoritmo específico de MCMC.
- **Métodos Bayesianos Empíricos:**  Métodos que mantêm φ e μ como parâmetros em vez de variáveis latentes. Contexto:  Hibridização de EM e MCMC.
- **Amostragem de Gibbs Colapsada:**  Integração analítica de φ e μ fora do modelo. Contexto:  Simplificação do processo de amostragem.
- **Aprendizado Espectral (Spectral Learning):**  Método baseado no método dos momentos para aprender com variáveis latentes, evitando o problema da não convexidade da verossimilhança. Contexto:  Método alternativo ao EM e MCMC.
- **Fatoração de Matrizes Não Negativas:**  Uso de fatoração de matrizes não negativas para garantir que as soluções sejam não negativas. Contexto:  Técnica para garantir a validade das soluções.

Esta lista fornece uma visão concisa e teórica dos conceitos-chave explorados em cada seção do capítulo 5, focada em conceitos avançados para um cientista de dados com conhecimento em IA, estatística e deep learning.  Não inclui ações ou passos práticos.