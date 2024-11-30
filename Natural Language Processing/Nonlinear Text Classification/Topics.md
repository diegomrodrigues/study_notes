## Chapter 3: Nonlinear Classification

### 3.1 Feedforward Neural Networks

#### Multilayer Classifiers

Os classificadores multilayer constituem a base das redes neurais profundas, onde múltiplas camadas de neurônios são empilhadas para realizar transformações sucessivas dos dados de entrada. Teoricamente, a composição de funções não lineares através de camadas sucessivas permite que a rede universalmente aproxime qualquer função contínua, conforme o **Teorema de Aproximação Universal**. Este teorema estabelece que, sob condições adequadas (como a existência de pelo menos uma camada oculta com um número finito de neurônios e funções de ativação não lineares), uma rede neural feedforward pode aproximar arbitrariamente bem qualquer função contínua definida em um espaço compacto.

Em NLP, essa capacidade é crucial para modelar interações complexas entre tokens e contextos, capturando relações semânticas de alto nível que não podem ser representadas por modelos lineares. Matemáticamente, considere uma rede com $L$ camadas, onde cada camada $l$ aplica uma transformação $h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$, sendo $\sigma$ a função de ativação não linear. A composição dessas transformações permite a modelagem de funções altamente complexas e não lineares necessárias para capturar nuances linguísticas em NLP.

#### Hidden Layers

As camadas ocultas introduzem representações intermediárias que transformam os dados de entrada em formas mais abstratas e informativas. Do ponto de vista teórico, essas camadas permitem a decomposição de problemas complexos em subproblemas mais gerenciáveis, facilitando a aprendizagem de representações hierárquicas. Em NLP, as camadas ocultas são responsáveis por capturar nuances linguísticas, como polissemia e ambiguidade, através da construção de embeddings de alta dimensionalidade que refletem relações semânticas profundas entre palavras e frases.

Matematicamente, cada camada oculta $l$ aplica uma transformação não linear $h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$, onde $W^{(l)}$ é a matriz de pesos e $b^{(l)}$ o vetor de vieses. A profundidade da rede (número de camadas ocultas) influencia a capacidade da rede de aprender representações de maior abstração. Teoricamente, camadas adicionais permitem a rede de capturar interações de ordem superior entre as entradas, aumentando a expressividade do modelo.

#### Activation Functions (sigmoid, softmax)

As funções de ativação são elementos cruciais que introduzem não linearidade nas redes neurais, permitindo a modelagem de relações complexas. A função **sigmoid**, definida como $\sigma(x) = \frac{1}{1 + e^{-x}}$, possui propriedades de mapeamento contínuo e diferenciável, sendo utilizada para tarefas de classificação binária. Sua derivada $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ permite o cálculo eficiente dos gradientes durante o treinamento.

A função **softmax**, definida para um vetor $z$ como $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$, normaliza as ativações da camada de saída em uma distribuição de probabilidade, sendo essencial para tarefas de classificação multiclasse em NLP, como a predição de palavras em modelos de linguagem ou a classificação de sentenças em categorias semânticas. A softmax é frequentemente combinada com a **perda de log-verossimilhança negativa** ($\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$), onde $y$ são as labels verdadeiras e $\hat{y}$ as predições da softmax, facilitando a otimização através de métodos de descida de gradiente.

#### Computation Graph

O grafo de computação é uma representação abstrata da arquitetura da rede neural, onde nós representam operações matemáticas e arestas representam o fluxo de dados. Teoricamente, os grafos de computação facilitam a análise da complexidade computacional e a aplicação de técnicas de otimização, como o **backpropagation**. Em NLP, a estrutura sequencial dos dados textuais exige grafos de computação que suportem operações como convoluções sobre sequências e integrações de embeddings, garantindo eficiência e modularidade na implementação de modelos complexos.

Matematicamente, um grafo de computação pode ser representado como um conjunto de operações $\{f_1, f_2, \dots, f_n\}$ conectadas por fluxos de dados $\{x_1, x_2, \dots, x_m\}$. Cada operação $f_i$ pode depender de uma ou mais entradas, formando uma estrutura acíclica que permite a aplicação eficiente da diferenciação automática e do cálculo de gradientes durante o treinamento.

### 3.2 Designing Neural Networks

#### Activation Functions (tanh, ReLU, Leaky ReLU)

A escolha da função de ativação impacta diretamente a capacidade de aprendizagem e a estabilidade da rede. A função **tanh**, definida como $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$, mapeia os inputs para um intervalo de \([-1, 1]\), oferecendo uma média zero centrada, o que facilita a convergência do treinamento ao reduzir a covariância entre as ativações.

A **ReLU** (Rectified Linear Unit), definida como $f(x) = \max(0, x)$, introduz sparsidade nas ativações e mitiga o problema do gradiente desaparecido, promovendo a eficiência computacional e a rapidez na convergência. No entanto, a ReLU pode levar ao problema de **neurônios mortos**, onde certos neurônios deixam de atualizar seus pesos devido a entradas negativas constantes.

A **Leaky ReLU**, uma variante da ReLU, é definida como $f(x) = \max(\alpha x, x)$, onde $\alpha$ é um pequeno valor positivo (por exemplo, 0.01). Esta função permite pequenas derivadas para inputs negativos, abordando o problema de neurônios mortos e mantendo a capacidade de aprendizagem mesmo em regiões de baixa ativação.

Em NLP, a escolha adequada da função de ativação é essencial para modelar dependências linguísticas complexas sem comprometer a estabilidade do treinamento. A ReLU e suas variantes são frequentemente preferidas devido à sua eficiência e capacidade de acelerar a convergência em modelos profundos.

#### Network Structure (width vs. depth)

A estrutura da rede, especificamente a largura versus a profundidade, afeta a capacidade de representação e a eficiência computacional. Redes mais **largas**, com mais neurônios por camada, podem capturar uma variedade maior de características em cada nível de abstração, promovendo a diversidade nas representações. Matemáticamente, a capacidade de representação de uma rede com $L$ camadas e $N$ neurônios por camada é proporcional a $N^L$, permitindo que redes largas representem uma gama ampla de funções.

Redes mais **profundas**, com mais camadas, permitem a composição de múltiplas transformações não lineares, capturando hierarquias linguísticas complexas. Teoricamente, redes profundas podem representar funções exponencialmente mais complexas com um número polinomial de neurônios, uma vantagem crucial para modelar a complexidade da linguagem natural. A profundidade permite que a rede capture interações de ordem superior entre as palavras, como dependências sintáticas de longo alcance e relações semânticas complexas.

Em tarefas de NLP, redes profundas são particularmente eficazes em capturar hierarquias linguísticas, enquanto redes largas podem processar múltiplas características linguísticas simultaneamente, como sintaxe e semântica. A escolha entre largura e profundidade deve considerar o trade-off entre a capacidade de representação e a eficiência computacional, bem como a disponibilidade de dados e recursos de computação.

#### Shortcut Connections (residual, highway networks)

As conexões de atalho, presentes em arquiteturas como **Redes Residuals** (ResNets) e **Highway Networks**, permitem a passagem direta de informações entre camadas não adjacentes. Teoricamente, essas conexões resolvem o problema do desaparecimento de gradientes em redes profundas, facilitando a propagação de sinais de erro durante o backpropagation. As ResNets introduzem blocos residuais onde a saída de uma camada é adicionada à saída de uma camada anterior, formulada como $h^{(l)} = f(h^{(l-1)}) + h^{(l-1)}$, onde $f$ é uma transformação não linear.

Em **Highway Networks**, as conexões de atalho são moduladas por portas de transporte e transformação, definidas como:
$$
h^{(l)} = T^{(l)} \circ f(h^{(l-1)}) + (1 - T^{(l)}) \circ h^{(l-1)}
$$
onde $T^{(l)}$ são as portas de transporte que controlam o fluxo de informações. Estas portas são aprendidas durante o treinamento, permitindo que a rede decida quanto da informação deve ser transmitida diretamente versus transformada.

Em NLP, essas conexões são cruciais para treinar modelos profundos que capturam relações contextuais em grandes corpora de texto, melhorando a fluidez e a coerência das representações linguísticas. Elas facilitam a aprendizagem de representações profundas sem a degradação da performance que ocorre em redes sem conexões de atalho.

#### Output Layers and Loss Functions (softmax, margin loss)

A camada de saída e a função de perda são componentes críticos que definem o objetivo de aprendizagem da rede. A função **softmax**, combinada com a perda de log-verossimilhança negativa ($\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$), é amplamente utilizada para tarefas de classificação multiclasse em NLP, como a predição de palavras em modelos de linguagem. A softmax normaliza as ativações da camada de saída em uma distribuição de probabilidade, permitindo que a rede interprete as saídas como probabilidades condicionais.

A **perda de margem** ($\mathcal{L} = \sum_{i} \max(0, m - y_i \cdot \hat{y}_i)$), onde $m$ é a margem desejada, impõe uma separação explícita entre as classes, melhorando a discriminação entre categorias semânticas. Esta função de perda é utilizada em tarefas de classificação de texto para assegurar que as predições para diferentes classes estejam suficientemente separadas no espaço de representação.

Teoricamente, a escolha da função de perda influencia a geometria do espaço de parâmetros e a dinâmica do treinamento, afetando a convergência e a generalização do modelo. Em NLP, a combinação de funções de ativação adequadas e funções de perda específicas permite a modelagem eficiente de distribuições linguísticas complexas e a separação clara entre diferentes categorias semânticas.

#### Input Representations (bag-of-words, word embeddings, lookup layers)

As representações de entrada são fundamentais para a eficácia dos modelos de NLP. A abordagem **bag-of-words** (BoW), embora simples, representa textos como vetores de contagem de palavras, perdendo a ordem e as relações semânticas entre palavras, limitando a capacidade de captura de dependências contextuais. Matemáticamente, um documento $d$ é representado como um vetor $\mathbf{x} \in \mathbb{R}^V$, onde $V$ é o vocabulário e cada elemento $x_i$ representa a frequência da palavra $i$ em $d$.

Os **embeddings de palavras**, como **Word2Vec** e **GloVe**, mapeiam palavras em espaços vetoriais contínuos, preservando similaridades semânticas e sintáticas através de propriedades geométricas dos vetores. Estes embeddings são aprendidos através de modelos preditivos ou baseados em co-ocorrência, onde a proximidade vetorial reflete a similaridade semântica. Matemáticamente, um embedding $\mathbf{w}_i$ para a palavra $i$ é um vetor em $\mathbb{R}^d$ onde $d$ é a dimensionalidade do espaço de embedding.

As **camadas de lookup** facilitam a integração eficiente dos embeddings na rede, permitindo a aprendizagem conjunta das representações e dos parâmetros do modelo. Estas camadas operam como tabelas de hash que mapeiam índices de palavras para seus vetores de embedding correspondentes, permitindo a rápida recuperação e atualização durante o treinamento.

Teoricamente, a qualidade das representações de entrada determina a capacidade da rede de modelar nuances linguísticas e de capturar relações semânticas complexas. Embeddings bem treinados proporcionam uma base rica e contínua para a aprendizagem de representações mais abstratas nas camadas ocultas, melhorando a performance em tarefas de NLP que requerem compreensão profunda da linguagem.

### 3.3 Learning Neural Networks

#### Stochastic Gradient Descent (SGD)

O **SGD** é um método de otimização iterativo que atualiza os parâmetros da rede com base em gradientes calculados a partir de mini-batches de dados. Matemáticamente, a atualização dos pesos $\theta$ em cada iteração $t$ é dada por:
$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t; x_t, y_t)
$$
onde $\eta$ é a taxa de aprendizado, $\mathcal{L}$ é a função de perda, e $(x_t, y_t)$ é um mini-batch de dados.

Teoricamente, o SGD converge para um mínimo global em funções convexas, mas em redes neurais, onde as funções objetivo são altamente não convexas, o SGD navega em paisagens complexas de múltiplos mínimos locais e pontos de sela. Em NLP, o SGD é essencial para treinar modelos em grandes conjuntos de dados textuais, onde a eficiência computacional e a capacidade de generalização são críticas.

#### Gradients and Backpropagation

O cálculo dos gradientes é realizado através do algoritmo de **backpropagation**, que aplica a regra da cadeia para propagar os erros das camadas de saída para as camadas de entrada. Matemáticamente, para cada peso $w$ na rede, o gradiente da função de perda $\mathcal{L}$ é calculado como:
$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \dots \frac{\partial h^{(l+1)}}{\partial h^{(l)}} \frac{\partial h^{(l)}}{\partial w}
$$
onde $h^{(l)}$ é a ativação na camada $l$.

Teoricamente, backpropagation permite a otimização eficiente dos parâmetros da rede, mesmo em arquiteturas profundas, facilitando a minimização da função de perda através de ajustes iterativos dos pesos baseados nos gradientes calculados. Em NLP, backpropagation é crucial para ajustar os pesos das redes neurais baseadas em sequências de texto, garantindo que os modelos aprendam representações semânticas e sintáticas precisas.

#### Computation Graphs

Os **grafos de computação** são estruturas que representam as operações matemáticas realizadas pela rede neural, facilitando a visualização e a implementação de processos de diferenciação automática. Matemáticamente, um grafo de computação $G = (V, E)$ consiste em vértices $V$ representando operações e arestas $E$ representando fluxos de dados entre essas operações.

Teoricamente, os grafos permitem a decomposição de cálculos complexos em operações mais simples e reutilizáveis, otimizando a eficiência computacional. Em NLP, grafos de computação suportam operações sequenciais e hierárquicas necessárias para processar textos de forma eficiente e modular, permitindo a implementação de arquiteturas complexas como redes convolucionais e recorrentes de maneira estruturada e escalável.

#### Automatic Differentiation

A **diferenciação automática** é uma técnica que computa os gradientes de forma eficiente e precisa, sem a necessidade de derivação manual das funções de perda. Existem dois modos principais de diferenciação automática: **forward mode** e **reverse mode**. O **reverse mode** é particularmente eficiente para funções com muitas variáveis de entrada e poucas variáveis de saída, o que é comum em redes neurais.

Teoricamente, a diferenciação automática permite a obtenção de gradientes exatos para qualquer arquitetura de rede, independentemente da complexidade. Em NLP, frameworks como TensorFlow e PyTorch utilizam diferenciação automática para facilitar o desenvolvimento e o treinamento de modelos complexos, acelerando o processo de implementação e garantindo a precisão dos gradientes utilizados na otimização.

#### Regularization (weight decay)

A **regularização L2**, ou **weight decay**, adiciona uma penalidade proporcional ao quadrado dos pesos à função de perda, formulada como:
$$
\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_{i} w_i^2
$$
onde $\lambda$ é o coeficiente de regularização.

Teoricamente, a regularização controla a complexidade do modelo, prevenindo o overfitting ao incentivar soluções com pesos menores. Em NLP, onde os modelos frequentemente lidam com dados de alta dimensionalidade e variabilidade linguística, a regularização é fundamental para garantir que os modelos aprendam representações robustas e generalizáveis, evitando que se ajustem excessivamente aos dados de treinamento.

#### Dropout

O **dropout** é uma técnica de regularização que, durante o treinamento, desativa aleatoriamente uma fração dos neurônios, impedindo a co-adaptação excessiva entre eles. Matemáticamente, para cada camada com ativação $h$, aplicamos uma máscara binária $m$ onde cada elemento $m_i$ é 0 com probabilidade $p$ e 1 com probabilidade $1-p$, resultando em $h' = m \odot h$.

Teoricamente, o dropout aproxima o treinamento de uma média de modelos distintos, reduzindo a variância e prevenindo o overfitting. Em NLP, o dropout promove a criação de representações mais robustas e generalizáveis, essenciais para tarefas como tradução automática e geração de texto, onde a diversidade e a flexibilidade das representações são cruciais para lidar com a variedade linguística.

#### Feature Noising

O **noising de características** envolve a adição de ruído aos inputs ou às ativações ocultas, atuando como uma forma de regularização. Matemáticamente, isso pode ser formulado como:
$$
h' = h + \epsilon
$$
onde $\epsilon$ é um ruído aleatório, tipicamente gaussian $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

Teoricamente, o noising promove a invariância a pequenas perturbações nos dados de entrada, melhorando a robustez do modelo. Em NLP, o noising pode ajudar a rede a lidar com variações linguísticas, erros tipográficos e ambiguidade semântica, promovendo a aprendizagem de representações mais resilientes e generalizáveis. Além do dropout, outras formas de noising, como a adição de ruído gaussiano ou de ruído de dropout em camadas intermediárias, podem ser utilizadas para aumentar a capacidade de generalização dos modelos.

#### Learning Theory (convexity, local optima, saddle points)

As funções objetivo em redes neurais são altamente não convexas, apresentando múltiplos mínimos locais e pontos de sela. Matemáticamente, considere uma função de perda $\mathcal{L}(\theta)$ onde $\theta$ são os parâmetros da rede. A não convexidade implica que $\mathcal{L}(\theta)$ pode ter muitos pontos onde o gradiente $\nabla_{\theta} \mathcal{L}(\theta) = 0$, incluindo mínimos locais e pontos de sela.

Teoricamente, isso complica a busca por soluções ótimas, pois os algoritmos de otimização podem ficar presos em mínimos locais subótimos ou navegar por regiões de pontos de sela. Em NLP, a alta dimensionalidade e a complexidade das representações linguísticas aumentam a presença desses desafios, exigindo estratégias de otimização robustas, como inicialização adequada dos pesos e uso de algoritmos adaptativos, para alcançar boas soluções de aprendizagem.

Além disso, estudos teóricos sugerem que, em alta dimensionalidade, a maioria dos pontos de sela possui gradientes suficientemente grandes para escapar, o que implica que os algoritmos de otimização como SGD podem evitar se prender em pontos de sela, concentrando-se na convergência para ótimos locais de boa qualidade.

#### Generalization Guarantees

Apesar da capacidade das redes neurais de memorizar dados de treinamento, entender teoricamente como elas generalizam para dados não vistos permanece um desafio. Conceitos como **capacidade do modelo**, **margem de classificação** e **regularização** desempenham papéis cruciais na generalização. Matemáticamente, a capacidade do modelo pode ser quantificada por métricas como a **VC dimension** ou **Rademacher complexity**, que medem a habilidade do modelo de se ajustar a diferentes conjuntos de dados.

Em NLP, onde os modelos são expostos a uma vasta diversidade de dados textuais, a capacidade de generalização é vital para assegurar que os modelos compreendam e processem corretamente novos textos, mantendo a coerência e a precisão sem depender excessivamente dos dados de treinamento. Teorias emergentes, como a **teoria do double descent** e a **teoria das redes profundas** que utilizam regularização implícita (como o dropout), estão sendo desenvolvidas para explicar empiricamente a capacidade das redes neurais de generalizar mesmo em regimes de sobreparametrização.

#### Training Tricks (initialization, gradient clipping, batch/layer normalization)

Truques de treinamento são técnicas práticas que melhoram a eficiência e a eficácia do treinamento de redes neurais. 

- **Inicialização de Pesos**: Estratégias como a **inicialização de He** (para ReLU) ou **Xavier** (para tanh) são usadas para manter a variância das ativações constante através das camadas, prevenindo problemas como o desaparecimento ou explosão de gradientes. Matemáticamente, a inicialização de He define $\sigma^2 = \frac{2}{n_{in}}$, onde $n_{in}$ é o número de entradas da camada, enquanto a inicialização de Xavier define $\sigma^2 = \frac{2}{n_{in} + n_{out}}$.

- **Gradient Clipping**: Limita o valor dos gradientes para evitar atualizações excessivamente grandes que podem desestabilizar o treinamento. Matemáticamente, isso pode ser formulado como:
$$
g = \frac{g}{\max(1, \frac{\|g\|}{c})}
$$
onde $g$ é o gradiente e $c$ é o limite.

- **Batch Normalization e Layer Normalization**: Técnicas que normalizam as ativações para cada mini-batch ou camada, respectivamente, estabilizando e acelerando a convergência do treinamento. Matemáticamente, a batch normalization aplica a normalização:
$$
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$
onde $\mu_{\mathcal{B}}$ e $\sigma_{\mathcal{B}}^2$ são a média e a variância do mini-batch.

Em NLP, essas técnicas são essenciais para treinar modelos profundos e complexos de forma estável, especialmente quando se lida com sequências longas e representações de alta dimensionalidade. A normalização em lote, por exemplo, facilita a aprendizagem de representações consistentes ao longo de diferentes mini-batches, enquanto o gradient clipping evita que atualizações abruptas desestabilizem o treinamento de modelos que processam sequências de texto extensas.

#### Online Optimization (AdaGrad, AdaDelta, Adam, early stopping)

Algoritmos de otimização online ajustam adaptativamente as taxas de aprendizado com base nas atualizações anteriores dos gradientes.

- **AdaGrad**: Adapta a taxa de aprendizado para cada parâmetro individualmente, acumulando os quadrados dos gradientes passados:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_{\theta} \mathcal{L}(\theta_t)
$$
onde $G_t$ é a soma acumulada dos quadrados dos gradientes até a iteração $t$.

- **AdaDelta**: Estende AdaGrad, evitando a diminuição contínua das taxas de aprendizado através do uso de médias móveis exponenciais dos quadrados dos gradientes e das atualizações:
$$
\theta_{t+1} = \theta_t - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_{t} + \epsilon}} \odot g_t
$$

- **Adam**: Combina os benefícios de AdaGrad e RMSProp, mantendo estimativas dos primeiros ($m_t$) e segundos ($v_t$) momentos dos gradientes:
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- **Early Stopping**: Técnica para prevenir o overfitting interrompendo o treinamento antes que o modelo comece a se ajustar excessivamente aos dados de treinamento, baseado no desempenho em um conjunto de validação.

Teoricamente, esses algoritmos proporcionam uma convergência mais rápida e robusta em comparação com o SGD padrão, ajustando dinamicamente as taxas de aprendizado para cada parâmetro e melhorando a eficiência do treinamento. Em NLP, onde os modelos são frequentemente treinados em grandes corpora de texto, esses métodos permitem a otimização eficiente e eficaz, adaptando-se às características específicas dos dados textuais e evitando armadilhas de otimização que podem levar a soluções subótimas.

### 3.4 Convolutional Neural Networks

#### Convolutional Layers

As **camadas convolucionais** aplicam filtros (kernels) que convoluem sobre a entrada para extrair características locais. Matemáticamente, uma convolução 1D em NLP pode ser definida como:
$$
(f * x)(t) = \sum_{k=0}^{K-1} f(k) \cdot x(t + k)
$$
onde $f$ é o filtro de comprimento $K$, e $x$ é a sequência de entrada.

Teoricamente, a convolução é uma operação linear que captura padrões invariantes a pequenas translações, essencial para a detecção de estruturas linguísticas como n-gramas e dependências locais entre palavras. Em NLP, as camadas convolucionais são utilizadas para capturar dependências sintáticas e semânticas locais em sequências de texto, permitindo que a rede identifique e represente padrões linguísticos fundamentais de forma eficiente.

#### Filters and Feature Maps

Os **filtros convolucionais** aprendem a detectar características específicas das entradas, produzindo **mapas de características** que representam a presença e a localização desses padrões. Matemáticamente, cada filtro $f_i$ gera um feature map $m_i$ através da operação de convolução:
$$
m_i(t) = (f_i * x)(t)
$$

Teoricamente, a capacidade dos filtros de aprender representações invariantes a transformações locais melhora a robustez e a expressividade das redes convolucionais. Em NLP, os feature maps capturam construções gramaticais, entidades nomeadas e outras estruturas linguísticas, permitindo que a rede construa representações hierárquicas e ricas das sequências de texto. Cada filtro pode ser interpretado como um detector de padrões específicos, como bigramas, trigramas ou outras combinações de palavras que são relevantes para a tarefa de classificação.

#### Wide vs. Narrow Convolution

A distinção entre **convoluções largas** (wide convolution) e **estreitas** (narrow convolution) está relacionada ao uso de padding na entrada. Convoluções largas mantêm o tamanho da saída igual ao da entrada através do preenchimento (padding), preservando informações nas bordas das sequências de texto. Matemáticamente, se a entrada tem comprimento $T$, o padding $P$ é escolhido tal que a saída também tenha comprimento $T$.

Convoluções estreitas reduzem o tamanho da saída, focando apenas nos pontos onde os filtros se sobrepõem completamente com a entrada, sem padding. Isso resulta em uma saída de comprimento $T - K + 1$, onde $K$ é o tamanho do filtro.

Teoricamente, a escolha entre wide e narrow convolution afeta a preservação de contexto nas extremidades das sequências, influenciando a capacidade da rede de capturar informações contextuais completas e coerentes em NLP. Convoluções largas são preferidas quando é necessário preservar a estrutura completa das sequências de texto, enquanto convoluções estreitas podem ser utilizadas para focar em padrões específicos sem considerar o contexto das bordas.

#### Pooling (max-pooling, average-pooling)

As operações de **pooling** agregam informações dos feature maps, reduzindo a dimensionalidade e extraindo características invariantes a pequenas variações. Matemáticamente, para uma janela de pooling de tamanho $S$, as operações são definidas como:

- **Max-Pooling**:
$$
m'(t) = \max_{s=1,\dots,S} m(t + s)
$$
- **Average-Pooling**:
$$
m'(t) = \frac{1}{S} \sum_{s=1}^{S} m(t + s)
$$

Teoricamente, o pooling reduz a complexidade computacional e o risco de overfitting, além de promover a invariância a deslocamentos e distorções. Em NLP, o pooling ajuda a resumir informações relevantes de diferentes regiões do texto, facilitando a captura de características invariantes a variações de tamanho e posição nas sequências textuais. Além disso, o pooling contribui para a estabilidade das representações, agregando informações de forma robusta contra variações locais nos dados de entrada.

#### Multi-Layer Convolutional Networks

Empilhar múltiplas camadas convolucionais permite a aprendizagem de representações hierárquicas e abstratas. Matemáticamente, cada camada convolucional $l$ aplica múltiplos filtros $f_i^{(l)}$ para gerar múltiplos feature maps $m_i^{(l)}$. A saída de uma camada convolucional $l$ serve como entrada para a camada $l+1$, permitindo a composição de características de nível superior.

Teoricamente, camadas mais profundas capturam características de nível superior, combinando informações de camadas anteriores para formar representações cada vez mais complexas e informativas. Em NLP, redes convolucionais profundas podem modelar desde padrões locais simples, como bigramas, até estruturas sintáticas e semânticas complexas, proporcionando uma compreensão profunda e multifacetada das sequências de texto. A hierarquia de características permite que a rede capture tanto dependências locais quanto globais dentro das sequências linguísticas.

#### Dilated Convolution and Multiscale Representations

A **convolução dilatada** (dilated convolution) introduz espaçamentos entre os elementos do kernel, aumentando o campo receptivo sem aumentar o número de parâmetros. Matemáticamente, uma convolução dilatada com fator de dilatação $d$ é definida como:
$$
(f * x)_t = \sum_{k=0}^{K-1} f_k \cdot x_{t + d \cdot k}
$$

Teoricamente, a convolução dilatada permite a captura de contextos maiores e a criação de representações multiescalares, essenciais para modelar dependências de longo alcance em sequências de texto. Em NLP, a convolução dilatada facilita a detecção de relações semânticas distantes e a integração de informações contextuais amplas, melhorando a capacidade da rede de entender contextos complexos e variados. Isso é particularmente útil em tarefas que requerem a compreensão de dependências sintáticas e semânticas que se estendem por longas sequências, como a co-referência e a interpretação de sentenças complexas.

#### Backpropagation through Max-Pooling

O cálculo dos gradientes através da operação de **max-pooling** envolve a identificação dos elementos máximos que contribuíram para a saída, propagando os gradientes apenas para esses elementos. Matemáticamente, se $m'(t) = \max_{s=1,\dots,S} m(t + s)$, então o gradiente $\frac{\partial \mathcal{L}}{\partial m(t + s^*)}$ é propagado onde $s^* = \arg\max_{s=1,\dots,S} m(t + s)$.

Teoricamente, isso assegura que os filtros convolucionais aprendam a identificar e fortalecer os padrões mais relevantes nas entradas. Em NLP, a retropropagação através do max-pooling garante que a rede aprenda a focar em informações linguísticas cruciais dentro das sequências de texto, melhorando a capacidade de detecção e representação de padrões linguísticos significativos. Isso resulta em representações mais discriminativas e robustas, capazes de capturar aspectos essenciais da linguagem natural de maneira eficiente.

