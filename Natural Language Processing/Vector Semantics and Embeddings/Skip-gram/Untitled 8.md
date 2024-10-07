# Embeddings Alvo e Contexto: Entendendo as Matrizes de Parâmetros do Modelo Skip-gram

![image-20241004115536870](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20241004115536870.png)

## Introdução

Os **embeddings alvo e de contexto** são componentes fundamentais do modelo Skip-gram, uma arquitetura neural projetada para aprender representações vetoriais densas de palavras a partir de grandes corpora de texto [1]. Esse modelo, parte da família word2vec, revolucionou o campo do processamento de linguagem natural (PLN) ao introduzir uma maneira eficiente e eficaz de capturar relações semânticas e sintáticas entre palavras em um espaço vetorial contínuo [2].

O conceito central por trás do modelo Skip-gram é a ideia de que palavras que ocorrem em contextos similares tendem a ter significados semelhantes. Esta noção, conhecida como hipótese distribucional, é operacionalizada através do uso de duas matrizes de embeddings distintas: a matriz de embeddings alvo (W) e a matriz de embeddings de contexto (C) [3].

## Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Embedding Alvo**        | Vetor que representa uma palavra quando ela é o foco da predição. Armazenado na matriz W. [4] |
| **Embedding de Contexto** | Vetor que representa uma palavra quando ela aparece no contexto de outra palavra. Armazenado na matriz C. [5] |
| **Modelo Skip-gram**      | Arquitetura neural que aprende a prever palavras de contexto dado um alvo, utilizando as matrizes W e C como parâmetros. [6] |

> ⚠️ **Nota Importante**: As matrizes W e C são inicializadas aleatoriamente e otimizadas durante o treinamento para maximizar a probabilidade de palavras de contexto corretas dado um alvo. [7]

> ❗ **Ponto de Atenção**: Embora W e C sejam matrizes distintas, após o treinamento, muitas implementações somam ou concatenam os vetores correspondentes para obter o embedding final de uma palavra. [8]

> ✔️ **Destaque**: A separação entre embeddings alvo e de contexto permite ao modelo capturar nuances sutis nas relações entre palavras, contribuindo para a riqueza semântica das representações aprendidas. [9]

### Arquitetura do Modelo Skip-gram

<imagem: Um diagrama detalhado da arquitetura Skip-gram, mostrando a palavra alvo de entrada, a camada de projeção (embedding alvo), a camada de saída (embeddings de contexto), e as conexões entre elas. Inclua setas indicando o fluxo de informação e rótulos para W e C.>

O modelo Skip-gram é fundamentado na ideia de prever as palavras de contexto dado um alvo. A arquitetura é composta por:

1. **Camada de Entrada**: Representa a palavra alvo como um vetor one-hot.
2. **Matriz de Embedding Alvo (W)**: Projeta o vetor one-hot em um espaço denso.
3. **Matriz de Embedding de Contexto (C)**: Utilizada para calcular a probabilidade de palavras de contexto.
4. **Camada de Saída**: Produz probabilidades para cada palavra do vocabulário ser um contexto. [10]

A função objetivo do Skip-gram visa maximizar:

$$
\mathcal{L} = \sum_{(w,c) \in \mathcal{D}} \log P(c|w)
$$

Onde $(w,c)$ são pares de palavras alvo e contexto no conjunto de dados $\mathcal{D}$, e $P(c|w)$ é modelada usando a função softmax:

$$
P(c|w) = \frac{\exp(c_c \cdot w_w)}{\sum_{c' \in \mathcal{V}} \exp(c_{c'} \cdot w_w)}
$$

Aqui, $c_c$ é o vetor de contexto para a palavra $c$, $w_w$ é o vetor alvo para a palavra $w$, e $\mathcal{V}$ é o vocabulário. [11]

#### Perguntas Teóricas

1. Derive a atualização do gradiente para os vetores $w_w$ e $c_c$ na otimização do modelo Skip-gram usando descida de gradiente estocástica.
2. Como a escolha da dimensionalidade dos embeddings afeta a capacidade do modelo de capturar relações semânticas? Forneça uma análise teórica.
3. Demonstre matematicamente por que a separação entre embeddings alvo e de contexto pode levar a representações mais ricas do que usar uma única matriz de embedding.

### Treinamento e Otimização

O treinamento do modelo Skip-gram envolve a otimização das matrizes W e C para maximizar a probabilidade de observar as palavras de contexto corretas dado um alvo. Este processo é computacionalmente intensivo devido ao cálculo do softmax sobre todo o vocabulário. Para mitigar isso, técnicas de aproximação são frequentemente empregadas:

1. **Negative Sampling**: Ao invés de atualizar todos os vetores de contexto, apenas um subconjunto de "amostras negativas" é usado, junto com a palavra de contexto positiva. [12]

2. **Hierarchical Softmax**: Utiliza uma estrutura de árvore para representar o vocabulário, reduzindo a complexidade computacional de $O(|V|)$ para $O(\log |V|)$. [13]

A função objetivo com Negative Sampling torna-se:

$$
\mathcal{L} = \log \sigma(c_{pos} \cdot w) + \sum_{i=1}^k \mathbb{E}_{c_{neg_i} \sim P_n(w)}[\log \sigma(-c_{neg_i} \cdot w)]
$$

Onde $\sigma$ é a função sigmóide, $c_{pos}$ é o vetor de contexto positivo, $c_{neg_i}$ são os vetores de contexto negativos, e $P_n(w)$ é a distribuição de amostragem negativa. [14]

> 💡 **Insight**: A separação entre W e C permite que o modelo capture assimetrias nas relações entre palavras, como a diferença entre "é um tipo de" e "tem um tipo". [15]

#### Análise Teórica da Convergência

A convergência do modelo Skip-gram com Negative Sampling pode ser analisada através da teoria de otimização estocástica. Considerando a função objetivo:

$$
J(\theta) = \mathbb{E}_{(w,c) \sim \mathcal{D}}[f_w(c;\theta)]
$$

Onde $\theta$ representa os parâmetros do modelo (W e C combinados), e $f_w(c;\theta)$ é a função de perda para um par $(w,c)$. A atualização do gradiente estocástico é dada por:

$$
\theta_{t+1} = \theta_t - \eta_t \nabla f_{w_t}(c_t;\theta_t)
$$

Onde $\eta_t$ é a taxa de aprendizado no tempo $t$. Sob certas condições de regularidade e escolha apropriada de $\eta_t$, pode-se provar a convergência quase certa para um ponto estacionário. [16]

#### Perguntas Teóricas

1. Derive a complexidade computacional do treinamento do Skip-gram com e sem Negative Sampling. Como isso afeta a escalabilidade do modelo para vocabulários muito grandes?
2. Analise teoricamente o impacto do número de amostras negativas na qualidade dos embeddings aprendidos. Existe um trade-off entre eficiência computacional e qualidade das representações?
3. Proponha e justifique matematicamente uma estratégia de inicialização para as matrizes W e C que poderia acelerar a convergência do modelo.

### Propriedades Semânticas dos Embeddings

Os embeddings alvo e de contexto capturados pelas matrizes W e C exibem propriedades semânticas interessantes:

1. **Similaridade Cossenoidal**: A similaridade entre palavras pode ser medida pelo cosseno entre seus vetores:

   $$
   \text{sim}(w_1, w_2) = \frac{w_1 \cdot w_2}{\|w_1\| \|w_2\|}
   $$

   Esta medida captura efetivamente relações semânticas e sintáticas entre palavras. [17]

2. **Analogias**: Os embeddings podem resolver analogias da forma "a está para b como c está para d" através de operações vetoriais:

   $$
   \arg\max_{x} \cos(x, w_b - w_a + w_c)
   $$

   Onde $w_a$, $w_b$, $w_c$ são os vetores das palavras a, b, c respectivamente. [18]

3. **Composicionalidade**: Embeddings de frases ou documentos podem ser obtidos através de operações sobre embeddings de palavras individuais, como média ou soma ponderada. [19]

> ⚠️ **Nota Importante**: A capacidade dos embeddings de capturar analogias e relações semânticas complexas emerge das regularidades estatísticas do corpus de treinamento, não de conhecimento linguístico explicitamente codificado. [20]

#### Análise Teórica da Informação Mútua

A separação entre embeddings alvo e de contexto pode ser analisada através da lente da teoria da informação. Definindo a informação mútua pontual (PMI) entre palavras alvo e contexto:

$$
\text{PMI}(w,c) = \log \frac{P(w,c)}{P(w)P(c)}
$$

Pode-se mostrar que, sob certas condições, o produto escalar dos embeddings alvo e de contexto aproxima o PMI:

$$
w_w \cdot c_c \approx \text{PMI}(w,c) - \log k
$$

Onde $k$ é o número de amostras negativas. Esta relação fornece uma interpretação teórica para a semântica capturada pelos embeddings. [21]

#### Perguntas Teóricas

1. Derive a relação entre o produto escalar dos embeddings e o PMI. Quais são as implicações desta relação para a interpretabilidade dos embeddings?
2. Como a dimensionalidade dos embeddings afeta a capacidade do modelo de preservar a informação mútua entre palavras? Forneça uma análise teórica.
3. Proponha e justifique matematicamente uma modificação na função objetivo do Skip-gram que poderia melhorar a captura de relações semânticas assimétricas.

### Discussão Crítica

Apesar do sucesso dos embeddings alvo e de contexto no modelo Skip-gram, existem limitações e desafios importantes a serem considerados:

1. **Polissemia**: O modelo atribui um único vetor para cada palavra, não capturando adequadamente múltiplos sentidos. [22]

2. **Viés e Estereótipos**: Os embeddings podem aprender e amplificar vieses presentes nos dados de treinamento. [23]

3. **Instabilidade**: Diferentes execuções do treinamento podem resultar em embeddings significativamente diferentes devido à natureza estocástica do processo. [24]

4. **Interpretabilidade**: As dimensões individuais dos embeddings não têm interpretações semânticas claras, dificultando a análise linguística detalhada. [25]

Pesquisas recentes têm abordado essas limitações através de:

- Modelos de embeddings contextuais (e.g., BERT, GPT) que produzem representações dinâmicas baseadas no contexto. [26]
- Técnicas de debiasing para mitigar vieses aprendidos. [27]
- Métodos de pós-processamento para melhorar a estabilidade e interpretabilidade dos embeddings. [28]

> 💡 **Perspectiva Futura**: A integração de conhecimento linguístico explícito e a incorporação de estruturas hierárquicas nos modelos de embeddings são direções promissoras para superar as limitações atuais. [29]

## Conclusão

Os embeddings alvo e de contexto, representados pelas matrizes W e C no modelo Skip-gram, constituem uma abordagem poderosa e flexível para a aprendizagem de representações distribuídas de palavras. A separação entre essas duas matrizes permite ao modelo capturar nuances sutis nas relações semânticas e sintáticas entre palavras, resultando em representações ricas e úteis para uma ampla gama de tarefas de PLN.

A fundamentação teórica desses embeddings na teoria da informação e na otimização estocástica fornece insights valiosos sobre sua capacidade de capturar estruturas linguísticas complexas. No entanto, desafios como a representação de polissemia, mitigação de vieses e melhoria da interpretabilidade permanecem áreas ativas de pesquisa.

À medida que o campo avança, a integração de abordagens neurais com conhecimento linguístico estruturado promete levar a representações ainda mais sofisticadas e úteis, potencialmente superando as limitações atuais dos embeddings estáticos. [30]

## Perguntas Teóricas Avançadas

1. Desenvolva uma prova formal da equivalência assintótica entre a otimização do Skip-gram com Negative Sampling e a fatoração implícita da matriz de PMI. Quais são as implicações desta equivalência para a interpretação semântica dos embeddings?

2. Proponha e analise matematicamente uma extensão do modelo Skip-gram que incorpore informações sintáticas explícitas (e.g., árvores de dependência) na função objetivo. Como isso afetaria a geometria do espaço de embeddings resultante?

3. Derive uma bound teórica para o erro de generalização do modelo Skip-gram em termos da dimensionalidade dos embeddings e do tamanho do corpus de treinamento. Como esta bound se compara com bounds similares para outros modelos de linguagem?

4. Desenvolva um framework teórico para analisar a estabilidade dos embeddings aprendidos em relação a perturbações no corpus de treinamento. Que propriedades do corpus e do algoritmo de treinamento afetam mais significativamente esta estabilidade?

5. Proponha e justifique matematicamente uma medida de "informatividade" para dimensões individuais dos embeddings. Como esta medida poderia ser utilizada para compressão ou interpretação dos embeddings?

## Referências

[1] "O modelo Skip-gram, parte da família word2vec, revolucionou o campo do processamento de linguagem natural (PLN) ao introduzir uma maneira eficiente e eficaz de capturar relações semânticas e sintáticas entre palavras em um espaço vetorial contínuo" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[2] "The skip-gram algorithm is one of two algorithms in a software package called word2vec, and so sometimes the algorithm is loosely referred to as word2vec (Mikolov et al. 2013a, Mikolov et al. 2013b). The word2vec methods are fast, efficient to train, and easily available online." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[3] "The skip-gram model stores two embeddings for each word: one for the word as a target and another for the word considered as context." *(Trecho de Vector Semantics and Embeddings.pdf.md)*





