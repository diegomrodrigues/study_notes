# Probabilidade de Palavra de Contexto: $P(+|w,c)$

<imagem: Um gráfico 3D mostrando a distribuição de probabilidade $P(+|w,c)$ em função de vetores de palavras $w$ e $c$ no espaço de embeddings, com cores representando valores de probabilidade de 0 a 1>

## Introdução

A probabilidade de palavra de contexto, denotada como $P(+|w,c)$, é um conceito fundamental no modelo skip-gram, uma técnica avançada de aprendizado de máquina para a criação de embeddings de palavras. Este conceito está no cerne da tarefa de classificação binária que o skip-gram utiliza para aprender representações vetoriais de palavras [1]. A importância deste conceito reside na sua capacidade de capturar relações semânticas entre palavras em um corpus de texto, permitindo a criação de representações vetoriais densas e significativas [2].

## Conceitos Fundamentais

| Conceito      | Explicação                                                   |
| ------------- | ------------------------------------------------------------ |
| **Skip-gram** | Modelo de aprendizado de máquina que aprende embeddings de palavras prevendo palavras de contexto dado uma palavra-alvo [3]. |
| **Embedding** | Representação vetorial densa de uma palavra em um espaço multidimensional [4]. |
| **Contexto**  | Palavras que ocorrem próximas à palavra-alvo em um texto [5]. |

> ⚠️ **Nota Importante**: A probabilidade $P(+|w,c)$ é fundamental para o treinamento do modelo skip-gram, pois quantifica a probabilidade de uma palavra $c$ ser um verdadeiro contexto para uma palavra-alvo $w$ [6].

> ❗ **Ponto de Atenção**: O modelo skip-gram utiliza uma tarefa de classificação binária para aprender embeddings, diferentemente de modelos baseados em contagem como tf-idf ou PPMI [7].

> ✔️ **Destaque**: A formulação matemática de $P(+|w,c)$ permite a otimização eficiente do modelo através de técnicas como amostragem negativa [8].

## Formulação Matemática

<imagem: Diagrama mostrando a arquitetura do modelo skip-gram, com vetores de entrada e saída, e a função sigmoid aplicada ao produto escalar>

A probabilidade $P(+|w,c)$ é definida matematicamente como:

$$
P(+|w,c) = \sigma(c \cdot w) = \frac{1}{1 + \exp(-c \cdot w)}
$$

Onde:
- $w$ é o vetor de embedding da palavra-alvo
- $c$ é o vetor de embedding da palavra de contexto
- $\sigma$ é a função sigmoid
- $\cdot$ denota o produto escalar entre vetores [9]

Esta formulação é derivada da intuição de que palavras similares devem ter vetores de embedding com alto produto escalar [10]. A função sigmoid é utilizada para mapear o produto escalar para o intervalo [0, 1], interpretável como uma probabilidade [11].

### Prova da Equivalência entre Sigmoid e Softmax Binário

Podemos demonstrar que a função sigmoid usada em $P(+|w,c)$ é equivalente a um softmax binário:

$$
\begin{align*}
P(+|w,c) &= \frac{\exp(c \cdot w)}{\exp(c \cdot w) + \exp(0)} \\
&= \frac{1}{1 + \exp(-c \cdot w)} \\
&= \sigma(c \cdot w)
\end{align*}
$$

Esta equivalência justifica o uso da sigmoid como uma escolha computacionalmente eficiente para a tarefa de classificação binária no skip-gram [12].

### Perguntas Teóricas

1. Derive a expressão para o gradiente de $P(+|w,c)$ em relação a $w$ e $c$. Como isso é utilizado no processo de aprendizagem do skip-gram?

2. Demonstre matematicamente por que o produto escalar $c \cdot w$ é uma medida adequada de similaridade entre palavras no espaço de embeddings.

3. Como a escolha da função sigmoid afeta a estabilidade numérica do treinamento do modelo? Existem alternativas que poderiam melhorar este aspecto?

## Amostragem Negativa e Função Objetivo

A probabilidade $P(+|w,c)$ é utilizada em conjunto com a técnica de amostragem negativa para formar a função objetivo do skip-gram. A função de perda para um par $(w,c)$ é definida como:

$$
L = -\log\sigma(c \cdot w) - \sum_{i=1}^k \log\sigma(-c_{neg_i} \cdot w)
$$

Onde $c_{neg_i}$ são $k$ amostras negativas (palavras que não são contexto de $w$) [13].

Esta formulação permite o treinamento eficiente do modelo, evitando o cálculo custoso do denominador do softmax completo sobre todo o vocabulário [14].

> 💡 **Insight**: A amostragem negativa transforma o problema de predição multiclasse em várias tarefas de classificação binária, tornando o treinamento mais eficiente e escalável [15].

### Algoritmo de Treinamento

O algoritmo de treinamento do skip-gram com amostragem negativa pode ser resumido nos seguintes passos:

1. Inicializar aleatoriamente os vetores de embedding $w$ e $c$ para todas as palavras.
2. Para cada palavra-alvo $w$ no corpus:
   a. Selecionar palavras de contexto $c$ dentro de uma janela.
   b. Amostrar $k$ palavras negativas $c_{neg}$.
   c. Calcular a perda $L$ usando a equação acima.
   d. Atualizar os vetores $w$ e $c$ usando gradiente descendente:
      $$w_{t+1} = w_t - \eta \frac{\partial L}{\partial w_t}$$
      $$c_{t+1} = c_t - \eta \frac{\partial L}{\partial c_t}$$
   Onde $\eta$ é a taxa de aprendizagem [16].

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target_word, context_word):
        target_emb = self.target_embeddings(target_word)
        context_emb = self.context_embeddings(context_word)
        return torch.sum(target_emb * context_emb, dim=1)

def train_step(model, optimizer, target_word, context_word, negative_samples):
    model.train()
    optimizer.zero_grad()
    
    # Positive sample
    pos_loss = -torch.log(torch.sigmoid(model(target_word, context_word)))
    
    # Negative samples
    neg_loss = -torch.sum(torch.log(torch.sigmoid(-model(target_word, negative_samples))), dim=1)
    
    loss = torch.mean(pos_loss + neg_loss)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Uso do modelo
vocab_size = 10000
embedding_dim = 300
model = SkipGramModel(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters())

# Exemplo de um passo de treinamento
target_word = torch.LongTensor([1])
context_word = torch.LongTensor([2])
negative_samples = torch.LongTensor([3, 4, 5])
loss = train_step(model, optimizer, target_word, context_word, negative_samples)
print(f"Loss: {loss}")
```

Este código implementa o modelo skip-gram usando PyTorch, demonstrando como $P(+|w,c)$ é utilizado na prática para treinar embeddings de palavras [17].

### Perguntas Teóricas

1. Como a escolha do número $k$ de amostras negativas afeta o tradeoff entre velocidade de treinamento e qualidade dos embeddings aprendidos?

2. Derive a expressão para o gradiente da função de perda $L$ em relação aos vetores de embedding $w$ e $c$. Como isso se relaciona com a atualização dos parâmetros no algoritmo de treinamento?

3. Analise teoricamente o impacto da inicialização dos vetores de embedding na convergência do modelo. Existe uma inicialização ótima?

## Propriedades e Implicações Teóricas

A formulação de $P(+|w,c)$ no skip-gram leva a várias propriedades interessantes dos embeddings resultantes:

1. **Captura de Similaridade Semântica**: Palavras com significados similares tendem a ter vetores de embedding próximos no espaço vetorial [18].

2. **Preservação de Analogias**: Os embeddings aprendidos podem capturar relações analógicas, como "rei - homem + mulher ≈ rainha" [19].

3. **Dimensionalidade Reduzida**: Comparado com representações one-hot, os embeddings são muito mais compactos, tipicamente com 50-300 dimensões [20].

4. **Generalização**: Os embeddings podem generalizar para palavras não vistas durante o treinamento, baseando-se em contextos similares [21].

> ⚠️ **Nota Importante**: Apesar dessas propriedades úteis, os embeddings podem também capturar e amplificar vieses presentes nos dados de treinamento, como estereótipos de gênero ou étnicos [22].

### Análise Teórica da Convergência

A convergência do modelo skip-gram pode ser analisada através da teoria de otimização não-convexa. Considerando que a função objetivo é não-convexa devido à sua natureza neural, podemos utilizar resultados recentes em otimização estocástica para garantir a convergência para um mínimo local [23].

Seja $f(w,c)$ a função objetivo do skip-gram. Podemos mostrar que, sob certas condições de regularidade e com uma taxa de aprendizado apropriada $\eta_t$, o algoritmo de gradiente estocástico converge em expectativa:

$$
\mathbb{E}[\|\nabla f(w_t, c_t)\|^2] \leq \frac{C}{\sqrt{T}}
$$

Onde $T$ é o número de iterações e $C$ é uma constante que depende das propriedades da função e da distribuição dos dados [24].

### Perguntas Teóricas

1. Como a dimensionalidade do espaço de embedding afeta a capacidade do modelo de capturar relações semânticas complexas? Existe um limite teórico para a informação que pode ser codificada?

2. Demonstre matematicamente como a propriedade de analogia emerge da estrutura do modelo skip-gram e da função objetivo utilizada.

3. Analise teoricamente o impacto do tamanho do vocabulário na complexidade computacional e na qualidade dos embeddings aprendidos. Existe um tradeoff ótimo?

## Extensões e Variantes

O conceito de $P(+|w,c)$ tem sido estendido e modificado em várias direções:

1. **GloVe**: Utiliza uma abordagem baseada em contagem, mas mantém a ideia de produto escalar entre vetores [25].

2. **FastText**: Estende o skip-gram para incorporar informações de subpalavras, melhorando a representação de palavras raras e fora do vocabulário [26].

3. **BERT**: Utiliza uma arquitetura de transformador para aprender representações contextuais, onde $P(+|w,c)$ é substituída por uma função mais complexa que considera o contexto bidirecionalmente [27].

```python
import torch
import torch.nn as nn

class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ngram_range=(3,6)):
        super(FastTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_range = ngram_range
        
    def forward(self, word, subwords):
        word_emb = self.embedding(word)
        subword_emb = self.embedding(subwords).mean(dim=1)
        return word_emb + subword_emb

# Uso do modelo
vocab_size = 100000
embedding_dim = 300
model = FastTextModel(vocab_size, embedding_dim)

# Exemplo de forward pass
word = torch.LongTensor([42])
subwords = torch.LongTensor([[1, 2, 3], [4, 5, 6]])  # Exemplo de n-gramas
embedding = model(word, subwords)
print(f"Embedding shape: {embedding.shape}")
```

Este código demonstra uma implementação simplificada do modelo FastText, que estende o conceito de $P(+|w,c)$ para incluir informações de subpalavras [28].

## Conclusão

A probabilidade de palavra de contexto $P(+|w,c)$ é um conceito central no modelo skip-gram e tem implicações profundas na aprendizagem de representações de palavras. Sua formulação matemática elegante e eficiente computacionalmente permitiu avanços significativos em várias tarefas de processamento de linguagem natural [29]. Compreender as nuances teóricas e práticas deste conceito é crucial para desenvolver e aplicar modelos de linguagem mais avançados e eficazes [30].

## Perguntas Teóricas Avançadas

1. Derive uma expressão para a complexidade de amostra do modelo skip-gram em termos do tamanho do vocabulário, dimensão do embedding e número de exemplos de treinamento. Como isso se compara com métodos baseados em contagem como LSA?

2. Analise teoricamente o impacto da distribuição de frequência de palavras no corpus de treinamento na qualidade dos embeddings aprendidos. Como podemos modificar $P(+|w,c)$ para mitigar o viés em direção a palavras frequentes?

3. Demonstre matematicamente como a propriedade de composicionalidade (e.g., vetor("king") - vetor("man") + vetor("woman") ≈ vetor("queen")) emerge da estrutura do modelo skip-gram e da função objetivo utilizada.

4. Desenvolva uma prova formal de que, sob certas condições, os embeddings aprendidos pelo skip-gram convergem para a fatoração da matriz de informação mútua pontual (PMI) entre palavras e contextos.

5. Proponha e analise teoricamente uma extensão do modelo skip-gram que incorpore informações sintáticas explícitas (e.g., dependências gramaticais) na definição de $P(+|w,c)$. Como isso afetaria as propriedades dos embeddings resultantes?

## Anexos

### A.1 Derivação do Gradiente de $P(+|w,c)$

O gradiente de $P(+|w,c)$ em relação a $w$ é dado por:

$$
\begin{align*}
\frac{\partial P(+|w,c)}{\partial w} &= \frac{\partial}{\partial w} \sigma(c \cdot w) \\
&= \sigma(c \cdot w)(1 - \sigma(c \cdot w)) \cdot c \\
&= P(+|w,c)(1 - P(+|w,c)) \cdot c
\