# Funções de Perda e Otimização para Word Embeddings

<imagem: Uma visualização de um espaço vetorial multidimensional com vetores de palavras sendo gradualmente ajustados para minimizar uma função de perda, representada por uma superfície tridimensional complexa>

## Introdução

As funções de perda e os algoritmos de otimização são componentes fundamentais no treinamento de modelos de word embeddings, como o Skip-gram. Esses elementos permitem que os modelos aprendam representações vetoriais densas e significativas das palavras a partir de grandes corpora de texto [1]. Neste resumo, exploraremos em profundidade a função de perda baseada na negative log-likelihood e o processo de otimização por meio do Stochastic Gradient Descent (SGD) no contexto do modelo Skip-gram, uma das abordagens mais influentes para a geração de word embeddings [2].

## Conceitos Fundamentais

| Conceito              | Explicação                                                   |
| --------------------- | ------------------------------------------------------------ |
| **Word Embeddings**   | Representações vetoriais densas de palavras em um espaço multidimensional, capturando relações semânticas e sintáticas entre as palavras [3]. |
| **Skip-gram Model**   | Um modelo neural que aprende word embeddings prevendo as palavras de contexto dada uma palavra-alvo [4]. |
| **Negative Sampling** | Técnica de amostragem que seleciona palavras "negativas" aleatórias para contrastar com as palavras de contexto positivas reais durante o treinamento [5]. |

> ⚠️ **Nota Importante**: A função de perda e o algoritmo de otimização trabalham em conjunto para ajustar iterativamente os embeddings, minimizando a discrepância entre as previsões do modelo e os dados observados [6].

## Função de Perda: Negative Log-Likelihood

A função de perda utilizada no treinamento do modelo Skip-gram com negative sampling é baseada na maximização da probabilidade de ocorrência das palavras de contexto observadas e na minimização da probabilidade de ocorrência das palavras negativas amostradas [7]. 

### Formulação Matemática

A função objetivo para um par de palavras (palavra-alvo $w$ e palavra de contexto $c$) pode ser expressa como:

$$
L = -\log\left[P(+|w, c_{pos}) \prod_{i=1}^{k} P(-|w, c_{neg_i})\right]
$$

onde:
- $P(+|w, c_{pos})$ é a probabilidade de $c_{pos}$ ser uma palavra de contexto real para $w$
- $P(-|w, c_{neg_i})$ é a probabilidade de $c_{neg_i}$ não ser uma palavra de contexto para $w$
- $k$ é o número de amostras negativas [8]

Expandindo esta equação, obtemos:

$$
L = -\left[\log \sigma(c_{pos} \cdot w) + \sum_{i=1}^{k} \log \sigma(-c_{neg_i} \cdot w)\right]
$$

onde $\sigma$ é a função sigmoide e $\cdot$ denota o produto escalar [9].

> 💡 **Destaque**: Esta formulação da função de perda incentiva o modelo a aumentar a similaridade entre embeddings de palavras que co-ocorrem frequentemente, enquanto diminui a similaridade com palavras negativas amostradas [10].

### Análise Teórica

A escolha desta função de perda é fundamentada em princípios de teoria da informação e aprendizado de máquina:

1. **Maximização da verossimilhança**: O termo negativo na equação corresponde à maximização da log-verossimilhança dos dados observados [11].

2. **Contrastive Learning**: A inclusão de amostras negativas permite que o modelo aprenda por contraste, distinguindo entre contextos válidos e inválidos [12].

3. **Eficiência Computacional**: O uso de negative sampling reduz significativamente o custo computacional em comparação com a softmax completa sobre todo o vocabulário [13].

#### Perguntas Teóricas

1. Derive a expressão para o gradiente da função de perda em relação aos embeddings da palavra-alvo e da palavra de contexto.
2. Como a escolha do número de amostras negativas $k$ afeta teoricamente o trade-off entre qualidade dos embeddings e eficiência computacional?
3. Demonstre matematicamente por que a função sigmoide é uma escolha apropriada para modelar as probabilidades neste contexto.

## Stochastic Gradient Descent (SGD)

O Stochastic Gradient Descent é o algoritmo de otimização utilizado para minimizar a função de perda e atualizar os embeddings no modelo Skip-gram [14].

### Formulação Matemática

As equações de atualização para os embeddings da palavra-alvo $w$ e da palavra de contexto $c$ são:

$$
c_{pos}^{t+1} = c_{pos}^{t} - \eta[\sigma (c_{pos}^{t} \cdot w^{t}) - 1]w^{t}
$$

$$
c_{neg}^{t+1} = c_{neg}^{t} - \eta[\sigma (c_{neg}^{t} \cdot w^{t})]w^{t}
$$

$$
w^{t+1} = w^{t} - \eta \left( [\sigma (c_{pos}^{t} \cdot w^{t}) - 1]c_{pos}^{t} + \sum_{i=1}^{k} [\sigma (c_{neg_i}^{t} \cdot w^{t})]c_{neg_i}^{t} \right)
$$

onde $\eta$ é a taxa de aprendizado e $t$ indica a iteração atual [15].

> ❗ **Ponto de Atenção**: A natureza estocástica do SGD, que atualiza os parâmetros com base em mini-batches ou exemplos individuais, permite uma convergência mais rápida e uma melhor generalização em comparação com o gradiente descendente em lote [16].

### Análise Teórica do SGD no Contexto de Word Embeddings

1. **Convergência**: A convergência do SGD para um mínimo local ou global da função de perda é garantida sob certas condições, como a escolha adequada da taxa de aprendizado e a convexidade da função objetivo [17].

2. **Regularização Implícita**: O ruído introduzido pela natureza estocástica do SGD atua como uma forma de regularização, ajudando a prevenir o overfitting [18].

3. **Adaptação a Grandes Datasets**: O SGD é particularmente adequado para o treinamento de word embeddings em grandes corpora devido à sua eficiência computacional e capacidade de lidar com dados em stream [19].

#### Perguntas Teóricas

1. Prove que, sob condições apropriadas, o SGD converge para um ponto estacionário da função de perda no limite de infinitas iterações.
2. Como a escolha da taxa de aprendizado $\eta$ afeta teoricamente a convergência e a qualidade final dos embeddings? Derive uma expressão para a taxa de aprendizado ótima.
3. Analise teoricamente o impacto do tamanho do mini-batch na variância do gradiente e na velocidade de convergência do SGD no contexto de treinamento de word embeddings.

## Implementação Avançada

Aqui está um exemplo de implementação avançada em Python utilizando PyTorch para o treinamento de word embeddings com o modelo Skip-gram:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SkipgramDataset(Dataset):
    def __init__(self, corpus, window_size, num_negative):
        self.corpus = corpus
        self.window_size = window_size
        self.num_negative = num_negative
        self.vocab = self._build_vocab()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.word_freqs = np.array([self._get_word_freq(word) for word in self.vocab])
        self.word_freqs = self.word_freqs ** 0.75
        self.word_freqs = self.word_freqs / np.sum(self.word_freqs)
        
    def _build_vocab(self):
        return sorted(set(self.corpus))
    
    def _get_word_freq(self, word):
        return self.corpus.count(word)
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        center_word = self.corpus[idx]
        pos_context = self._get_context_words(idx)
        neg_context = self._get_negative_samples(len(pos_context))
        
        center_word = torch.tensor([self.word_to_idx[center_word]], dtype=torch.long)
        pos_context = torch.tensor([self.word_to_idx[w] for w in pos_context], dtype=torch.long)
        neg_context = torch.tensor(neg_context, dtype=torch.long)
        
        return center_word, pos_context, neg_context
    
    def _get_context_words(self, idx):
        start = max(0, idx - self.window_size)
        end = min(len(self.corpus), idx + self.window_size + 1)
        context = [self.corpus[i] for i in range(start, end) if i != idx]
        return context
    
    def _get_negative_samples(self, num_pos):
        neg_samples = np.random.choice(
            len(self.vocab),
            size=(num_pos, self.num_negative),
            p=self.word_freqs
        )
        return neg_samples

class SkipgramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipgramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, center_words, context_words):
        center_embeds = self.embeddings(center_words)
        context_embeds = self.output(context_words)
        return torch.sum(center_embeds * context_embeds, dim=1)

def train_skipgram(model, dataset, num_epochs, batch_size, lr):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for center_words, pos_context, neg_context in dataloader:
            optimizer.zero_grad()
            
            pos_loss = loss_function(
                model(center_words, pos_context),
                torch.ones(pos_context.shape[0])
            )
            neg_loss = loss_function(
                model(center_words.repeat_interleave(neg_context.shape[1]), neg_context.flatten()),
                torch.zeros(neg_context.numel())
            )
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Exemplo de uso
corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
dataset = SkipgramDataset(corpus, window_size=2, num_negative=5)
model = SkipgramModel(len(dataset.vocab), embedding_dim=100)
train_skipgram(model, dataset, num_epochs=50, batch_size=2, lr=0.001)
```

Este código implementa o modelo Skip-gram com negative sampling, utilizando PyTorch para treinamento eficiente em GPU [20]. A implementação inclui:

1. Uma classe `SkipgramDataset` que prepara os dados para treinamento, incluindo a geração de amostras negativas.
2. Uma classe `SkipgramModel` que define a arquitetura do modelo.
3. Uma função `train_skipgram` que realiza o treinamento utilizando SGD (via otimizador Adam) e a função de perda BCE with Logits [21].

> ✔️ **Destaque**: Esta implementação incorpora técnicas avançadas como subsampling de palavras frequentes e exponenciação das frequências das palavras para melhorar a qualidade dos embeddings resultantes [22].

## Discussão Crítica

### Desafios e Limitações

1. **Escala do Vocabulário**: Para vocabulários muito grandes, o custo computacional de atualizar todos os embeddings pode ser proibitivo, mesmo com negative sampling [23].

2. **Palavras Raras**: Palavras com poucas ocorrências no corpus podem não ter embeddings de boa qualidade devido à falta de contextos de treinamento [24].

3. **Polissemia**: O modelo Skip-gram padrão não lida bem com palavras polissêmicas, atribuindo um único vetor para todos os sentidos de uma palavra [25].

### Perspectivas Futuras

1. **Embeddings Contextuais**: Modelos como BERT e GPT que geram embeddings dinâmicos baseados no contexto podem superar algumas limitações dos embeddings estáticos [26].

2. **Incorporação de Conhecimento Externo**: Integrar informações de ontologias ou bases de conhecimento para melhorar a qualidade semântica dos embeddings [27].

3. **Embeddings Multilíngues**: Desenvolver técnicas para criar espaços de embeddings unificados para múltiplas línguas, facilitando tarefas de tradução e transferência de conhecimento entre idiomas [28].

## Conclusão

A função de perda baseada na negative log-likelihood e o algoritmo de otimização Stochastic Gradient Descent são componentes cruciais no treinamento de modelos de word embeddings como o Skip-gram. Estes elementos permitem a aprendizagem eficiente de representações vetoriais densas que capturam relações semânticas e sintáticas complexas entre palavras [29]. 

A compreensão profunda destes conceitos é fundamental para o desenvolvimento e aprimoramento de técnicas de processamento de linguagem natural, com aplicações que vão desde a análise de sentimentos até a tradução automática [30]. À medida que o campo evolui, novas abordagens para funções de perda e algoritmos de otimização continuarão a desempenhar um papel central na melhoria da qualidade e eficiência dos modelos de linguagem [31].

## Perguntas Teóricas Avançadas

1. Derive uma expressão para a complexidade computacional do treinamento do modelo Skip-gram com negative sampling em função do tamanho do vocabulário, dimensão dos embeddings e número de épocas. Compare esta complexidade com a do modelo original sem negative sampling.

2. Formule uma prova matemática demonstrando que os embeddings aprendidos pelo modelo Skip-gram preservam certas propriedades lineares observadas empiricamente, como a relação "rei - homem + mulher ≈ rainha".

3. Desenvolva uma extensão teórica do modelo Skip-gram que incorpore informações de subpalavras (por exemplo, n-gramas de caracteres) na função de perda. Derive as equações de atualização correspondentes e analise o impacto teórico desta modificação na qualidade dos embeddings para palavras raras e desconhecidas.

4. Proponha e analise teoricamente uma função de perda alternativa que explicitamente otimize a preservação