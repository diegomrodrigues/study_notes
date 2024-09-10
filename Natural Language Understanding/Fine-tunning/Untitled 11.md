## Token Sampling Strategies em Masked Language Models

<image: Uma representação visual de diferentes estratégias de amostragem de tokens, mostrando máscaras, substituições aleatórias e tokens inalterados em uma sequência de texto>

### Introdução

As estratégias de amostragem de tokens desempenham um papel crucial no treinamento de modelos de linguagem mascarados (Masked Language Models - MLMs). Estas técnicas formam a base do processo de aprendizagem em modelos como BERT e suas variantes, permitindo que eles capturem relações contextuais profundas em dados textuais [1]. Este resumo explorará em detalhes as diferentes abordagens para selecionar e manipular tokens durante o treinamento de MLMs, focando nas três principais estratégias: mascaramento, substituição aleatória e manutenção inalterada de tokens.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Masked Language Modeling** | Paradigma de treinamento onde o modelo aprende a prever tokens mascarados em uma sequência, permitindo aprendizado bidirecional [1]. |
| **Token Sampling**           | Processo de seleção de tokens em uma sequência de entrada para aplicação de estratégias de mascaramento ou modificação durante o treinamento [1]. |
| **Span-based Masking**       | Técnica avançada que envolve a seleção e mascaramento de spans (sequências contíguas) de tokens, em vez de tokens individuais [5]. |

> ⚠️ **Nota Importante**: A escolha e implementação adequada das estratégias de amostragem de tokens são críticas para o desempenho e generalização dos MLMs.

### Estratégias de Amostragem de Tokens

<image: Diagrama mostrando o fluxo de processamento de uma sequência de texto através das diferentes estratégias de amostragem, com exemplos visuais de cada técnica>

As estratégias de amostragem de tokens em MLMs são projetadas para criar um equilíbrio entre o aprendizado de representações contextuais robustas e a prevenção de overfitting. O BERT, um dos modelos pioneiros nesta abordagem, implementa uma combinação específica destas estratégias [1].

#### 1. Mascaramento de Tokens

O mascaramento de tokens é a principal técnica utilizada em MLMs. Nesta abordagem, tokens selecionados são substituídos por um token especial [MASK].

> ✔️ **Ponto de Destaque**: 80% dos tokens selecionados para manipulação são mascarados no BERT [1].

Processo matemático de mascaramento:

Seja $x = (x_1, ..., x_n)$ uma sequência de tokens de entrada, e $M$ o conjunto de índices dos tokens selecionados para manipulação. Para cada $i \in M$:

$$
x_i' = \begin{cases} 
      [MASK] & \text{com probabilidade } 0.8 \\
      x_i & \text{caso contrário}
   \end{cases}
$$

Esta estratégia força o modelo a aprender representações contextuais bidirecionais, pois ele deve prever o token mascarado usando tanto o contexto à esquerda quanto à direita [1].

#### 2. Substituição Aleatória

Para evitar que o modelo se torne dependente demais do token [MASK] durante o fine-tuning, uma parte dos tokens selecionados é substituída por tokens aleatórios do vocabulário.

> ✔️ **Ponto de Destaque**: 10% dos tokens selecionados são substituídos aleatoriamente no BERT [1].

Processo de substituição aleatória:

Para $i \in M$ não mascarados na etapa anterior:

$$
x_i' = \begin{cases} 
      r_i & \text{com probabilidade } 0.1 \\
      x_i & \text{caso contrário}
   \end{cases}
$$

Onde $r_i$ é um token aleatório do vocabulário.

Esta técnica ajuda o modelo a manter a distribuição de saída próxima à distribuição real dos tokens, mesmo quando não há mascaramento [1].

#### 3. Manutenção Inalterada

Uma parte dos tokens selecionados é mantida inalterada. Isso ajuda o modelo a aprender a distinguir entre tokens que precisam ser previstos e tokens que fornecem contexto.

> ✔️ **Ponto de Destaque**: 10% dos tokens selecionados permanecem inalterados no BERT [1].

Matematicamente, para os tokens restantes em $M$:

$$
x_i' = x_i
$$

Esta estratégia ajuda o modelo a lidar com discrepâncias entre o treinamento (onde há tokens mascarados) e a inferência (onde todos os tokens estão presentes) [1].

#### Questões Técnicas/Teóricas

1. Como a proporção de 80% (mascaramento), 10% (substituição aleatória) e 10% (inalterado) impacta o aprendizado do modelo? Justifique matematicamente.

2. Descreva um cenário em que aumentar a proporção de substituições aleatórias poderia ser benéfico para o desempenho do modelo.

### Implementação Técnica das Estratégias de Amostragem

A implementação destas estratégias de amostragem requer um algoritmo eficiente que possa processar grandes volumes de texto. Aqui está um exemplo simplificado de como isso poderia ser implementado em Python:

```python
import random
import torch

def apply_token_sampling(tokens, mask_token_id, vocab_size, mask_prob=0.15):
    output_tokens = tokens.clone()
    mask = torch.rand(tokens.shape) < mask_prob
    
    # Seleciona tokens para manipulação
    selected_tokens = tokens[mask]
    
    # Aplica mascaramento (80%)
    mask_indices = torch.rand(selected_tokens.shape) < 0.8
    selected_tokens[mask_indices] = mask_token_id
    
    # Aplica substituição aleatória (10% dos 20% restantes)
    random_indices = torch.rand(selected_tokens.shape) < 0.5
    random_tokens = torch.randint(0, vocab_size, (random_indices.sum(),))
    selected_tokens[~mask_indices & random_indices] = random_tokens
    
    # Os 10% restantes permanecem inalterados
    
    output_tokens[mask] = selected_tokens
    return output_tokens

# Exemplo de uso
vocab_size = 30000
mask_token_id = 103
tokens = torch.randint(0, vocab_size, (1, 512))
sampled_tokens = apply_token_sampling(tokens, mask_token_id, vocab_size)
```

Este código implementa as três estratégias de amostragem do BERT, selecionando aleatoriamente 15% dos tokens para manipulação e aplicando as proporções de 80%, 10% e 10% para mascaramento, substituição aleatória e manutenção inalterada, respectivamente [1].

### Span-based Masking

Uma evolução das estratégias de amostragem de tokens é o mascaramento baseado em spans, introduzido pelo SpanBERT [5]. Esta técnica envolve a seleção e mascaramento de sequências contíguas de tokens, em vez de tokens individuais.

Processo de seleção de spans:

1. O comprimento do span é amostrado de uma distribuição geométrica:

   $$P(l) = p_l(1-p)^{l-1}, l \geq 1$$

   onde $p$ é um hiperparâmetro que controla a distribuição de comprimentos.

2. A posição inicial do span é selecionada uniformemente.

3. Todos os tokens no span selecionado são tratados como uma unidade para mascaramento, substituição ou manutenção.

> ❗ **Ponto de Atenção**: O mascaramento baseado em spans melhora o desempenho em tarefas que requerem compreensão de frases ou entidades completas.

A função de perda para o mascaramento baseado em spans inclui tanto o objetivo de Masked Language Modeling (MLM) quanto o Span Boundary Objective (SBO):

$$L(x) = L_{MLM}(x) + L_{SBO}(x)$$

onde $L_{SBO}(x_i) = -\log P(x_i|x_{k-1}, x_{l+1}, p_{l-i+1})$ [5].

#### Questões Técnicas/Teóricas

1. Como o mascaramento baseado em spans afeta a capacidade do modelo de capturar dependências de longo alcance em comparação com o mascaramento de tokens individuais?

2. Proponha uma modificação na distribuição de probabilidade para seleção de comprimento de spans que poderia melhorar o desempenho em tarefas específicas de NLP.

### Conclusão

As estratégias de amostragem de tokens são fundamentais para o treinamento eficaz de Masked Language Models. A combinação de mascaramento, substituição aleatória e manutenção inalterada de tokens permite que modelos como BERT aprendam representações contextuais robustas e generalizáveis [1]. Avanços como o mascaramento baseado em spans demonstram o potencial de refinamento dessas técnicas para melhorar ainda mais o desempenho em tarefas específicas de NLP [5].

A escolha e implementação cuidadosa dessas estratégias são cruciais para o desenvolvimento de modelos de linguagem mais poderosos e versáteis. À medida que o campo avança, é provável que vejamos novas variações e otimizações dessas técnicas, adaptadas para diferentes domínios e aplicações de processamento de linguagem natural.

### Questões Avançadas

1. Compare e contraste o impacto das estratégias de amostragem de tokens em modelos bidirecionais (como BERT) e modelos unidirecionais (como GPT). Como essas diferenças afetam o desempenho em tarefas específicas de NLP?

2. Proponha e justifique matematicamente uma nova estratégia de amostragem de tokens que poderia potencialmente superar as limitações das abordagens atuais em cenários específicos de aprendizado de máquina.

3. Analise criticamente como as estratégias de amostragem de tokens poderiam ser adaptadas para melhorar o desempenho de modelos multilíngues, considerando as diferenças estruturais entre línguas.

### Referências

[1] "In BERT, 15% of the input tokens in a training sequence are sampled for learning. Of these, 80% are replaced with [MASK], 10% are replaced with randomly selected tokens, and the remaining 10% are left unchanged." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[2] "To deal with this misalignment, we need a way to assign BIO tags to subword tokens during training and a corresponding way to recover word-level tags from subwords during decoding." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[3] "The MLM training objective is to predict the original inputs for each of the masked tokens using a bidirectional encoder of the kind described in the last section." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[4] "The cross-entropy loss from these predictions drives the training process for all the parameters in the model." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[5] "A span is a contiguous sequence of one or more words selected from a training text, prior to subword tokenization. In span-based masking, a set of randomly selected spans from a training sequence are chosen." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[6] "In the SpanBERT work that originated this technique (Joshi et al., 2020), a span length is first chosen by sampling from a geometric distribution that is biased towards shorter spans and with an upper bound of 10." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[7] "Given this span length, a starting location consistent with the desired span length and the length of the input is sampled uniformly." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[8] "Once a span is chosen for masking, all the tokens within the span are substituted according to the same regime used in BERT: 80% of the time the span elements are substituted with the [MASK] token, 10% of the time they are replaced by randomly sampled tokens from the vocabulary, and 10% of the time they are left as is." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)