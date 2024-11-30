## Bidirectional vs. Causal Transformers: Comparação do fluxo de informação e adequação para diferentes tarefas de NLP

<image: Um diagrama comparativo mostrando o fluxo de informação em transformers bidirecionais e causais, com setas indicando a direção da atenção e exemplos de aplicações para cada tipo>

### Introdução

Os modelos de linguagem baseados em transformers revolucionaram o campo do Processamento de Linguagem Natural (NLP) nos últimos anos. Duas arquiteturas principais emergiram: os transformers bidirecionais e os transformers causais (ou unidirecionais). Estas arquiteturas diferem fundamentalmente em como processam a informação e, consequentemente, são adequadas para diferentes tipos de tarefas de NLP [1]. Este resumo explorará em profundidade as diferenças entre essas arquiteturas, focando no fluxo de informação e na adequação para tarefas específicas como classificação de sequências e geração de texto.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Transformer Bidirecional** | Modelo que permite que a atenção flua em ambas as direções, considerando o contexto completo da sequência de entrada. Adequado para tarefas que requerem compreensão do contexto completo, como classificação de sequências [1]. |
| **Transformer Causal**       | Modelo que restringe o fluxo de atenção apenas para tokens anteriores na sequência, mantendo a causalidade. Ideal para tarefas de geração de texto e modelagem de linguagem autoregressive [1]. |
| **Atenção Self-Attention**   | Mecanismo que permite que um modelo pese a importância de diferentes partes da entrada ao processar cada token. É o componente central que diferencia o fluxo de informação entre transformers bidirecionais e causais [1]. |

> ⚠️ **Nota Importante**: A escolha entre transformers bidirecionais e causais deve ser baseada na natureza da tarefa de NLP em questão, considerando se a aplicação requer acesso ao contexto completo ou se deve manter a causalidade para geração de texto [1].

### Fluxo de Informação em Transformers Bidirecionais vs. Causais

<image: Um diagrama detalhado mostrando o fluxo de atenção em um transformer bidirecional (com setas bidirecionais entre todos os tokens) e um transformer causal (com setas unidirecionais apenas para tokens anteriores), destacando as diferenças na máscara de atenção>

Os transformers bidirecionais e causais diferem fundamentalmente na forma como a informação flui através da rede durante o processamento de uma sequência [1].

#### Transformer Bidirecional

Em um transformer bidirecional, como o BERT (Bidirectional Encoder Representations from Transformers), cada token na sequência de entrada pode atender a todos os outros tokens, independentemente de sua posição [1]. Isso é realizado através de uma matriz de atenção completa:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Onde $Q$, $K$, e $V$ são as matrizes de query, key e value, respectivamente, e $d_k$ é a dimensão das chaves.

> ✔️ **Ponto de Destaque**: A ausência de máscara na atenção permite que cada token tenha acesso ao contexto completo da sequência, tanto à esquerda quanto à direita [1].

#### Transformer Causal

Em contraste, um transformer causal, como o GPT (Generative Pre-trained Transformer), utiliza uma máscara de atenção triangular inferior para garantir que cada token só possa atender aos tokens anteriores na sequência [1]:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

Onde $M$ é uma matriz de máscara com valores negativos muito grandes (-∞) nas posições superiores à diagonal principal.

> ❗ **Ponto de Atenção**: A máscara causal é crucial para manter a propriedade autoregressive necessária para a geração de texto [1].

### Adequação para Diferentes Tarefas de NLP

A escolha entre transformers bidirecionais e causais depende fortemente da natureza da tarefa de NLP em questão [1].

#### 👍Vantagens do Transformer Bidirecional

* Compreensão contextual completa: Ideal para tarefas que requerem entendimento profundo do contexto, como classificação de sequências e resposta a perguntas [1].
* Representações ricas: Gera embeddings contextuais que capturam informações de toda a sequência [1].

#### 👎Desvantagens do Transformer Bidirecional

* Não adequado para geração de texto: A visão completa do contexto torna difícil a geração autoregressive [1].
* Maior complexidade computacional: O acesso ao contexto completo pode resultar em maior uso de recursos [1].

#### 👍Vantagens do Transformer Causal

* Geração de texto natural: Perfeito para tarefas de geração de linguagem, como completar frases ou tradução automática [1].
* Treinamento eficiente: A máscara causal permite paralelização eficiente durante o treinamento [1].

#### 👎Desvantagens do Transformer Causal

* Contexto limitado: A visão unidirecional pode limitar a compreensão em tarefas que requerem contexto bidirecional [1].
* Potencial viés para tokens iniciais: Tokens no início da sequência podem ter maior influência nas previsões [1].

#### Questões Técnicas/Teóricas

1. Como a diferença no fluxo de informação entre transformers bidirecionais e causais afeta a capacidade de capturar dependências de longo alcance em uma sequência?
2. Em um cenário de classificação de sentimentos, como você justificaria a escolha de um transformer bidirecional sobre um causal, considerando o fluxo de informação em cada arquitetura?

### Implementação e Considerações Práticas

Ao implementar transformers bidirecionais ou causais, é crucial entender como a atenção é calculada e aplicada. Vamos examinar um exemplo simplificado de implementação da atenção em PyTorch para cada tipo:

#### Transformer Bidirecional

```python
import torch
import torch.nn.functional as F

def bidirectional_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, value)
```

#### Transformer Causal

```python
import torch
import torch.nn.functional as F

def causal_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Criar máscara causal
    mask = torch.triu(torch.ones_like(scores), diagonal=1)
    scores = scores.masked_fill(mask == 1, float('-inf'))
    
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, value)
```

> ✔️ **Ponto de Destaque**: A principal diferença na implementação está na aplicação da máscara causal, que garante que cada token só atenda aos tokens anteriores [1].

### Aplicações Específicas

#### Classificação de Sequências

Para tarefas de classificação de sequências, como classificação de sentimentos ou detecção de tópicos, transformers bidirecionais como BERT são geralmente preferidos [1]. A capacidade de acessar o contexto completo permite uma compreensão mais profunda da sequência:

```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    return torch.argmax(logits, dim=1)
```

#### Geração de Texto

Para geração de texto, transformers causais como GPT são mais apropriados [1]. A natureza autoregressive permite a geração de texto coerente e fluente:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

> ❗ **Ponto de Atenção**: A escolha entre BERT e GPT para estas tarefas reflete diretamente as diferenças no fluxo de informação e na adequação de cada arquitetura para tipos específicos de problemas de NLP [1].

#### Questões Técnicas/Teóricas

1. Como a implementação da máscara causal em transformers afeta o desempenho computacional durante o treinamento e a inferência?
2. Descreva um cenário de NLP onde a combinação de transformers bidirecionais e causais poderia ser benéfica, e explique como você integraria os dois tipos de modelos.

### Conclusão

A distinção entre transformers bidirecionais e causais é fundamental para compreender e aplicar efetivamente modelos de linguagem em diversas tarefas de NLP [1]. Transformers bidirecionais, com sua capacidade de acessar o contexto completo, são ideais para tarefas que requerem compreensão profunda, como classificação de sequências [1]. Por outro lado, transformers causais, com sua natureza autoregressive, são superiores em tarefas de geração de texto [1].

A escolha entre essas arquiteturas deve ser guiada pela natureza específica da tarefa em questão, considerando cuidadosamente as vantagens e limitações de cada abordagem [1]. À medida que o campo de NLP continua a evoluir, é provável que vejamos desenvolvimentos que busquem combinar os pontos fortes de ambas as arquiteturas, potencialmente levando a modelos ainda mais poderosos e versáteis.

### Questões Avançadas

1. Considerando as diferenças no fluxo de informação entre transformers bidirecionais e causais, como você projetaria um modelo híbrido que pudesse alternar eficientemente entre modos de atenção completa e causal dependendo da tarefa?

2. Analise criticamente o impacto do fluxo de informação unidirecional em transformers causais na geração de texto de longa duração. Como isso afeta a coerência global, e quais estratégias poderiam ser empregadas para mitigar possíveis limitações?

3. Em um cenário de tradução automática, compare e contraste o uso de um transformer bidirecional (como BERT) para codificação e um transformer causal (como GPT) para decodificação. Quais são os trade-offs em termos de qualidade de tradução, eficiência computacional e capacidade de lidar com diferentes pares de idiomas?

### Referências

[1] "Let's begin by introducing the bidirectional transformer encoder that underlies models like BERT and its descendants like RoBERTa (Liu et al., 2019) or SpanBERT (Joshi et al., 2020). In Chapter 10 we explored causal (left-to-right) transformers that can serve as the basis for powerful language models—models that can easily be applied to autoregressive generation problems such as contextual generation, summarization and machine translation. However, when applied to sequence classification and labeling problems causal models have obvious shortcomings since they are based on an incremental, left-to-right processing of their inputs. If we want to assign the correct named-entity tag to each word in a sentence, or other sophisticated linguistic labels like the parse tags we'll introduce in later chapters, we'll want to be able to take into account information from the right context as we process each element. Fig. 11.1a, reproduced here from Chapter 10, illustrates the information flow in the purely left-to-right approach of Chapter 10. As can be seen, the hidden state computation at each point in time is based solely on the current and earlier elements of the input, ignoring potentially useful information located to the right of each tagging decision." (Trecho de Fine-Tuning and Masked Language Models)