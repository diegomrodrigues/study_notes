## Absolute vs. Relative Positional Embeddings em Transformers

<image: Uma ilustração comparativa mostrando dois transformers lado a lado - um utilizando embeddings posicionais absolutos e outro usando embeddings posicionais relativos. A imagem deve destacar como cada tipo de embedding codifica a informação posicional de forma diferente nos vetores de entrada.>

### Introdução

Os **embeddings posicionais** são um componente crucial na arquitetura Transformer, permitindo que o modelo capture informações sobre a ordem sequencial dos tokens de entrada [1]. Essa funcionalidade é essencial, uma vez que o mecanismo de atenção em si é invariante à ordem dos tokens. Neste resumo, exploraremos em profundidade duas abordagens principais para embeddings posicionais: **absolutos** e **relativos**, analisando suas características, vantagens, desvantagens e aplicabilidades em diferentes cenários e tarefas de processamento de linguagem natural.

### Conceitos Fundamentais

| Conceito                             | Explicação                                                   |
| ------------------------------------ | ------------------------------------------------------------ |
| **Embeddings Posicionais Absolutos** | Vetores únicos que codificam a posição absoluta de cada token na sequência. São adicionados diretamente aos embeddings dos tokens. [1] |
| **Embeddings Posicionais Relativos** | Codificam a distância relativa entre pares de tokens, permitindo que o modelo capture relações posicionais de forma mais flexível. [2] |
| **Invariância à Translação**         | Propriedade onde o significado de uma subsequência não muda com sua posição absoluta na sequência. Importante para certos tipos de tarefas de NLP. [3] |

> ⚠️ **Nota Importante**: A escolha entre embeddings posicionais absolutos e relativos pode impactar significativamente o desempenho do modelo em diferentes tarefas e comprimentos de sequência.

### Embeddings Posicionais Absolutos

<image: Um diagrama mostrando como os embeddings posicionais absolutos são somados aos embeddings dos tokens de entrada em um transformer. O diagrama deve ilustrar a correspondência um-para-um entre posições e vetores de embedding.>

Os embeddings posicionais absolutos, introduzidos no artigo original do Transformer [1], são vetores únicos associados a cada posição na sequência de entrada. Eles são somados diretamente aos embeddings dos tokens antes de serem processados pelas camadas de atenção.

A formulação matemática para os embeddings posicionais absolutos, conforme proposta originalmente, é:

$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

Onde:
- $pos$ é a posição absoluta do token na sequência
- $i$ é a dimensão do embedding
- $d_{model}$ é a dimensão do modelo

Esta formulação permite que o modelo aprenda a atender a posições absolutas na sequência. Os embeddings posicionais absolutos têm algumas propriedades interessantes:

1. **Determinísticos**: Não são parâmetros aprendidos, mas calculados de forma determinística.
2. **Periodicidade**: As funções seno e cosseno fornecem uma certa periodicidade, permitindo que o modelo generalize para sequências mais longas do que as vistas durante o treinamento.
3. **Unicidade**: Cada posição tem um vetor único, permitindo que o modelo diferencie tokens em diferentes posições.

#### Vantagens e Desvantagens dos Embeddings Posicionais Absolutos

👍 **Vantagens**:
- Simplicidade de implementação [4]
- Eficiência computacional (podem ser pré-computados) [4]
- Capacidade de generalizar para sequências mais longas do que as vistas durante o treinamento (até certo ponto) [1]

👎 **Desvantagens**:
- Limitação em capturar relações relativas entre tokens distantes [5]
- Potencial perda de desempenho em tarefas que requerem invariância à translação [3]
- Dificuldade em lidar com sequências muito longas além do comprimento máximo visto durante o treinamento [6]

#### Questões Técnicas/Teóricas

1. Como a periodicidade dos embeddings posicionais absolutos contribui para a generalização do modelo para sequências mais longas? Explique matematicamente.

2. Em uma tarefa de classificação de documentos longos, como os embeddings posicionais absolutos podem impactar o desempenho do modelo? Considere documentos que excedem significativamente o comprimento máximo visto durante o treinamento.

### Embeddings Posicionais Relativos

<image: Um diagrama ilustrando como os embeddings posicionais relativos são incorporados no cálculo da atenção em um transformer. O diagrama deve mostrar como as distâncias relativas entre tokens são utilizadas para modular os scores de atenção.>

Os embeddings posicionais relativos foram introduzidos como uma alternativa aos embeddings absolutos, visando superar algumas de suas limitações [7]. Em vez de codificar posições absolutas, eles codificam as distâncias relativas entre pares de tokens.

A implementação dos embeddings posicionais relativos pode variar, mas uma abordagem comum é modificar o cálculo da atenção para incorporar informações posicionais relativas. Uma formulação simplificada pode ser expressa como:

$$
Attention(Q, K, V) = softmax(\frac{QK^T + R}{\sqrt{d_k}})V
$$

Onde $R$ é uma matriz que codifica as distâncias relativas entre todas as posições na sequência.

Uma implementação mais sofisticada, proposta por Shaw et al. [7], introduz embeddings de posição relativa $a_{ij}$ e $b_{ij}$ no cálculo da atenção:

$$
e_{ij} = \frac{x_iW^Q(x_jW^K + a_{ij})^T}{\sqrt{d_z}}
$$

$$
y_i = \sum_{j=1}^n \alpha_{ij}(x_jW^V + b_{ij})
$$

Onde:
- $x_i$ e $x_j$ são os embeddings dos tokens nas posições $i$ e $j$
- $W^Q$, $W^K$, $W^V$ são matrizes de projeção para query, key e value
- $a_{ij}$ e $b_{ij}$ são embeddings da posição relativa entre $i$ e $j$
- $\alpha_{ij}$ são os pesos de atenção derivados de $e_{ij}$ através de softmax

Esta formulação permite que o modelo capture relações posicionais de forma mais flexível e eficiente.

#### Vantagens e Desvantagens dos Embeddings Posicionais Relativos

👍 **Vantagens**:
- Melhor captura de relações locais e distantes entre tokens [7]
- Maior invariância à translação, beneficiando certas tarefas de NLP [3]
- Potencial para melhor generalização em sequências longas [8]

👎 **Desvantagens**:
- Maior complexidade computacional [9]
- Potencial aumento no número de parâmetros do modelo [7]
- Implementação mais complexa comparada aos embeddings absolutos [9]

#### Questões Técnicas/Teóricas

1. Como a incorporação de embeddings posicionais relativos no cálculo da atenção afeta a complexidade computacional do modelo? Analise em termos de operações de matriz.

2. Em uma tarefa de tradução automática, como os embeddings posicionais relativos podem melhorar o alinhamento entre palavras de idiomas com estruturas sintáticas diferentes? Dê um exemplo concreto.

### Comparação e Análise

Para uma comparação mais detalhada entre embeddings posicionais absolutos e relativos, consideremos os seguintes aspectos:

| Aspecto                                  | Embeddings Absolutos | Embeddings Relativos |
| ---------------------------------------- | -------------------- | -------------------- |
| **Captura de Contexto Local**            | Limitada [5]         | Superior [7]         |
| **Invariância à Translação**             | Baixa [3]            | Alta [3]             |
| **Eficiência Computacional**             | Alta [4]             | Moderada a Baixa [9] |
| **Generalização para Sequências Longas** | Moderada [1]         | Alta [8]             |
| **Complexidade de Implementação**        | Baixa [4]            | Alta [9]             |

> ✔️ **Ponto de Destaque**: A escolha entre embeddings absolutos e relativos deve ser baseada nas características específicas da tarefa, no comprimento das sequências e nos recursos computacionais disponíveis.

### Implementação e Considerações Práticas

Ao implementar embeddings posicionais em um modelo Transformer, é crucial considerar o trade-off entre desempenho e eficiência. Aqui está um exemplo simplificado de como implementar embeddings posicionais absolutos em PyTorch:

```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]
```

Para embeddings posicionais relativos, a implementação é mais complexa e geralmente envolve modificar o cálculo da atenção. Aqui está um esboço simplificado:

```python
import torch
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.rel_embeddings = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
        
    def forward(self, q, k, v, seq_len):
        rel_pos = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        rel_pos += seq_len - 1  # shift to positive indices
        rel_emb = self.rel_embeddings[rel_pos]
        
        # Incorporate relative embeddings into attention calculation
        logits = torch.matmul(q, k.transpose(-2, -1)) + torch.einsum('bhid,jd->bhij', q, rel_emb)
        return torch.matmul(nn.functional.softmax(logits, dim=-1), v)
```

> ❗ **Ponto de Atenção**: A implementação de embeddings posicionais relativos pode variar significativamente dependendo da abordagem específica escolhida. A versão acima é uma simplificação e pode requerer ajustes para casos de uso específicos.

### Conclusão

A escolha entre embeddings posicionais absolutos e relativos é crucial para o desempenho de modelos Transformer em diversas tarefas de NLP. Embeddings absolutos oferecem simplicidade e eficiência, sendo adequados para muitas aplicações padrão. Por outro lado, embeddings relativos proporcionam maior flexibilidade e potencial de generalização, especialmente em tarefas que envolvem sequências longas ou requerem invariância à translação [3][7][8].

A decisão deve ser baseada nas características específicas da tarefa, nos recursos computacionais disponíveis e nos requisitos de desempenho do modelo. Em alguns casos, abordagens híbridas ou variações mais recentes desses métodos podem oferecer o melhor equilíbrio entre desempenho e eficiência.

À medida que a pesquisa em NLP avança, é provável que surjam novas técnicas de embedding posicional, possivelmente combinando os pontos fortes das abordagens absoluta e relativa ou introduzindo conceitos inteiramente novos para capturar informações posicionais em modelos de linguagem.

### Questões Avançadas

1. Como você projetaria um experimento para comparar o desempenho de embeddings posicionais absolutos e relativos em uma tarefa de sumarização de documentos longos? Considere aspectos como a estrutura do documento, a importância da ordem das sentenças e a capacidade de generalização para diferentes comprimentos de texto.

2. Analise o impacto potencial de embeddings posicionais relativos na interpretabilidade de modelos de linguagem. Como eles podem afetar nossa capacidade de entender as decisões do modelo em tarefas como análise de sentimentos ou extração de entidades?

3. Proponha uma abordagem híbrida que combine elementos de embeddings posicionais absolutos e relativos. Como essa abordagem poderia superar as limitações de cada método individual? Discuta os desafios de implementação e os potenciais benefícios em diferentes cenários de NLP.

### Referências

[1] "Embeddings posicionais absolutos, introduzidos no artigo original do Transformer, são vetores únicos associados a cada posição na sequência de entrada. Eles são somados diretamente aos embeddings dos tokens antes de serem processados pelas camadas de atenção." (Trecho de Transformers and Large Language Models - Chapter 10)

[2] "Embeddings Posicionais Relativos: Codificam a distância relativa entre pares de tokens, permitindo que o modelo capture relações posicionais de forma mais flexível." (Trecho de Transformers and Large Language Models - Chapter 10)

[3] "Invariância à Translação: Propriedade onde o significado de uma subsequência não muda com sua posição absoluta na sequência. Importante para certos tipos de tarefas de NLP." (Trecho de Transformers and Large Language Models - Chapter 10)

[4] "Simplicidade de implementação [...] Eficiência computacional (podem ser pré-computados)" (Trecho de Transformers and Large Language Models - Chapter 10)

[5] "Limitação em capturar relações relativas entre tokens distantes" (Trecho de Transformers and Large Language Models - Chapter 10)

[6] "Dificuldade em lidar com sequências muito longas além do comprimento máximo visto durante o treinamento" (Trecho de Transformers and Large Language Models - Chapter 10)

[7] "Os embeddings posicionais relativos foram introduzidos como uma alternativa aos embeddings absolutos, visando superar algumas de suas limitações [...] Uma implementação mais sofisticada, proposta por Shaw et al., introduz embeddings de posição relativa a_{ij} e b_{ij} no cálculo da atenção" (Trecho de Transformers and Large Language Models - Chapter 10)

[8] "Potencial para melhor generalização em sequências longas" (Trecho de Transformers and Large Language Models - Chapter 10)

[9] "Maior complexidade computacional [...] Implementação mais complexa comparada aos embeddings absolutos" (Trecho de Transformers and Large Language Models - Chapter 10)