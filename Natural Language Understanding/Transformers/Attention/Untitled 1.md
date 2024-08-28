## Transformers: Investigando os Papéis de Query, Key e Value no Processo de Atenção

<image: Um diagrama mostrando três vetores coloridos (query, key, value) interagindo em um espaço vetorial multidimensional, com linhas pontilhadas representando a atenção entre eles>

### Introdução

Os transformers revolucionaram o processamento de linguagem natural (NLP) ao introduzir um mecanismo de atenção baseado em Query, Key e Value (QKV). Este resumo aprofunda-se na interpretação geométrica e funcional desses componentes, explorando como eles capturam diferentes aspectos das relações entre palavras e investigando mecanismos alternativos de atenção além da atenção de produto escalar escalado.

### Conceitos Fundamentais

| Conceito    | Explicação                                                   |
| ----------- | ------------------------------------------------------------ |
| **Query**   | Vetor que representa a palavra atual sendo processada, usado para consultar informações relevantes no contexto. [1] |
| **Key**     | Vetor que codifica informações sobre palavras no contexto, usado para comparação com a query. [1] |
| **Value**   | Vetor que contém o conteúdo semântico real da palavra, usado para computar a saída da camada de atenção. [1] |
| **Atenção** | Mecanismo que permite ao modelo focar em partes relevantes do input ao processar uma sequência. [2] |

> ✔️ **Ponto de Destaque**: A decomposição em Query, Key e Value permite que o modelo aprenda diferentes aspectos das relações entre palavras de forma paralela e eficiente.

### Interpretação Geométrica dos Vetores QKV

<image: Um espaço vetorial 3D com vetores query, key e value representados como setas coloridas, mostrando ângulos e projeções entre eles>

A interpretação geométrica dos vetores QKV oferece insights valiosos sobre o funcionamento do mecanismo de atenção:

1. **Query como Direção de Busca**: O vetor query pode ser visto como uma direção no espaço vetorial que representa a informação que estamos buscando. [3]

2. **Key como Descritor de Conteúdo**: Os vetores key funcionam como descritores do conteúdo de cada palavra no contexto. [3]

3. **Value como Conteúdo Semântico**: Os vetores value carregam o conteúdo semântico real que será usado para computar a saída. [3]

4. **Produto Escalar como Similaridade**: A operação de produto escalar entre query e key mede a similaridade entre a informação buscada e o conteúdo disponível. [4]

Matematicamente, podemos expressar a atenção como:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Onde:
- $Q \in \mathbb{R}^{n \times d_k}$ é a matriz de queries
- $K \in \mathbb{R}^{n \times d_k}$ é a matriz de keys
- $V \in \mathbb{R}^{n \times d_v}$ é a matriz de values
- $d_k$ é a dimensão dos vetores query e key
- $n$ é o número de tokens na sequência

#### Questões Técnicas/Teóricas

1. Como a dimensionalidade dos vetores query e key ($d_k$) afeta a estabilidade numérica do cálculo de atenção, e por que a divisão por $\sqrt{d_k}$ é necessária?

2. Descreva geometricamente como o produto escalar entre um vetor query e um vetor key captura a relevância entre duas palavras no contexto.

### Papéis Distintos no Processo de Atenção

Os vetores QKV desempenham papéis complementares no processo de atenção:

1. **Query: Foco da Atenção**
   - Determina o que é relevante para a palavra atual
   - Projeta a "pergunta" que o modelo está fazendo ao contexto [5]

2. **Key: Índice de Conteúdo**
   - Fornece um "índice" para o conteúdo de cada palavra no contexto
   - Permite comparação eficiente com a query [5]

3. **Value: Informação Contextual**
   - Contém a informação real que será agregada
   - Representa o "conteúdo" que será ponderado pela atenção [5]

> ❗ **Ponto de Atenção**: A separação em QKV permite que o modelo aprenda diferentes transformações para cada aspecto da atenção, aumentando a flexibilidade e expressividade do mecanismo.

### Contribuição para Captura de Relações entre Palavras

Os vetores QKV permitem capturar diferentes tipos de relações entre palavras:

1. **Relações Sintáticas**: 
   - Query e Key podem aprender a capturar padrões sintáticos, como concordância sujeito-verbo. [6]
   - Exemplo: Em "The keys to the cabinet are on the table", a atenção pode focar em "keys" ao processar "are".

2. **Relações Semânticas**:
   - Value vectors podem codificar informações semânticas profundas. [6]
   - Exemplo: Em "The chicken crossed the road because it wanted to get to the other side", o modelo pode relacionar "it" com "chicken" baseado no conteúdo semântico.

3. **Relações de Longa Distância**:
   - A atenção permite capturar dependências de longa distância eficientemente. [7]
   - Exemplo: Em textos longos, informações relevantes de parágrafos anteriores podem ser acessadas diretamente.

#### Questões Técnicas/Teóricas

1. Como a multi-head attention contribui para a captura de diferentes tipos de relações entre palavras? Explique em termos dos papéis de QKV.

2. Proponha uma modificação no mecanismo de atenção que poderia melhorar a captura de relações hierárquicas em textos. Como isso afetaria os vetores QKV?

### Mecanismos Alternativos de Atenção

Além da atenção de produto escalar escalado, existem alternativas que exploram diferentes aspectos das relações entre palavras:

1. **Atenção Aditiva**:
   - Usa uma rede feed-forward para computar scores de atenção
   - Fórmula: $\text{score}(q, k) = v^T \tanh(W_q q + W_k k)$ [8]
   - Vantagem: Pode capturar relações não-lineares entre query e key

2. **Atenção Baseada em Distância**:
   - Incorpora informação de posição relativa no cálculo da atenção
   - Fórmula: $\text{score}(q, k, d) = q^T k + w^T f(d)$, onde $d$ é a distância e $f$ é uma função de codificação de distância [9]
   - Vantagem: Melhora a modelagem de dependências locais

3. **Atenção Esparsa**:
   - Limita a atenção a um subconjunto de posições baseado em algum critério (e.g., top-k)
   - Vantagem: Reduz complexidade computacional e pode focar em relações mais importantes [10]

> 💡 **Inovação**: Experimentos recentes mostram que combinar diferentes tipos de atenção em um único modelo pode levar a melhorias significativas na performance em várias tarefas de NLP.

### Implementação Avançada: Multi-Head Attention

A multi-head attention é uma extensão crucial do mecanismo básico de atenção, permitindo que o modelo aprenda múltiplas representações de atenção em paralelo. Vamos implementar uma versão simplificada usando PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        q = self.split_heads(self.W_q(query))
        k = self.split_heads(self.W_k(key))
        v = self.split_heads(self.W_v(value))
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        
        context = context.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_probs
```

Esta implementação demonstra como os papéis de Query, Key e Value são manipulados em um contexto de multi-head attention, permitindo que o modelo aprenda múltiplas representações de atenção simultaneamente.

### Conclusão

A decomposição do mecanismo de atenção em Query, Key e Value representa uma inovação fundamental nos transformers, permitindo uma modelagem rica e flexível das relações entre palavras. A interpretação geométrica desses vetores oferece insights valiosos sobre como o modelo captura diferentes aspectos do contexto linguístico. Mecanismos alternativos de atenção expandem ainda mais as possibilidades, abrindo caminhos para futuras inovações em arquiteturas de transformers e processamento de linguagem natural.

### Questões Avançadas

1. Como você projetaria um mecanismo de atenção que pudesse capturar explicitamente relações hierárquicas em textos (por exemplo, estrutura de árvore sintática)? Considere modificações nos papéis de QKV e na função de scoring.

2. Discuta as implicações computacionais e de modelagem de usar diferentes dimensionalidades para Query, Key e Value. Como isso poderia afetar a capacidade do modelo de capturar diferentes tipos de relações linguísticas?

3. Proponha e descreva matematicamente um novo mecanismo de atenção que combine aspectos da atenção baseada em produto escalar e da atenção aditiva. Quais seriam as potenciais vantagens deste híbrido?

### Referências

[1] "To capture these three different roles, transformers introduce weight matrices WQ, WK, and WV. These weights will be used to project each input vector xi into a representation of its role as a key, query, or value." (Trecho de Transformers and Large Language Models - Chapter 10)

[2] "The core intuition of attention is the idea of comparing an item of interest to a collection of other items in a way that reveals their relevance in the current context." (Trecho de Transformers and Large Language Models - Chapter 10)

[3] "Given these projections, the score between a current focus of attention, x, and an element in the preceding context, x, consists of a dot product between its query vector qi and the preceding element's key vectors k." (Trecho de Transformers and Large Language Models - Chapter 10)

[4] "score(x, x ) = qi · k ji j √dk" (Trecho de Transformers and Large Language Models - Chapter 10)

[5] "As the current focus of attention when being compared to all of the other preceding inputs. We'll refer to this role as a query. In its role as a preceding input being compared to the current focus of attention. We'll refer to this role as a key. And finally, as a value used to compute the output for the current focus of attention." (Trecho de Transformers and Large Language Models - Chapter 10)

[6] "In (10.1), the phrase The keys is the subject of the sentence, and in English and many languages, must agree in grammatical number with the verb are; in this case both are plural. In English we can't use a singular verb like is with a plural subject like keys; we'll discuss agreement more in Chapter 17. In (10.2), the pronoun it corefers to the chicken; it's the chicken that wants to get to the other side." (Trecho de Transformers and Large Language Models - Chapter 10)

[7] "These helpful contextual words can be quite far way in the sentence or paragraph." (Trecho de Transformers and Large Language Models - Chapter 10)

[8] "To capture these three different roles, transformers introduce weight matrices WQ, WK, and WV. These weights will be used to project each input vector xi into a representation of its role as a key, query, or value." (Trecho de Transformers and Large Language Models - Chapter 10)

[9] "Even more complex positional embedding methods exist, such as ones that represent relative position instead of absolute position, often implemented in the attention mechanism at each layer rather than being added once at the initial input." (Trecho de Transformers and Large Language Models - Chapter 10)

[10] "Transformers address this issue with multihead self-attention layers. These are sets of self-attention layers, called heads, that reside in parallel layers at the same depth in a model, each with its own set of parameters. By using these distinct sets of parameters, each head can learn different aspects of the relationships among inputs at the same level of abstraction." (Trecho de Transformers and Large Language Models - Chapter 10)