## Scaled Dot-Product Attention: Fundamentos, Formulação Matemática e Impacto na Estabilidade de Modelos

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240829113857364.png" alt="image-20240829113857364" style="zoom:67%;" />

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240829113918302.png" alt="image-20240829113918302" style="zoom:67%;" />

### Introdução

A **Scaled Dot-Product Attention** é um componente fundamental dos modelos Transformer, revolucionando o processamento de sequências em tarefas de aprendizado profundo, especialmente em processamento de linguagem natural (NLP) [1]. Este mecanismo permite que os modelos foquem seletivamente em diferentes partes da entrada, melhorando significativamente a capacidade de capturar dependências de longo alcance e relações complexas entre os elementos de uma sequência.

Este resumo explora em profundidade a formulação matemática da ==atenção por produto escalar escalado==, sua motivação, implementação e ==impacto na estabilidade e desempenho dos modelos==. Analisaremos também como ==diferentes fatores de escala podem influenciar o comportamento e a eficácia dos modelos Transformer.==

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Dot-Product Attention** | Mecanismo de atenção baseado no ==produto escalar entre queries e keys== para computar scores de atenção. [2] |
| **Scaling Factor**        | ==Fator introduzido para mitigar instabilidades numéricas em dimensões elevadas, tipicamente √dk. [2]== |
| **Softmax**               | Função aplicada aos scores de atenção para obter pesos normalizados. [2] |

> ⚠️ **Nota Importante**: A escala no dot-product attention é crucial para ==manter gradientes estáveis durante o treinamento==, especialmente em modelos com alta dimensionalidade.

### Formulação Matemática do Scaled Dot-Product Attention

A atenção por produto escalar escalado é definida matematicamente como [2]:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Onde:
- $Q \in \mathbb{R}^{n \times d_k}$: matriz de queries
- $K \in \mathbb{R}^{m \times d_k}$: matriz de keys
- $V \in \mathbb{R}^{m \times d_v}$: matriz de values
- $d_k$: ==dimensão das queries e keys==
- $d_v$: dimensão dos values
- $n$: número de queries
- $m$: número de key-value pairs

Vamos analisar cada componente desta fórmula:

1. **Produto Matricial $QK^T$**: 
   - ==Computa os scores de atenção brutos entre cada query e key.==
   - Resultado: matriz $n \times m$ de scores de atenção.

2. **Fator de Escala $\frac{1}{\sqrt{d_k}}$**:
   - ==Mitiga o problema de gradientes pequenos em dimensões elevadas.==
   - ==Mantém a variância dos scores de atenção aproximadamente constante==, independentemente de $d_k$.

3. **Softmax**:
   - Normaliza os scores de atenção escalados.
   - Transforma scores em pesos de atenção entre 0 e 1, somando 1 para cada query.

4. **Multiplicação por $V$**:
   - Pondera os values pelos pesos de atenção normalizados.
   - Produz a saída final da camada de atenção.

> ✔️ **Ponto de Destaque**: ==A escala $\frac{1}{\sqrt{d_k}}$ é crucial para evitar que o argumento do softmax cresça descontroladamente com $d_k$==, o que levaria a gradientes extremamente pequenos.

#### Demonstração da Necessidade de Escala

![image-20240829113115775](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240829113115775.png)

Para entender por que a escala é necessária, consideremos o comportamento do produto escalar em alta dimensão:

1. Assuma que elementos de $Q$ e $K$ são variáveis aleatórias independentes com média 0 e variância 1.
2. O produto escalar $q \cdot k$ para uma query $q$ e uma key $k$ tem:
   - $\mathbb{E}[q \cdot k] = 0$
   - $\text{Var}(q \cdot k) = d_k$

3. Sem escala, ==à medida que $d_k$ aumenta, a variância do produto escalar cresce linearmente==, levando a valores extremos no softmax.

4. Aplicando a escala $\frac{1}{\sqrt{d_k}}$:
   - $\mathbb{E}[\frac{q \cdot k}{\sqrt{d_k}}] = 0$
   - $\text{Var}(\frac{q \cdot k}{\sqrt{d_k}}) = 1$

Isso mantém a variância constante, independentemente de $d_k$, estabilizando o treinamento.

#### Questões Técnicas/Teóricas

1. Como a variância dos scores de atenção muda se usarmos um fator de escala $\frac{1}{d_k}$ em vez de $\frac{1}{\sqrt{d_k}}$? Explique o impacto potencial no treinamento do modelo.

2. Se aumentarmos $d_k$ em um modelo Transformer, mantendo outras dimensões constantes, como isso afetará o número de parâmetros e o custo computacional da camada de atenção? Justifique matematicamente.

### Implementação em PyTorch

Vamos implementar a função de Scaled Dot-Product Attention em PyTorch:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights
```

Esta implementação:
1. Calcula os scores de atenção com o produto escalar escalado.
2. Aplica uma máscara opcional (útil para atenção causal em decodificadores).
3. Normaliza os scores com softmax.
4. Computa a saída ponderada e retorna também os pesos de atenção.

> ❗ **Ponto de Atenção**: A aplicação da máscara com `-1e9` antes do softmax efetivamente zera a atenção para posições mascaradas.

### Impacto de Diferentes Fatores de Escala

O fator de escala $\frac{1}{\sqrt{d_k}}$ foi escolhido empiricamente por Vaswani et al. (2017) [2], mas é interessante considerar o impacto de diferentes fatores:

1. **Sem Escala ($\frac{1}{1}$)**:
   - 👎 Gradientes extremamente pequenos para $d_k$ grande.
   - 👎 Treinamento instável e convergência lenta.

2. **Escala Padrão ($\frac{1}{\sqrt{d_k}}$)**:
   - 👍 Balanceia bem a magnitude dos gradientes.
   - 👍 Funciona bem para uma ampla gama de arquiteturas.

3. **Escala Quadrática ($\frac{1}{d_k}$)**:
   - 👍 Pode ser benéfica para dimensões muito altas.
   - 👎 Risco de subamortecer o sinal para dimensões menores.

4. **Escala Adaptativa**:
   - Ideia: Ajustar o fator de escala durante o treinamento.
   - 👍 ==Potencial para melhor adaptação a diferentes regimes.==
   - 👎 ==Aumenta a complexidade do modelo e pode ser instável.==

> 💡 **Insight**: A escolha do fator de escala ideal pode depender da arquitetura específica e da tarefa. Experimentos empíricos são cruciais para determinar o melhor fator para um dado modelo.

#### Análise Matemática do Impacto da Escala

![image-20240829114439625](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240829114439625.png)

Consideremos o gradiente da função softmax em relação aos scores de atenção:

$$
\frac{\partial \text{softmax}(x_i)}{\partial x_j} = \text{softmax}(x_i)(\delta_{ij} - \text{softmax}(x_j))
$$

Onde $\delta_{ij}$ é o delta de Kronecker.

Para scores de atenção $s = \frac{QK^T}{\alpha}$, onde $\alpha$ é o fator de escala:

1. Se $\alpha$ for muito pequeno (ou ausente), $s$ terá valores grandes, levando a:
   - ==softmax(s) próximo a one-hot vectors.==
   - Gradientes próximos a zero, dificultando o aprendizado.

2. Se $\alpha$ for muito grande, $s$ terá valores pequenos, resultando em:
   - ==softmax(s) próximo a distribuição uniforme.==
   - Gradientes pequenos, mas não tão próximos de zero.

3. Com $\alpha = \sqrt{d_k}$:
   - ==Mantém $s$ em uma faixa que permite gradientes informativos.==
   - ==Facilita o fluxo de gradientes através da rede.==

#### Questões Técnicas/Teóricas

1. Como o fator de escala afeta a saturação da função softmax? Descreva matematicamente como isso impacta os gradientes durante o backpropagation.

2. Se quiséssemos implementar um fator de escala adaptativo, que depende da distribuição dos valores em Q e K, como poderíamos formular isso? Considere usar estatísticas como a variância dos elementos de QK^T.

### Conclusão

A Scaled Dot-Product Attention é um componente crucial dos modelos Transformer, permitindo uma atenção eficiente e estável sobre sequências de entrada. O fator de escala $\frac{1}{\sqrt{d_k}}$ desempenha um papel fundamental na estabilização do treinamento, especialmente para modelos de alta dimensionalidade [2].

Embora $\frac{1}{\sqrt{d_k}}$ seja o padrão, a escolha do fator de escala pode impactar significativamente o desempenho e a estabilidade do modelo. Pesquisas futuras podem explorar fatores de escala adaptativos ou específicos para determinadas arquiteturas ou tarefas.

A compreensão profunda da matemática por trás da atenção escalada é essencial para o desenvolvimento e otimização de modelos Transformer avançados, abrindo caminho para inovações em arquiteturas de atenção e suas aplicações em diversos domínios do aprendizado profundo.

### Questões Avançadas

1. Considere um modelo Transformer com múltiplas cabeças de atenção. Como a escolha de diferentes fatores de escala para cada cabeça poderia afetar a capacidade do modelo de capturar diferentes tipos de relações? Proponha um método para determinar fatores de escala ótimos para cada cabeça.

2. Na prática, observa-se que alguns modelos Transformer muito grandes (como GPT-3) usam fatores de escala ligeiramente diferentes de $\sqrt{d_k}$. Desenvolva uma análise teórica que possa explicar por que desvios do fator padrão podem ser benéficos em certos regimes de escala.

3. O mecanismo de atenção pode ser visto como uma forma de recuperação de informação soft. Como a escolha do fator de escala afeta esta interpretação? Relacione sua resposta com conceitos de teoria da informação, como entropia e informação mútua entre queries e keys.

### Referências

[1] "Transformers são uma arquitetura de rede neural que revolucionou o processamento de sequências em tarefas de aprendizado profundo, especialmente em processamento de linguagem natural (NLP)." (Trecho de Transformers and Large Language Models - Chapter 10)

[2] "The core intuition of attention is the idea of comparing an item of interest to a collection of other items in a way that reveals their relevance in the current context. In the case of self-attention for language, the set of comparisons are to other words (or tokens) within a given sequence. The result of these comparisons is then used to compute an output sequence for the current input sequence." (Trecho de Transformers and Large Language Models - Chapter 10)

[3] "To capture these three different roles, transformers introduce weight matrices WQ, WK, and WV. These weights will be used to project each input vector xi into a representation of its role as a key, query, or value." (Trecho de Transformers and Large Language Models - Chapter 10)

[4] "Given these projections, the score between a current focus of attention, x, and an element in the preceding context, x, consists of a dot product between its query vector qi and the preceding element's key vectors k. This dot product has the right shape since both the query and the key are of dimensionality 1 × d." (Trecho de Transformers and Large Language Models - Chapter 10)

[5] "There is one final part of the self-attention model. The result of a dot product can be an arbitrarily large (positive or negative) value. Exponentiating large values can lead to numerical issues and to an effective loss of gradients during training. To avoid this, we scale down the result of the dot product, by dividing it by a factor related to the size of the embeddings. A typical approach is to divide by the square root of the dimensionality of the query and key vectors (dk)" (Trecho de Transformers and Large Language Models - Chapter 10)

[6] "score(x, x ) = qi · k ji j √dk" (Trecho de Transformers and Large Language Models - Chapter 10)

[7] "A = SelfAttention(Q, K, V) = softmax(QKᵀ)V√dk" (Trecho de Transformers and Large Language Models - Chapter 10)

[8] "The values of Nc, Dc, Cc, αN, αD, and αC depend on the exact transformer architecture, tokenization, and vocabulary size, so rather than all the precise values, scaling laws focus on the relationship with loss." (Trecho de Transformers and Large Language Models - Chapter 10)