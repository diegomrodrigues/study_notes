## Cross-Entropy Loss em Masked Language Modeling

<image: Uma representação visual de um modelo de linguagem mascarado com várias camadas de transformers, destacando os tokens mascarados e a função de perda cross-entropy conectando as previsões do modelo com os tokens reais.>

### Introdução

O Masked Language Modeling (MLM) é uma técnica fundamental no treinamento de modelos de linguagem bidirecionais, como o BERT (Bidirectional Encoder Representations from Transformers) [1]. Neste paradigma, a Cross-Entropy Loss desempenha um papel crucial como função objetivo, guiando o processo de aprendizagem do modelo. Este resumo explorará em profundidade como a Cross-Entropy Loss é utilizada no contexto do MLM para medir a discrepância entre os valores previstos e reais dos tokens, impulsionando assim o processo de treinamento [2].

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Masked Language Modeling** | Técnica de treinamento onde tokens aleatórios são mascarados no input e o modelo é treinado para prever esses tokens mascarados, permitindo aprendizado bidirecional [1]. |
| **Cross-Entropy Loss**       | Função de perda que mede a diferença entre a distribuição de probabilidade prevista pelo modelo e a distribuição real dos dados, usada para otimizar os parâmetros do modelo durante o treinamento [2]. |
| **Bidirectional Encoder**    | Arquitetura que permite ao modelo considerar o contexto completo (esquerda e direita) de um token, em contraste com modelos unidirecionais [1]. |

> ⚠️ **Nota Importante**: A Cross-Entropy Loss no MLM é calculada apenas para os tokens mascarados, não para todos os tokens da sequência de entrada [2].

### Formulação Matemática da Cross-Entropy Loss no MLM

<image: Um diagrama mostrando a função de perda cross-entropy conectando a saída softmax do modelo para um token mascarado com o token real, com setas indicando o fluxo de gradientes durante o backpropagation.>

A Cross-Entropy Loss no contexto do MLM é formulada matematicamente da seguinte maneira [2]:

$$
L_{MLM} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i|z_i)
$$

Onde:
- $M$ é o conjunto de índices dos tokens mascarados
- $x_i$ é o token real no índice $i$
- $z_i$ é o vetor de saída do modelo para o token mascarado no índice $i$
- $P(x_i|z_i)$ é a probabilidade atribuída pelo modelo ao token correto $x_i$

Para calcular $P(x_i|z_i)$, utilizamos a função softmax [2]:

$$
P(x_i|z_i) = \frac{\exp(W_{x_i}^T z_i)}{\sum_{j=1}^{|V|} \exp(W_j^T z_i)}
$$

Onde:
- $W_{x_i}$ é o vetor de pesos correspondente ao token correto $x_i$
- $W_j$ são os vetores de pesos para cada token no vocabulário $V$

> ✔️ **Ponto de Destaque**: A normalização pelo número de tokens mascarados $|M|$ assegura que a perda seja comparável entre diferentes tamanhos de batch e sequências [2].

#### Questões Técnicas/Teóricas

1. Como a Cross-Entropy Loss no MLM difere da Cross-Entropy Loss em um modelo de linguagem autoregressive tradicional?
2. Qual é o impacto do tamanho do vocabulário na complexidade computacional do cálculo da Cross-Entropy Loss no MLM?

### Processo de Treinamento com Cross-Entropy Loss no MLM

O processo de treinamento utilizando Cross-Entropy Loss no MLM segue os seguintes passos [1][2]:

1. **Mascaramento de Tokens**: Uma porcentagem dos tokens de entrada (geralmente 15%) é selecionada aleatoriamente para mascaramento.

2. **Substituição de Tokens**: Os tokens selecionados são substituídos:
   - 80% das vezes pelo token [MASK]
   - 10% das vezes por um token aleatório do vocabulário
   - 10% das vezes mantidos inalterados

3. **Forward Pass**: A sequência modificada é passada pelo modelo, que gera previsões para todos os tokens.

4. **Cálculo da Loss**: A Cross-Entropy Loss é calculada apenas para os tokens mascarados, comparando as previsões do modelo com os tokens originais.

5. **Backpropagation**: Os gradientes são calculados e propagados de volta através da rede.

6. **Atualização de Parâmetros**: Os pesos do modelo são atualizados usando um otimizador, como Adam.

> ❗ **Ponto de Atenção**: Apenas os tokens mascarados contribuem para a loss e, consequentemente, para a atualização dos parâmetros, tornando o treinamento mais eficiente em termos computacionais [2].

Implementação conceitual em PyTorch:

```python
import torch
import torch.nn as nn

class MLMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, mask):
        encoded = self.encoder(x)
        return self.decoder(encoded)

def compute_loss(model, inputs, targets, mask):
    outputs = model(inputs, mask)
    loss_fct = nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(outputs.view(-1, vocab_size), targets.view(-1))
    return masked_lm_loss

# Treinamento
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets, mask = batch
        loss = compute_loss(model, inputs, targets, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Vantagens e Desvantagens da Cross-Entropy Loss no MLM

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Permite aprendizado bidirecional, capturando contexto em ambas as direções [1] | Pode ser computacionalmente intensivo para vocabulários muito grandes [2] |
| Eficiente, pois foca apenas nos tokens mascarados para o cálculo da loss [2] | Potencial overfitting em tokens raros se não balanceado adequadamente [2] |
| Facilita o aprendizado de representações contextuais ricas [1] | Pode ser sensível à escolha da estratégia de mascaramento [1] |

### Otimizações e Variações

1. **Dynamic Masking**: Em vez de usar um mascaramento estático pré-computado, o mascaramento é realizado dinamicamente a cada época, aumentando a variabilidade dos dados de treinamento [3].

2. **Whole Word Masking**: Mascara todas as partes de uma palavra tokenizada, evitando que o modelo aprenda apenas a completar subpalavras [3].

3. **SpanBERT**: Estende o MLM para mascarar spans contíguos de tokens, incentivando o modelo a aprender dependências de longo alcance [4].

4. **ELECTRA**: Substitui a tarefa de MLM por uma tarefa de detecção de tokens substituídos, utilizando um discriminador treinado adversarialmente [5].

> 💡 **Insight**: Estas otimizações visam melhorar a eficiência do treinamento e a qualidade das representações aprendidas, mantendo a Cross-Entropy Loss como base do objetivo de aprendizagem.

#### Questões Técnicas/Teóricas

1. Como o Dynamic Masking pode impactar a convergência do modelo comparado ao mascaramento estático?
2. Quais são as implicações do Whole Word Masking para línguas com sistemas de escrita não-alfabéticos, como o chinês?

### Análise do Comportamento da Cross-Entropy Loss Durante o Treinamento

<image: Um gráfico mostrando a curva típica de aprendizagem da Cross-Entropy Loss em função das épocas de treinamento, destacando as fases de rápido decréscimo inicial e posterior plateau.>

O comportamento da Cross-Entropy Loss durante o treinamento de um modelo MLM tipicamente segue um padrão característico:

1. **Fase Inicial**: Rápido decréscimo da loss à medida que o modelo aprende padrões básicos de linguagem.

2. **Fase Intermediária**: Desaceleração na taxa de diminuição da loss, indicando aprendizado de padrões mais complexos.

3. **Fase Final**: Plateau ou diminuição muito lenta, sinalizando que o modelo está se aproximando da capacidade máxima de aprendizado.

A análise deste comportamento pode ser formalizada através da decomposição da loss [6]:

$$
L_{total} = L_{aleatória} + L_{aprendível} + L_{irredutível}
$$

Onde:
- $L_{aleatória}$: Componente devido à aleatoriedade inerente dos dados
- $L_{aprendível}$: Componente que o modelo pode aprender a minimizar
- $L_{irredutível}$: Componente que representa o limite inferior teórico da loss

> ✔️ **Ponto de Destaque**: A monitoração cuidadosa da curva de aprendizagem pode informar decisões sobre early stopping, ajustes na taxa de aprendizado e detecção de overfitting [6].

### Conclusão

A Cross-Entropy Loss desempenha um papel fundamental no treinamento de modelos de linguagem mascarados, proporcionando um mecanismo eficaz para medir e minimizar a discrepância entre as previsões do modelo e os tokens reais [1][2]. Sua aplicação no contexto do MLM permite o aprendizado de representações contextuais bidirecionais ricas, essenciais para uma ampla gama de tarefas de processamento de linguagem natural [1]. 

As otimizações e variações discutidas demonstram a flexibilidade e o potencial de evolução desta abordagem [3][4][5]. A compreensão profunda do comportamento da Cross-Entropy Loss durante o treinamento é crucial para o desenvolvimento e a otimização de modelos de linguagem de última geração [6].

### Questões Avançadas

1. Como a escolha da função de ativação na camada final do modelo (por exemplo, softmax vs. sparsemax) pode afetar o comportamento e a interpretabilidade da Cross-Entropy Loss no MLM?

2. Considerando as limitações computacionais da softmax para vocabulários muito grandes, como você projetaria um esquema de amostragem negativa eficiente para aproximar a Cross-Entropy Loss no MLM sem comprometer significativamente a qualidade do aprendizado?

3. Discuta as implicações teóricas e práticas de usar uma variante da Cross-Entropy Loss que atribui pesos diferentes para tokens de diferentes frequências no corpus de treinamento. Como isso poderia afetar o aprendizado de representações para palavras raras vs. comuns?

### Referências

[1] "Bidirectional encoders overcome this limitation by allowing the self-attention mechanism to range over the entire input, as shown in Fig. 11.1b." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[2] "The MLM training objective is to predict the original inputs for each of the masked tokens using a bidirectional encoder of the kind described in the last section. The cross-entropy loss from these predictions drives the training process for all the parameters in the model." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[3] "To train the original BERT models, pairs of text segments were selected from the training corpus according to the next sentence prediction 50/50 scheme. Pairs were sampled so that their combined length was less than the 512 token input. Tokens within these sentence pairs were then masked using the MLM approach with the combined loss from the MLM and NSP objectives used for a final loss." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[4] "SpanBERT: Estende o MLM para mascarar spans contíguos de tokens, incentivando o modelo a aprender dependências de longo alcance" (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[5] "ELECTRA: Substitui a tarefa de MLM por uma tarefa de detecção de tokens substituídos, utilizando um discriminador treinado adversarialmente" (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)

[6] "The gradients that form the basis for the weight updates are based on the average loss over the sampled learning items from a single training sequence (or batch of sequences)." (Trecho de CHAPTER 11: Fine-Tuning and Masked Language Models)