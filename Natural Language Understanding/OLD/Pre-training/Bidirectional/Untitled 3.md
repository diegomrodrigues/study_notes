## Self-Attention Mechanism in Bidirectional Models

<image: Uma visualização do mecanismo de self-attention em um modelo bidirecional, mostrando como cada token pode atender a todos os outros tokens na sequência, incluindo os futuros. A imagem deve destacar as conexões bidirecionais entre os tokens, enfatizando a ausência de mascaramento.>

### Introdução

O mecanismo de self-attention é um componente fundamental dos modelos de linguagem baseados em transformers, permitindo que esses modelos capturem dependências de longo alcance em sequências de texto. Nos modelos bidirecionais, como o BERT (Bidirectional Encoder Representations from Transformers), a self-attention opera de maneira diferente dos modelos causais (unidirecionais), permitindo que cada token atenda a todos os outros tokens na sequência, incluindo os que estão à frente [1]. Esta abordagem bidirecional permite uma contextualização mais rica e completa das representações de palavras, crucial para tarefas como compreensão de linguagem natural e classificação de texto.

### Conceitos Fundamentais

| Conceito              | Explicação                                                   |
| --------------------- | ------------------------------------------------------------ |
| **Self-Attention**    | Mecanismo que permite a um modelo pesar a importância de diferentes partes de uma sequência de entrada para cada elemento dessa sequência. Em modelos bidirecionais, cada token pode atender a todos os outros tokens, independentemente de sua posição. [1] |
| **Bidirecionalidade** | Característica que permite ao modelo processar a entrada em ambas as direções (da esquerda para a direita e vice-versa), permitindo uma compreensão mais completa do contexto. [1] |
| **Contextualização**  | Processo pelo qual as representações de palavras são ajustadas com base no contexto circundante, incluindo palavras à esquerda e à direita na sequência. [1] |

> ⚠️ **Nota Importante**: A ausência de mascaramento futuro em modelos bidirecionais é crucial para permitir a contextualização completa, mas torna esses modelos inadequados para tarefas de geração autoregressive.

### Funcionamento da Self-Attention Bidirecional

<image: Um diagrama detalhado mostrando o fluxo de informações em uma camada de self-attention bidirecional, destacando as matrizes de Query (Q), Key (K) e Value (V), bem como a matriz de atenção resultante.>

O mecanismo de self-attention em modelos bidirecionais opera permitindo que cada token na sequência interaja com todos os outros tokens, incluindo aqueles que viriam depois em uma leitura da esquerda para a direita. Este processo pode ser descrito matematicamente e implementado de forma eficiente usando operações matriciais [2].

1. **Geração de embeddings Q, K, V**:
   Para cada token de entrada $x_i$, são gerados três vetores através de projeções lineares:
   
   $$
   q_i = W^Qx_i; \quad k_i = W^Kx_i; \quad v_i = W^Vx_i
   $$

   onde $W^Q, W^K, W^V$ são matrizes de peso aprendidas. [1]

2. **Cálculo dos scores de atenção**:
   Para cada par de tokens $(i, j)$, um score de atenção é calculado:
   
   $$
   \text{score}_{ij} = q_i \cdot k_j
   $$

   Estes scores são organizados em uma matriz $QK^T$. [2]

3. **Aplicação do softmax**:
   Os scores são normalizados usando softmax para obter os pesos de atenção:
   
   $$
   \alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{k=1}^n \exp(\text{score}_{ik})}
   $$

4. **Computação da saída**:
   A saída para cada posição é uma soma ponderada dos valores:
   
   $$
   y_i = \sum_{j=1}^n \alpha_{ij} v_j
   $$

> ✔️ **Ponto de Destaque**: A operação completa de self-attention pode ser expressa de forma concisa usando notação matricial:

$$
\text{SelfAttention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

onde $d_k$ é a dimensão das chaves, usado para escalar os scores e estabilizar o gradiente. [2]

#### Questões Técnicas/Teóricas

1. Como a ausência de mascaramento futuro na self-attention bidirecional afeta a capacidade do modelo de capturar contexto em comparação com modelos unidirecionais?
2. Explique como a normalização dos scores de atenção através do softmax contribui para a estabilidade e eficácia do mecanismo de self-attention.

### Implementação Eficiente da Self-Attention Bidirecional

A implementação eficiente da self-attention bidirecional é crucial para o desempenho dos modelos. Utilizando operações matriciais, podemos processar toda a sequência de entrada de uma vez, aproveitando o paralelismo dos hardwares modernos.

````python
import torch
import torch.nn.functional as F

def self_attention(query, key, value):
    # Assumindo que query, key, value têm shape (batch_size, seq_len, d_model)
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Compute output
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights
````

Este código implementa a operação de self-attention de forma vetorizada, permitindo o processamento paralelo de toda a sequência. A divisão por $\sqrt{d_k}$ é uma técnica de escalonamento para evitar gradientes muito pequenos durante o treinamento [2].

> ❗ **Ponto de Atenção**: A implementação acima não inclui mascaramento, permitindo que cada token atenda a todos os outros, incluindo os futuros, característica essencial dos modelos bidirecionais.

### Impacto da Bidirecionalidade nas Representações Contextuais

A natureza bidirecional da self-attention em modelos como BERT permite a criação de representações contextuais mais ricas. Cada token pode incorporar informações de todo o contexto, tanto precedente quanto subsequente, resultando em embeddings que capturam nuances semânticas de forma mais completa [1].

#### Vantagens e Desvantagens da Self-Attention Bidirecional

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Captura de contexto mais rico, incorporando informações de toda a sequência [1] | Não adequado para tarefas de geração autoregressive [3]      |
| Melhora significativa em tarefas de compreensão de linguagem e classificação [1] | Maior complexidade computacional, pois cada token atende a todos os outros [2] |
| Permite a criação de representações de palavras sensíveis ao contexto bidirecional [1] | Pode introduzir "vazamento de informação" em certas tarefas que requerem causalidade [3] |

### Aplicações e Implicações

A self-attention bidirecional tem implicações profundas para várias tarefas de processamento de linguagem natural:

1. **Classificação de Sequências**: A capacidade de considerar o contexto completo melhora significativamente a precisão em tarefas como análise de sentimento e classificação de tópicos [4].

2. **Preenchimento de Mascaramento**: Em tarefas como o Masked Language Modeling (MLM), a bidirecionalidade permite prever palavras mascaradas com base em todo o contexto circundante [5].

3. **Resposta a Perguntas**: A compreensão bidirecional do contexto é crucial para localizar e extrair informações relevantes de um texto para responder perguntas [4].

4. **Representações Contextuais**: As embeddings resultantes são altamente contextualizadas, capturando nuances semânticas que dependem do contexto completo da frase [1].

> 💡 **Insight**: A bidirecionalidade na self-attention é um dos principais fatores que permitem a modelos como BERT superar modelos unidirecionais em uma ampla gama de tarefas de compreensão de linguagem.

#### Questões Técnicas/Teóricas

1. Como você adaptaria o mecanismo de self-attention bidirecional para uma tarefa que requer causalidade parcial, onde alguns tokens futuros são permitidos, mas não todos?
2. Discuta as implicações computacionais e de desempenho de usar self-attention bidirecional em sequências muito longas. Que estratégias poderiam ser empregadas para mitigar possíveis limitações?

### Conclusão

O mecanismo de self-attention bidirecional representa um avanço significativo na modelagem de linguagem, permitindo a criação de representações contextuais ricas que capturam dependências complexas em ambas as direções em uma sequência de texto. Esta abordagem, exemplificada por modelos como BERT, tem demonstrado sucesso notável em uma ampla gama de tarefas de processamento de linguagem natural, superando consistentemente abordagens unidirecionais anteriores [1][4].

A capacidade de cada token atender a todos os outros na sequência, sem restrições de direcionalidade, permite uma compreensão mais profunda e nuançada do contexto linguístico. Isso se traduz em melhorias significativas em tarefas que requerem uma compreensão holística do texto, como classificação de documentos, resposta a perguntas e análise de sentimentos [4].

No entanto, é importante reconhecer que a bidirecionalidade também introduz limitações, particularmente em cenários que requerem geração de texto autoregressive. A busca por modelos que possam combinar as vantagens da compreensão bidirecional com a capacidade de geração eficaz continua sendo uma área ativa de pesquisa no campo do processamento de linguagem natural [3].

À medida que a field evolui, é provável que vejamos refinamentos adicionais e novas arquiteturas que busquem equilibrar as vantagens da atenção bidirecional com as necessidades específicas de diferentes tarefas de linguagem, potencialmente levando a modelos ainda mais versáteis e poderosos no futuro.

### Questões Avançadas

1. Considerando as limitações da self-attention bidirecional para tarefas de geração, como você projetaria um modelo híbrido que pudesse alternar eficientemente entre modos de compreensão bidirecional e geração unidirecional?

2. Analise criticamente o impacto computacional e de memória da self-attention bidirecional em sequências muito longas. Proponha e discuta possíveis modificações arquitetônicas para tornar o mecanismo mais escalável para textos extremamente longos, mantendo a capacidade de capturar dependências de longo alcance.

3. Considerando o conceito de "atenção esparsa", onde nem todos os tokens precisam atender a todos os outros, como você modificaria o mecanismo de self-attention bidirecional para implementar uma forma de atenção esparsa adaptativa que pudesse aprender automaticamente quais conexões são mais importantes em diferentes contextos?

### Referências

[1] "Bidirectional encoders overcome this limitation by allowing the self-attention mechanism to range over the entire input, as shown in Fig. 11.1b." (Trecho de Fine-Tuning and Masked Language Models)

[2] "As with causal transformers, since each output vector, y_i, is computed independently, the processing of an entire sequence can be parallelized via matrix operations. The first step is to pack the input embeddings x_i into a matrix X ∈ ℝ^(N×d_k). That is, each row of X is the embedding of one token of the input. We then multiply X by the key, query, and value weight matrices (all of dimensionality d × d) to produce matrices Q ∈ ℝ^(N×d), K ∈ ℝ^(N×d), and V ∈ ℝ^(N×d), containing all the key, query, and value vectors in a single step." (Trecho de Fine-Tuning and Masked Language Models)

[3] "The key architecture difference is in bidirectional models we don't mask the future. As shown in Fig. 11.2, the full set of self-attention scores represented by QK^T constitute an all-pairs comparison between the keys and queries for each element of the input. In the case of causal language models in Chapter 10, we masked the upper triangular portion of this matrix (in Fig. ??) to eliminate information about future words since this would make the language modeling training task trivial. With bidirectional encoders we simply skip the mask, allowing the model to contextualize each token using information from the entire input." (Trecho de Fine-Tuning and Masked Language Models)

[4] "To make this more concrete, the original English-only bidirectional transformer encoder model, BERT (Devlin et al., 2019), consisted of the following:" (Trecho de Fine-Tuning and Masked Language Models)

[5] "MLM training objective is to predict the original inputs for each of the masked tokens using a bidirectional encoder of the kind described in the last section. The cross-entropy loss from these predictions drives the training process for all the parameters in the model. Note that all of the input tokens play a role in the self-attention process, but only the sampled tokens are used for learning." (Trecho de Fine-Tuning and Masked Language Models)