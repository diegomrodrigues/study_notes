## Arquiteturas Encoder-only: Compreendendo o Papel dos Transformers Bidirecionais como Codificadores

<image: Um diagrama mostrando a arquitetura de um transformer bidirecional, destacando o fluxo de informação em ambas as direções e a ausência de um componente decoder>

### Introdução

As arquiteturas encoder-only, baseadas em transformers bidirecionais, representam um avanço significativo no processamento de linguagem natural (NLP). Diferentemente dos modelos causais ou autoregressive que vimos anteriormente, essas arquiteturas se concentram na produção de representações contextualizadas ao invés da geração de texto [1]. Este resumo explora em profundidade o funcionamento, as aplicações e as implicações dessas arquiteturas, com foco especial no modelo BERT (Bidirectional Encoder Representations from Transformers) e suas variantes.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Transformer Bidirecional** | Arquitetura que permite que o modelo atenda a todo o contexto de entrada, tanto à esquerda quanto à direita, para cada token [1]. |
| **Masked Language Modeling** | Técnica de treinamento onde o modelo aprende a prever tokens mascarados em uma sequência, permitindo aprendizado bidirecional [1]. |
| **Embeddings Contextuais**   | Representações vetoriais de palavras que variam de acordo com o contexto em que aparecem, capturando nuances semânticas [3]. |
| **Fine-tuning**              | Processo de adaptar um modelo pré-treinado para tarefas específicas, ajustando seus pesos com dados rotulados para a tarefa em questão [1]. |

> ✔️ **Ponto de Destaque**: As arquiteturas encoder-only são projetadas para criar representações ricas e contextualizadas do texto de entrada, sendo particularmente eficazes em tarefas que requerem compreensão profunda do contexto.

### Arquitetura do Transformer Bidirecional

<image: Um diagrama detalhado da arquitetura interna de um bloco transformer bidirecional, mostrando as camadas de atenção, normalização e feed-forward>

A arquitetura do transformer bidirecional é uma evolução dos transformers originais, otimizada para a criação de representações contextuais. Vamos explorar seus componentes principais:

1. **Camada de Embedding**: 
   - Converte tokens de entrada em vetores densos.
   - Combina embeddings de tokens com embeddings posicionais.

2. **Blocos de Transformer**:
   - Múltiplas camadas de self-attention e feed-forward networks.
   - A self-attention permite que cada token atenda a todos os outros tokens na sequência.

3. **Camada de Saída**:
   - Produz representações contextuais para cada token de entrada.

A operação chave é a self-attention, definida matematicamente como [1]:

$$
\text{SelfAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Onde:
- $Q$, $K$, e $V$ são as matrizes de Query, Key e Value, respectivamente.
- $d_k$ é a dimensão das chaves.

Esta fórmula permite que o modelo pese a importância de diferentes partes da sequência de entrada para cada token, criando representações ricas e contextuais.

> ❗ **Ponto de Atenção**: Diferentemente dos modelos causais, os transformers bidirecionais não usam mascaramento para prevenir o acesso a tokens futuros, permitindo uma contextualização completa.

#### Questões Técnicas/Teóricas

1. Como a ausência de mascaramento na self-attention afeta a capacidade do modelo de capturar dependências de longo alcance em comparação com modelos causais?
2. Descreva como você ajustaria a arquitetura do transformer bidirecional para lidar com sequências de entrada muito longas, mantendo a eficiência computacional.

### Treinamento com Masked Language Modeling (MLM)

O treinamento de modelos encoder-only como o BERT utiliza principalmente a técnica de Masked Language Modeling (MLM). Este método permite que o modelo aprenda representações bidirecionais robustas [1].

Processo de MLM:

1. Seleção aleatória de tokens para mascaramento (tipicamente 15% dos tokens).
2. Substituição dos tokens selecionados:
   - 80% substituídos pelo token [MASK]
   - 10% substituídos por um token aleatório
   - 10% mantidos inalterados

3. O modelo é treinado para prever os tokens originais nos locais mascarados.

A função de perda para o MLM é definida como [1]:

$$
L_{MLM} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i|z_i)
$$

Onde:
- $M$ é o conjunto de tokens mascarados
- $x_i$ é o token original
- $z_i$ é a representação de saída do modelo para o token mascarado

> 💡 **Insight**: O MLM força o modelo a usar o contexto bidirecionalmente, aprendendo representações mais ricas do que modelos unidirecionais.

#### Implementação em PyTorch

Aqui está um esboço simplificado de como implementar o mascaramento para MLM:

```python
import torch
import torch.nn.functional as F

def create_mlm_input(inputs, tokenizer, mask_prob=0.15):
    # Crie uma cópia dos inputs para modificação
    masked_inputs = inputs.clone()
    
    # Crie uma máscara aleatória
    rand = torch.rand(inputs.shape)
    mask_arr = (rand < mask_prob) * (inputs != tokenizer.pad_token_id)
    
    # Aplique o mascaramento
    selection = torch.flatten(mask_arr.nonzero()).tolist()
    masked_inputs[selection] = tokenizer.mask_token_id

    return masked_inputs, inputs

# Uso em um loop de treinamento
for batch in dataloader:
    masked_inputs, labels = create_mlm_input(batch['input_ids'], tokenizer)
    outputs = model(masked_inputs)
    loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1), ignore_index=tokenizer.pad_token_id)
    loss.backward()
    optimizer.step()
```

Este código demonstra como criar entradas mascaradas para MLM e como calcular a perda usando essas entradas mascaradas.

### Fine-tuning para Tarefas Específicas

O processo de fine-tuning adapta um modelo pré-treinado para tarefas específicas. Para modelos encoder-only, isso geralmente envolve adicionar camadas de classificação específicas da tarefa no topo da saída do encoder [1].

Passos típicos de fine-tuning:

1. Inicialização com pesos pré-treinados.
2. Adição de camadas específicas da tarefa (por exemplo, classificação).
3. Treinamento em dados rotulados para a tarefa específica.

Por exemplo, para classificação de sequências:

$$
y = \text{softmax}(W_C z_{CLS})
$$

Onde:
- $z_{CLS}$ é a representação do token [CLS] (usado para representar toda a sequência)
- $W_C$ são os pesos da camada de classificação

> ⚠️ **Nota Importante**: Durante o fine-tuning, é comum congelar ou atualizar minimamente os pesos do modelo pré-treinado, focando o aprendizado nas novas camadas adicionadas.

#### Questões Técnicas/Teóricas

1. Como você abordaria o problema de "catastrophic forgetting" durante o fine-tuning de um modelo encoder-only para uma tarefa específica?
2. Descreva uma estratégia para fine-tuning eficiente em cenários de poucos dados (few-shot learning) usando um modelo encoder-only pré-treinado.

### Aplicações e Variantes

Os modelos encoder-only têm sido aplicados com sucesso em uma variedade de tarefas de NLP:

1. **Classificação de Sequências**: Sentimento, tópicos, etc.
2. **Rotulação de Sequências**: NER, POS tagging.
3. **Inferência de Linguagem Natural**: Entailment, paráfrase.
4. **Resposta a Perguntas**: Extração de respostas de textos.

Variantes notáveis incluem:

- **RoBERTa**: Otimização do BERT com treinamento mais robusto [4].
- **XLM-RoBERTa**: Versão multilíngue com vocabulário expandido [5].
- **SpanBERT**: Foco em representações de spans ao invés de tokens individuais [6].

> ✔️ **Ponto de Destaque**: A versatilidade dos modelos encoder-only permite sua aplicação em uma ampla gama de tarefas de NLP, muitas vezes superando modelos específicos de tarefa.

### Análise de Embeddings Contextuais

Os embeddings contextuais produzidos por modelos encoder-only têm propriedades interessantes:

1. **Anisotropia**: Tendência dos vetores de apontarem na mesma direção [7].
2. **Geometria do Espaço de Embeddings**: Clusters de sentido de palavras [8].

Para mitigar a anisotropia e melhorar a utilidade dos embeddings, pode-se aplicar padronização (z-scoring) [9]:

$$
z = \frac{x - \mu}{\sigma}
$$

Onde:
- $x$ é o vetor de embedding original
- $\mu$ é a média do corpus de embeddings
- $\sigma$ é o desvio padrão do corpus

Esta normalização ajuda a tornar os embeddings mais isotrópicos e melhora sua eficácia em tarefas de similaridade e classificação.

### Conclusão

As arquiteturas encoder-only, exemplificadas por modelos como BERT e suas variantes, representam um avanço significativo na criação de representações contextuais ricas para tarefas de NLP. Através do uso de atenção bidirecional e técnicas de treinamento como MLM, esses modelos são capazes de capturar nuances semânticas e relações complexas dentro do texto. 

A capacidade de fine-tuning desses modelos para tarefas específicas, combinada com sua habilidade de produzir embeddings contextuais de alta qualidade, os torna ferramentas poderosas para uma ampla gama de aplicações em processamento de linguagem natural. À medida que a pesquisa avança, podemos esperar refinamentos adicionais nessas arquiteturas, potencialmente levando a melhorias ainda maiores na compreensão e geração de linguagem natural por máquinas.

### Questões Avançadas

1. Compare e contraste as vantagens e desvantagens de usar uma arquitetura encoder-only versus uma arquitetura encoder-decoder para uma tarefa de tradução automática. Como você decidiria qual abordagem usar em um cenário específico?

2. Discuta as implicações éticas e práticas do uso de modelos encoder-only multilíngues como XLM-RoBERTa em aplicações globais. Como esses modelos podem perpetuar ou mitigar vieses linguísticos e culturais?

3. Proponha uma arquitetura híbrida que combine as forças dos modelos encoder-only com modelos generativos. Como essa arquitetura poderia ser treinada e quais seriam seus potenciais casos de uso?

4. Analise criticamente o trade-off entre o tamanho do modelo (número de parâmetros) e a qualidade das representações contextuais em arquiteturas encoder-only. Como você abordaria o problema de escalar esses modelos para capturar conhecimentos ainda mais amplos sem comprometer a eficiência computacional?

5. Descreva uma metodologia para avaliar a "compreensão" real de um modelo encoder-only pré-treinado, além de métricas de desempenho em tarefas específicas. Como podemos distinguir entre memorização superficial e entendimento profundo nesses modelos?

### Referências

[1] "Bidirectional encoders overcome this limitation by allowing the self-attention mechanism to range over the entire input, as shown in Fig. 11.1b." (Trecho de Fine-Tuning and Masked Language Models)

[2] "Pretrained language models based on bidirectional encoders can be learned using a masked language model objective where a model is trained to guess the missing information from an input." (Trecho de Fine-Tuning and Masked Language Models)

[3] "By contrast, with contextual embeddings, such as those learned by masked language models like BERT, each word w will be represented by a different vector each time it appears in a different context." (Trecho de Fine-Tuning and Masked Language Models)

[4] "RoBERTa: A robustly optimized BERT pretraining approach." (Trecho de Fine-Tuning and Masked Language Models)

[5] "The larger multilingual XLM-RoBERTa model, trained on 100 languages, has" (Trecho de Fine-Tuning and Masked Language Models)

[6] "SpanBERT: Improving pre-training by representing and predicting spans." (Trecho de Fine-Tuning and Masked Language Models)

[7] "Ethayarajh (2019) defines the anisotropy of a model as the expected cosine similarity of any pair of words in a corpus." (Trecho de Fine-Tuning and Masked Language Models)

[8] "Fig. 11.7 shows a two-dimensional project of many instances of the BERT embeddings of the word die in English and German." (Trecho de Fine-Tuning and Masked Language Models)

[9] "Timkey and van Schijndel (2021) shows that we can make the embeddings more isotropic by standardizing (z-scoring) the vectors, i.e., subtracting the mean and dividing by the variance." (Trecho de Fine-Tuning and Masked Language Models)