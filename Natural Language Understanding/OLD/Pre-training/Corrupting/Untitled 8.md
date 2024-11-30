## Cloze Task: Treinamento de Encoders Bidirecionais através de Preenchimento de Lacunas

<image: Um diagrama mostrando um texto com palavras mascaradas, representando a tarefa de Cloze, com setas bidirecionais indicando o fluxo de informação em ambas as direções no modelo de linguagem>

### Introdução

A tarefa de Cloze, também conhecida como tarefa de preenchimento de lacunas, emergiu como um objetivo de treinamento fundamental para encoders bidirecionais em modelos de linguagem de última geração [1]. Esta abordagem contrasta significativamente com o método tradicional de predição da próxima palavra utilizado em modelos causais, oferecendo uma nova perspectiva sobre como treinar modelos de linguagem para compreender contextos mais amplos e bidirecionais [2].

> ✔️ **Ponto de Destaque**: A tarefa de Cloze permite que os modelos aprendam representações contextuais bidirecionais, superando as limitações dos modelos causais que só consideram o contexto à esquerda.

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Cloze Task**            | Tarefa de aprendizado onde o modelo deve prever elementos faltantes em uma sequência de entrada, dado o contexto circundante [1]. |
| **Encoder Bidirecional**  | Arquitetura que processa a entrada em ambas as direções, permitindo que cada token seja contextualizado por todo o seu entorno [2]. |
| **Masked Language Model** | Abordagem de treinamento onde tokens aleatórios são mascarados e o modelo é treinado para prevê-los, utilizando tanto o contexto esquerdo quanto o direito [3]. |

### Cloze Task vs. Predição da Próxima Palavra

<image: Um diagrama comparativo mostrando um modelo causal predizendo a próxima palavra à esquerda, e um modelo bidirecional preenchendo lacunas no meio do texto à direita>

A tarefa de Cloze representa uma evolução significativa em relação à predição da próxima palavra, método tradicionalmente utilizado em modelos causais. Enquanto a predição da próxima palavra foca em antecipar o token seguinte baseando-se apenas no contexto anterior, a tarefa de Cloze desafia o modelo a compreender e utilizar o contexto completo da sequência [4].

#### 👍 Vantagens da Cloze Task

* Permite aprendizado de representações bidirecionais [2]
* Facilita a captura de dependências de longo alcance em ambas as direções [3]
* Melhora a compreensão contextual geral do modelo [4]

#### 👎 Desvantagens da Predição da Próxima Palavra

* Limitada ao contexto esquerdo, ignorando informações futuras [2]
* Pode levar a vieses direcionais na representação de palavras [4]
* Menos eficaz para tarefas que requerem compreensão bidirecional do contexto [3]

> ❗ **Ponto de Atenção**: A eliminação da máscara causal em modelos bidirecionais torna a tarefa de predição da próxima palavra trivial, necessitando de uma nova abordagem de treinamento [5].

### Formulação Matemática da Cloze Task

A tarefa de Cloze pode ser formalizada matematicamente da seguinte forma [6]:

Dado um conjunto de tokens de entrada $X = \{x_1, x_2, ..., x_n\}$, selecionamos aleatoriamente um subconjunto $M \subset X$ para mascaramento. O objetivo é maximizar a probabilidade logarítmica dos tokens mascarados, dado o restante da sequência:

$$
\mathcal{L} = \sum_{x_i \in M} \log P(x_i | X \setminus M)
$$

Onde:
- $\mathcal{L}$ é a função de perda (loss)
- $P(x_i | X \setminus M)$ é a probabilidade do token mascarado $x_i$ dado o contexto não mascarado

Esta formulação incentiva o modelo a utilizar tanto o contexto esquerdo quanto o direito para prever os tokens mascarados, promovendo assim uma compreensão bidirecional do texto [7].

#### Questões Técnicas/Teóricas

1. Como a tarefa de Cloze difere fundamentalmente da predição da próxima palavra em termos de aprendizado de contexto?
2. Quais são as implicações da tarefa de Cloze para a captura de dependências de longo alcance em modelos de linguagem?

### Implementação da Cloze Task

A implementação da tarefa de Cloze em modelos como BERT envolve várias etapas críticas [8]:

1. **Tokenização**: A entrada é primeiro tokenizada usando um modelo de subpalavras.
2. **Seleção de Tokens**: Aproximadamente 15% dos tokens de entrada são selecionados aleatoriamente para mascaramento.
3. **Mascaramento**: Os tokens selecionados são tratados da seguinte forma:
   - 80% são substituídos pelo token especial [MASK]
   - 10% são substituídos por um token aleatório
   - 10% são mantidos inalterados

> ⚠️ **Nota Importante**: A variação no tratamento dos tokens selecionados (mascaramento, substituição, manutenção) ajuda o modelo a manter uma representação robusta de cada token e reduz a discrepância entre o treinamento e a inferência [9].

Aqui está um exemplo simplificado de como implementar o mascaramento para a tarefa de Cloze usando PyTorch:

```python
import torch
import random

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
```

Este código implementa a lógica de mascaramento descrita anteriormente, preparando os dados para o treinamento com a tarefa de Cloze [10].

### Impacto da Cloze Task em Modelos de Linguagem

A introdução da tarefa de Cloze como objetivo de treinamento teve um impacto profundo no desenvolvimento de modelos de linguagem [11]:

1. **Representações Contextuais Melhoradas**: Os modelos treinados com Cloze Task capturam melhor o contexto bidirecional, resultando em representações mais ricas e informativas [12].

2. **Flexibilidade em Downstream Tasks**: A natureza bidirecional das representações aprendidas facilita a adaptação dos modelos para uma variedade de tarefas downstream, como classificação de texto, resposta a perguntas e inferência de linguagem natural [13].

3. **Eficiência de Treinamento**: Apesar de mascarar apenas uma porção dos tokens de entrada, a tarefa de Cloze permite um treinamento eficiente, aproveitando toda a sequência de entrada para aprendizado [14].

4. **Generalização Melhorada**: A exposição a diversos contextos durante o treinamento com Cloze Task melhora a capacidade de generalização do modelo para textos e tarefas não vistos [15].

> ✔️ **Ponto de Destaque**: A tarefa de Cloze permitiu o desenvolvimento de modelos como BERT, que revolucionaram o processamento de linguagem natural ao fornecer representações contextuais poderosas e versáteis [16].

#### Questões Técnicas/Teóricas

1. Como a porcentagem de tokens mascarados (15% no BERT) afeta o desempenho e a eficiência do treinamento do modelo?
2. Quais são as considerações ao escolher entre uma abordagem de Cloze Task e um modelo autoregressive para uma tarefa específica de NLP?

### Conclusão

A tarefa de Cloze emergiu como um paradigma de treinamento crucial para modelos de linguagem bidirecionais, oferecendo uma alternativa poderosa à predição da próxima palavra usada em modelos causais [17]. Ao permitir que os modelos aprendam representações contextuais ricas e bidirecionais, a Cloze Task abriu caminho para avanços significativos em uma ampla gama de tarefas de processamento de linguagem natural [18].

A capacidade de utilizar tanto o contexto esquerdo quanto o direito para prever tokens mascarados permite que os modelos capturem dependências complexas e de longo alcance no texto, resultando em representações mais informativas e versáteis [19]. Esta abordagem não apenas melhorou o desempenho em tarefas downstream, mas também aumentou a flexibilidade e adaptabilidade dos modelos de linguagem pré-treinados [20].

À medida que o campo do processamento de linguagem natural continua a evoluir, a tarefa de Cloze permanece uma técnica fundamental, impulsionando o desenvolvimento de modelos cada vez mais sofisticados e capazes [21].

### Questões Avançadas

1. Como a tarefa de Cloze poderia ser adaptada para capturar melhor estruturas sintáticas ou semânticas específicas em diferentes idiomas?

2. Considerando as limitações da tarefa de Cloze, como você projetaria um objetivo de pré-treinamento que pudesse capturar ainda melhor as nuances contextuais e as relações de longo alcance em textos?

3. Discuta as implicações teóricas e práticas de combinar a tarefa de Cloze com outros objetivos de treinamento, como a modelagem de linguagem contrastiva ou a predição de span, para melhorar ainda mais as representações aprendidas.

### Referências

[1] "We're asked to predict a missing item given the rest of the sentence." (Trecho de Fine-Tuning and Masked Language Models)

[2] "Instead of predicting the next word, the model learns to perform a fill-in-the-blank task, technically called the cloze task (Taylor, 1953)." (Trecho de Fine-Tuning and Masked Language Models)

[3] "To see this, let's return to the motivating example from Chapter 3. Instead of predicting which words are likely to come next in this example:

Please turn your homework ____ .

we're asked to predict a missing item given the rest of the sentence.

Please turn ____ homework in." (Trecho de Fine-Tuning and Masked Language Models)

[4] "That is, given an input sequence with one or more elements missing, the learning task is to predict the missing elements." (Trecho de Fine-Tuning and Masked Language Models)

[5] "eliminating the causal mask makes the guess-the-next-word language modeling task trivial since the answer is now directly available from the context" (Trecho de Fine-Tuning and Masked Language Models)

[6] "More precisely, during training the model is deprived of one or more elements of an input sequence and must generate a probability distribution over the vocabulary for each of the missing items. We then use the cross-entropy loss from each of the model's predictions to drive the learning process." (Trecho de Fine-Tuning and Masked Language Models)

[7] "This approach can be generalized to any of a variety of methods that corrupt the training input and then asks the model to recover the original input. Examples of the kinds of manipulations that have been used include masks, substitutions, reorderings, deletions, and extraneous insertions into the training text." (Trecho de Fine-Tuning and Masked Language Models)

[8] "The original approach to training bidirectional encoders is called Masked Language Modeling (MLM) (Devlin et al., 2019). As with the language model training methods we've already seen, MLM uses unannotated text from a large corpus. Here, the model is presented with a series of sentences from the training corpus where a random sample of tokens from each training sequence is selected for use in the learning task. Once chosen, a token is used in one of three ways:

• It is replaced with the unique vocabulary token [MASK].
• It is replaced with another token from the vocabulary, randomly sampled based on token unigram probabilities.

• It is left unchanged." (Trecho de Fine-Tuning and Masked Language Models)

[9] "In BERT, 15% of the input tokens in a training sequence are sampled for learning. Of these, 80% are replaced with [MASK], 10% are replaced with randomly selected tokens, and the remaining 10% are left unchanged." (Trecho de Fine-Tuning and Masked Language Models)

[10] "The MLM training objective is to predict the original inputs for each of the masked tokens using a bidirectional encoder of the kind described in the last section. The cross-entropy loss from these predictions drives the training process for all the parameters in the model. Note that all of the input tokens play a role in the self-attention process, but only the sampled tokens are used for learning." (Trecho de Fine-Tuning and Masked Language Models)

[11] "More specifically, the original input sequence is first tokenized using a subword model. The sampled items which drive the learning process are chosen from among the set of tokenized inputs. Word embeddings for all of the tokens in the input are retrieved from the word embedding matrix and then combined with positional embeddings to form the input to the transformer." (Trecho de Fine-Tuning and Masked Language Models)

[12] "Fig. 11.4 illustrates this approach with a simple example. Here, long, thanks and the have been sampled from the training sequence, with the first two masked and the replaced with the randomly sampled token apricot. The resulting embeddings are passed through a stack of bidirectional transformer blocks." (Trecho de Fine-Tuning and Masked Language Models)

[13] "To produce a probability distribution over the vocabulary for each of the masked tokens, the output vector zi from the final transformer layer for each masked token i is multiplied by a learned set of classification weights WV ∈ R|V|×dh and then through a softmax to yield the required predictions over the vocabulary." (Trecho de Fine-Tuning and Masked Language Models)

[14] "yi = softmax(WVzi)" (Trecho de Fine-Tuning and Masked Language Models)

[15] "With a predicted probability distribution for each masked item, we can use cross-entropy to compute the loss for each masked item—the negative log probability assigned to the actual masked word, as shown in Fig. 11.4." (Trecho de Fine-Tuning and Masked Language Models)

[16] "More formally, for a given vector of input tokens in a sentence or batch be x, let the set of tokens that are masked be M, the version of that sentence with some tokens replaced by masks be x^mask, and the sequence of output vectors be z. For a given input token x_i, such as the word long in Fig. 11.4, the loss is the probability of the correct word long, given x^mask (