## Métodos de Corrupção de Input: Técnicas para Treinamento de Encoders Bidirecionais

<image: Um diagrama mostrando diferentes tipos de corrupção de texto, incluindo mascaramento, substituições, reordenações, deleções e inserções, aplicados a uma frase de exemplo.>

### Introdução

Os métodos de corrupção de input são técnicas fundamentais no treinamento de modelos de linguagem baseados em encoders bidirecionais, como o BERT (Bidirectional Encoder Representations from Transformers) [1]. Estas técnicas visam criar tarefas de preenchimento de lacunas (cloze tasks) que forçam o modelo a aprender representações contextuais robustas [2]. Ao corromper o texto de entrada de maneiras específicas, criamos um desafio para o modelo reconstruir a informação original, promovendo assim uma compreensão mais profunda da estrutura e semântica da linguagem [3].

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Masked Language Model** | Paradigma de treinamento onde o modelo aprende a prever tokens mascarados em uma sequência, permitindo aprendizado bidirecional [1]. |
| **Cloze Task**            | Tarefa de preenchimento de lacunas, onde o modelo deve prever informações faltantes em um texto, fundamental para o treinamento de modelos como BERT [2]. |
| **Tokenização Subword**   | Processo de dividir palavras em unidades menores (subwords), permitindo lidar com vocabulários extensos e palavras desconhecidas de forma eficiente [4]. |

> ⚠️ **Nota Importante**: A eficácia dos métodos de corrupção de input está intimamente ligada à capacidade do modelo de aprender representações contextuais robustas, essenciais para tarefas de NLP downstream.

### Técnicas de Corrupção de Input

<image: Uma série de exemplos visuais mostrando cada técnica de corrupção aplicada a uma mesma frase, destacando as diferenças entre cada método.>

#### 1. Mascaramento (Masking)

O mascaramento é a técnica central no treinamento de modelos como BERT [1]. Neste método:

- Tokens são aleatoriamente substituídos por um token especial [MASK].
- Tipicamente, 15% dos tokens são selecionados para mascaramento [1].
- Dos tokens selecionados:
  - 80% são substituídos por [MASK]
  - 10% são substituídos por um token aleatório
  - 10% permanecem inalterados

Exemplo matemático:

Seja $X = [x_1, x_2, ..., x_n]$ uma sequência de tokens de entrada. A probabilidade de um token $x_i$ ser selecionado para mascaramento é:

$$
P(x_i \text{ é selecionado}) = 0.15
$$

Para cada token selecionado, a distribuição de substituição é:

$$
P(x_i \rightarrow \text{[MASK]}) = 0.8
$$
$$
P(x_i \rightarrow \text{token aleatório}) = 0.1
$$
$$
P(x_i \rightarrow x_i) = 0.1
$$

> ✔️ **Ponto de Destaque**: O mascaramento parcial (10% de substituição aleatória e 10% sem alteração) ajuda o modelo a manter a distribuição de tokens esperada e reduz a discrepância entre treinamento e inferência.

#### 2. Substituições (Substitutions)

As substituições envolvem trocar tokens por outros do vocabulário, desafiando o modelo a identificar incongruências contextuais [5].

- Tokens são substituídos com base em probabilidades unigrama do vocabulário.
- Ajuda o modelo a aprender nuances semânticas e relações entre palavras.

Exemplo matemático:

Seja $V$ o vocabulário e $f(w)$ a frequência do token $w$ em $V$. A probabilidade de substituir um token $x_i$ por $w$ é:

$$
P(x_i \rightarrow w) = \frac{f(w)}{\sum_{v \in V} f(v)}
$$

#### 3. Reordenações (Reorderings)

Reordenações envolvem alterar a ordem dos tokens em uma sequência, desafiando o modelo a entender estruturas sintáticas [6].

- Pode envolver trocas de pares adjacentes ou permutações mais complexas.
- Ajuda o modelo a capturar dependências de longa distância e ordem das palavras.

Exemplo de implementação simplificada:

```python
import random

def reorder_sequence(sequence, p=0.15):
    for i in range(len(sequence) - 1):
        if random.random() < p:
            sequence[i], sequence[i+1] = sequence[i+1], sequence[i]
    return sequence
```

#### 4. Deleções (Deletions)

Deleções removem tokens da sequência, forçando o modelo a inferir informações faltantes [7].

- Tipicamente, uma pequena porcentagem de tokens é removida aleatoriamente.
- Ajuda o modelo a lidar com informações incompletas e melhorar a robustez.

Exemplo matemático:

Seja $p_d$ a probabilidade de deleção. Para cada token $x_i$:

$$
P(x_i \text{ é deletado}) = p_d
$$

A sequência resultante após deleções $X' = [x'_1, x'_2, ..., x'_m]$, onde $m \leq n$.

#### 5. Inserções (Insertions)

Inserções adicionam tokens extras à sequência, desafiando o modelo a identificar informações irrelevantes ou redundantes [8].

- Tokens são inseridos aleatoriamente na sequência.
- Ajuda o modelo a lidar com ruído e melhorar a capacidade de extração de informações relevantes.

Exemplo de implementação simplificada:

```python
import random

def insert_random_tokens(sequence, vocab, p=0.15):
    new_sequence = []
    for token in sequence:
        new_sequence.append(token)
        if random.random() < p:
            new_sequence.append(random.choice(vocab))
    return new_sequence
```

> ❗ **Ponto de Atenção**: A combinação adequada destas técnicas é crucial para o treinamento eficaz de modelos de linguagem bidirecionais.

#### Questões Técnicas/Teóricas

1. Como o mascaramento parcial (80% [MASK], 10% aleatório, 10% inalterado) contribui para reduzir a discrepância entre treinamento e inferência em modelos como BERT?

2. Descreva como a técnica de reordenação pode ajudar um modelo a capturar dependências de longa distância em uma sentença.

### Implementação e Considerações Práticas

A implementação eficaz das técnicas de corrupção de input requer um equilíbrio cuidadoso entre diferentes fatores:

1. **Taxa de Corrupção**: A porcentagem de tokens afetados deve ser suficiente para desafiar o modelo, mas não tão alta a ponto de tornar a tarefa impossível. Tipicamente, 15% é usado para mascaramento em BERT [1].

2. **Combinação de Técnicas**: Muitos modelos modernos utilizam uma combinação de técnicas. Por exemplo, o SpanBERT [9] utiliza mascaramento de spans contíguos ao invés de tokens individuais.

3. **Adaptação ao Domínio**: As técnicas de corrupção podem ser adaptadas para domínios específicos. Por exemplo, em tarefas de compreensão de documentos, pode-se dar mais ênfase à corrupção de entidades nomeadas ou termos-chave.

Exemplo de implementação de uma função de corrupção combinada:

```python
import random

def corrupt_input(sequence, vocab, mask_token="[MASK]", p_mask=0.15, p_random=0.1, p_unchanged=0.1, p_reorder=0.1, p_delete=0.05, p_insert=0.05):
    corrupted = []
    i = 0
    while i < len(sequence):
        r = random.random()
        if r < p_mask:
            corrupted.append(mask_token)
        elif r < p_mask + p_random:
            corrupted.append(random.choice(vocab))
        elif r < p_mask + p_random + p_unchanged:
            corrupted.append(sequence[i])
        elif r < p_mask + p_random + p_unchanged + p_reorder and i+1 < len(sequence):
            corrupted.append(sequence[i+1])
            corrupted.append(sequence[i])
            i += 1
        elif r < p_mask + p_random + p_unchanged + p_reorder + p_delete:
            pass  # Skip this token (delete)
        elif r < p_mask + p_random + p_unchanged + p_reorder + p_delete + p_insert:
            corrupted.append(random.choice(vocab))
            corrupted.append(sequence[i])
        else:
            corrupted.append(sequence[i])
        i += 1
    return corrupted
```

> ✔️ **Ponto de Destaque**: A implementação acima combina todas as técnicas discutidas, permitindo um controle fino sobre a aplicação de cada método de corrupção.

### Avaliação e Métricas

A eficácia das técnicas de corrupção de input pode ser avaliada através de várias métricas:

1. **Perplexidade**: Mede quão bem o modelo prevê os tokens mascarados ou corrompidos.

$$
\text{Perplexidade} = \exp(-\frac{1}{N}\sum_{i=1}^N \log p(x_i|\text{contexto}))
$$

onde $N$ é o número de tokens no conjunto de teste e $p(x_i|\text{contexto})$ é a probabilidade que o modelo atribui ao token correto $x_i$ dado o contexto.

2. **Acurácia de Reconstrução**: Porcentagem de tokens corrompidos que o modelo reconstrói corretamente.

3. **Desempenho em Tarefas Downstream**: Avaliação do modelo em tarefas como classificação de texto, NER, ou resposta a perguntas após o pré-treinamento.

#### Questões Técnicas/Teóricas

1. Como a perplexidade pode ser interpretada no contexto de modelos de linguagem mascarados, e quais são suas limitações como métrica de avaliação?

2. Proponha uma estratégia para avaliar a eficácia relativa de diferentes técnicas de corrupção (por exemplo, mascaramento vs. deleção) no treinamento de um modelo de linguagem bidirecional.

### Conclusão

As técnicas de corrupção de input são fundamentais para o treinamento eficaz de modelos de linguagem bidirecionais [1][2][3]. Ao aplicar métodos como mascaramento, substituições, reordenações, deleções e inserções, criamos tarefas de aprendizado desafiadoras que forçam o modelo a desenvolver representações contextuais robustas [5][6][7][8]. A combinação cuidadosa destas técnicas, junto com uma implementação equilibrada e avaliação adequada, é crucial para o desenvolvimento de modelos de linguagem de alta performance capazes de capturar nuances semânticas e estruturais complexas da linguagem natural [9].

### Questões Avançadas

1. Desenhe um experimento para comparar a eficácia de diferentes estratégias de mascaramento (por exemplo, mascaramento de tokens individuais vs. mascaramento de spans) em tarefas de transferência de aprendizado para diferentes idiomas. Como você lidaria com as diferenças estruturais entre línguas na concepção deste experimento?

2. Discuta como as técnicas de corrupção de input poderiam ser adaptadas para modelos multimodais que combinam texto e imagem. Quais desafios específicos surgiriam e como eles poderiam ser abordados?

3. Proponha uma nova técnica de corrupção de input que poderia ser particularmente eficaz para capturar relações de longa distância em documentos extensos. Como você avaliaria a eficácia desta nova técnica em comparação com os métodos existentes?

### Referências

[1] "Masked Language Modeling (MLM). As with language model training methods we've already seen, MLM uses unannotated text from a large corpus. Here, the model is presented with a series of sentences from the training corpus where a random sample of tokens from each training sequence is selected for use in the learning task." (Trecho de Fine-Tuning and Masked Language Models)

[2] "To see this, let's return to the motivating example from Chapter 3. Instead of predicting which words are likely to come next in this example:

Please turn your homework ____ .

we're asked to predict a missing item given the rest of the sentence.

Please turn ____ homework in." (Trecho de Fine-Tuning and Masked Language Models)

[3] "That is, given an input sequence with one or more elements missing, the learning task is to predict the missing elements. More precisely, during training the model is deprived of one or more elements of an input sequence and must generate a probability distribution over the vocabulary for each of the missing items." (Trecho de Fine-Tuning and Masked Language Models)

[4] "The use of WordPiece or SentencePiece Unigram LM tokenization (two of the large family of subword tokenization algorithms that includes the BPE algorithm we saw in Chapter 2) means that—like the large language models of Chapter 10— BERT and its descendants are based on subword tokens rather than words." (Trecho de Fine-Tuning and Masked Language Models)

[5] "It is replaced with another token from the vocabulary, randomly sampled based on token unigram probabilities." (Trecho de Fine-Tuning and Masked Language Models)

[6] "Examples of the kinds of manipulations that have been used include masks, substitutions, reorderings, deletions, and extraneous insertions into the training text." (Trecho de Fine-Tuning and Masked Language Models)

[7] "Examples of the kinds of manipulations that have been used include masks, substitutions, reorderings, deletions, and extraneous insertions into the training text." (Trecho de Fine-Tuning and Masked Language Models)

[8] "Examples of the kinds of manipulations that have been used include masks, substitutions, reorderings, deletions, and extraneous insertions into the training text." (Trecho de Fine-Tuning and Masked Language Models)

[9] "SpanBERT (Joshi et al., 2020)" (Trecho de Fine-Tuning and Masked Language Models)