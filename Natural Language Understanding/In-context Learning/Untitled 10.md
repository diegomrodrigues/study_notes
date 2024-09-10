## Seleção de Demonstrações: Heurísticas para Escolha de Demonstrações Eficazes

<image: Um diagrama mostrando diferentes métodos de seleção de demonstrações, incluindo similaridade com a entrada e seleção automatizada baseada em melhoria de desempenho. O diagrama deve incluir setas conectando os métodos a um conjunto de demonstrações, representando o processo de seleção.>

### Introdução

A seleção de demonstrações é um aspecto crucial no campo do prompt engineering e few-shot learning para Large Language Models (LLMs). Este tópico explora as heurísticas e métodos utilizados para escolher demonstrações eficazes que melhoram o desempenho dos modelos em tarefas específicas. A escolha adequada de demonstrações pode significativamente impactar a capacidade do modelo de generalizar e realizar tarefas com maior precisão [1].

### Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Few-shot Learning**      | Técnica onde o modelo aprende a realizar uma tarefa com apenas alguns exemplos de demonstração [2]. |
| **Demonstrações**          | Exemplos rotulados fornecidos no prompt para guiar o modelo na realização de uma tarefa específica [3]. |
| **Heurísticas de Seleção** | Métodos e critérios utilizados para escolher as demonstrações mais eficazes para uma determinada tarefa [4]. |

> ⚠️ **Important Note**: A eficácia das demonstrações não depende apenas da quantidade, mas principalmente da qualidade e relevância para a tarefa em questão.

### Heurísticas para Seleção de Demonstrações

#### Similaridade com a Entrada

Uma das heurísticas mais comuns para selecionar demonstrações eficazes é baseada na similaridade com a entrada atual [5]. Esta abordagem parte do princípio de que exemplos semelhantes à tarefa em questão podem fornecer um contexto mais relevante para o modelo.

<image: Um gráfico de dispersão mostrando a distribuição de exemplos em um espaço bidimensional, com clusters representando diferentes tipos de tarefas. Destaque alguns pontos como "entrada atual" e "demonstrações selecionadas" baseadas em proximidade.>

Para implementar esta heurística, pode-se utilizar técnicas de embedding e cálculo de similaridade:

```python
import torch
from transformers import AutoTokenizer, AutoModel

def get_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def select_similar_demonstrations(input_text, demonstrations, model, tokenizer, top_k=3):
    input_embedding = get_embeddings([input_text], model, tokenizer)
    demo_embeddings = get_embeddings(demonstrations, model, tokenizer)
    
    similarities = torch.cosine_similarity(input_embedding, demo_embeddings)
    top_indices = similarities.argsort(descending=True)[:top_k]
    
    return [demonstrations[i] for i in top_indices]
```

Esta função seleciona as `top_k` demonstrações mais similares à entrada com base na similaridade de cosseno entre seus embeddings [6].

> ✔️ **Highlight**: A seleção baseada em similaridade pode ser particularmente eficaz para tarefas onde o contexto específico é crucial, como em tradução ou resposta a perguntas específicas de domínio.

#### Seleção Automatizada Baseada em Desempenho

Outra abordagem avançada é a seleção automatizada de demonstrações com base na melhoria de desempenho que elas proporcionam [7]. Este método envolve a avaliação iterativa de diferentes conjuntos de demonstrações em um conjunto de validação.

<image: Um fluxograma mostrando o processo de seleção automatizada: conjunto inicial de demonstrações -> avaliação em conjunto de validação -> seleção das melhores -> iteração.>

A implementação desta abordagem pode ser feita utilizando um algoritmo de busca, como a busca em feixe (beam search):

```python
def evaluate_demonstrations(model, task, demonstrations, validation_set):
    # Implementação da avaliação do modelo com as demonstrações no conjunto de validação
    pass

def beam_search_demonstrations(model, task, candidate_demos, validation_set, beam_width=3, iterations=5):
    current_beam = [set()]
    for _ in range(iterations):
        candidates = []
        for demo_set in current_beam:
            for demo in candidate_demos:
                if demo not in demo_set:
                    new_set = demo_set.union({demo})
                    score = evaluate_demonstrations(model, task, new_set, validation_set)
                    candidates.append((new_set, score))
        
        current_beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return current_beam[0][0]  # Retorna o melhor conjunto de demonstrações
```

Este algoritmo realiza uma busca em feixe para encontrar o conjunto de demonstrações que maximiza o desempenho no conjunto de validação [8].

> ❗ **Attention Point**: A seleção automatizada pode ser computacionalmente intensiva, especialmente para grandes conjuntos de candidatos a demonstrações. É importante balancear a qualidade da seleção com a eficiência computacional.

#### Questões Técnicas/Teóricas

1. Como a dimensionalidade dos embeddings pode afetar a eficácia da seleção baseada em similaridade? Discuta possíveis trade-offs.
2. Em um cenário de few-shot learning para classificação de texto, como você abordaria a seleção de demonstrações para garantir uma representação equilibrada das classes?

### Comparação de Métodos de Seleção

| 👍 Vantagens                                                  | 👎 Desvantagens                                           |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| Similaridade: Rápido e intuitivo [9]                         | Similaridade: Pode não capturar nuances da tarefa [10]   |
| Baseado em Desempenho: Otimiza diretamente para a tarefa [11] | Baseado em Desempenho: Computacionalmente intensivo [12] |

### Considerações Teóricas

A seleção de demonstrações pode ser formalizada como um problema de otimização. Dado um conjunto de candidatos a demonstrações $D = \{d_1, ..., d_n\}$, queremos encontrar um subconjunto $S \subset D$ que maximize o desempenho do modelo $M$ na tarefa $T$:

$$
S^* = \argmax_{S \subset D, |S| \leq k} \mathbb{E}_{x \sim X}[P(M(x|S) = y)]
$$

Onde $X$ é o espaço de entradas, $y$ é a saída correta, e $k$ é o número máximo de demonstrações permitidas [13].

Esta formulação captura a essência do problema, mas na prática, a otimização exata é intratável para grandes conjuntos de demonstrações. Por isso, utilizamos heurísticas e métodos aproximados.

#### Questões Técnicas/Teóricas

1. Como você adaptaria a formulação matemática acima para lidar com múltiplas tarefas simultaneamente em um cenário de aprendizado multi-tarefa?
2. Discuta as implicações teóricas de usar embeddings pré-treinados versus embeddings específicos da tarefa para a seleção baseada em similaridade.

### Conclusão

A seleção eficaz de demonstrações é um aspecto crítico no desenvolvimento de sistemas de prompt engineering e few-shot learning. As heurísticas baseadas em similaridade oferecem uma abordagem rápida e intuitiva, enquanto métodos automatizados baseados em desempenho podem proporcionar resultados mais otimizados, embora com maior custo computacional [14]. A escolha do método depende do equilíbrio entre eficiência computacional, desempenho desejado e características específicas da tarefa em questão.

### Questões Avançadas

1. Como você abordaria o problema de seleção de demonstrações em um cenário de aprendizado contínuo, onde novas tarefas e dados estão constantemente sendo introduzidos?
2. Proponha e discuta um método híbrido que combine seleção baseada em similaridade e otimização de desempenho, visando maximizar a eficácia das demonstrações enquanto minimiza o custo computacional.
3. Considerando as limitações de contexto em LLMs, como você equilibraria a quantidade e qualidade das demonstrações com a necessidade de espaço para a tarefa real e a resposta do modelo?

### Referências

[1] "Demonstrations are generally created by formatting examples drawn from a labeled training set" (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[2] "Few-shot prompting, as contrasted with zero-shot prompting which means instructions that don't include labeled examples." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[3] "We call such examples demonstrations." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[4] "There are some heuristics about what makes a good demonstration." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[5] "For example, using demonstrations that are similar to the current input seems to improve performance." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[6] "It can thus be useful to dynamically retrieve demonstrations for each input, based on their similarity to the current example (for example, comparing the embedding of the current example with embeddings of each of the training set example to find the best top-T)." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[7] "But more generally, the best way to select demonstrations from the training set is programmatically: choosing the set of demonstrations that most increases task performance of the prompt on a test set." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[8] "Systems like DSPy (Khattab et al., 2024), a framework for algorithmically optimizing LM prompts, can automatically find the optimum set of demonstrations to include by searching through the space of possible demonstrations to include." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[9] "For example, using demonstrations that are similar to the current input seems to improve performance." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[10] "The reason is that the primary benefit in examples is to demonstrate the task to be performed to the LLM and the format of the sequence, not to provide relevant information as to the right answer for any particular question." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[11] "But more generally, the best way to select demonstrations from the training set is programmatically: choosing the set of demonstrations that most increases task performance of the prompt on a test set." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[12] "Adding too many examples seems to cause the model to overfit to details of the exact examples chosen and generalize poorly." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[13] "Task performance for sentiment analysis or multiple-choice question answering can be measured in accuracy; for machine translation with chrF; and for summarization via Rouge." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)

[14] "The goal is to continue to seek improved prompts given the computational resources available." (Excerpt from CHAPTER 12: Model Alignment, Prompting, and In-Context Learning)