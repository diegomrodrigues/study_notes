## Few-Shot Prompting: Aprimorando o Desempenho com Exemplos Rotulados

<image: Um diagrama mostrando um modelo de linguagem grande recebendo um prompt com múltiplos exemplos rotulados, seguido por uma nova entrada não rotulada, e produzindo uma saída. O diagrama deve enfatizar visualmente como os exemplos no prompt guiam o modelo a gerar uma resposta para a nova entrada.>

### Introdução

Few-shot prompting é uma técnica poderosa que permite melhorar significativamente o desempenho de grandes modelos de linguagem (LLMs) em diversas tarefas, sem a necessidade de fine-tuning ou modificações nos parâmetros do modelo [1]. Esta abordagem envolve a inclusão de exemplos rotulados (demonstrações) diretamente no prompt, fornecendo ao modelo um contexto rico para compreender e executar a tarefa desejada [2].

### Conceitos Fundamentais

| Conceito                | Explicação                                                   |
| ----------------------- | ------------------------------------------------------------ |
| **Few-shot learning**   | Capacidade de um modelo de aprender e generalizar a partir de poucos exemplos, sem necessidade de treinamento extensivo [1]. |
| **Demonstrações**       | Exemplos rotulados incluídos no prompt para guiar o modelo na execução da tarefa [2]. |
| **In-context learning** | Processo pelo qual o modelo aprende a realizar novas tarefas ou melhorar seu desempenho sem atualizações em seus parâmetros, apenas com base no contexto fornecido [3]. |

> ⚠️ **Nota Importante**: O few-shot prompting difere do fine-tuning tradicional, pois não altera os parâmetros do modelo, mas sim aproveita sua capacidade de aprendizado in-context [3].

### Mecanismo de Funcionamento

<image: Um fluxograma detalhando o processo de few-shot prompting, desde a seleção de exemplos até a geração da resposta pelo modelo, destacando como o modelo utiliza as demonstrações para inferir o padrão da tarefa.>

O few-shot prompting funciona aproveitando a capacidade dos LLMs de realizar geração contextual. Dado o prompt como contexto, o modelo gera o próximo token baseado em sua probabilidade, condicionada ao prompt: P(wi|w<i) [4]. Este processo pode ser dividido em etapas:

1. **Seleção de Demonstrações**: Escolha cuidadosa de exemplos representativos da tarefa.
2. **Construção do Prompt**: Incorporação dos exemplos no prompt, seguidos pela nova entrada.
3. **Inferência do Modelo**: O LLM processa o prompt completo, incluindo as demonstrações e a nova entrada.
4. **Geração da Resposta**: O modelo gera uma resposta para a nova entrada, guiado pelos padrões observados nas demonstrações.

> 💡 **Destaque**: A eficácia do few-shot prompting está na sua capacidade de induzir o modelo a reconhecer padrões e aplicá-los a novas entradas, sem necessidade de atualização de parâmetros [2].

#### Formulação Matemática

A probabilidade de geração de uma sequência de tokens $y = (y_1, ..., y_n)$ dado um prompt $x$ pode ser expressa como:

$$
P(y|x) = \prod_{i=1}^n P(y_i|x, y_{<i})
$$

Onde $y_{<i}$ representa todos os tokens gerados antes de $y_i$ [4].

#### Questões Técnicas/Teóricas

1. Como o tamanho do contexto do modelo afeta a eficácia do few-shot prompting?
2. Qual é a relação entre o número de demonstrações e o desempenho do modelo em tarefas complexas?

### Aplicações e Exemplos

O few-shot prompting tem se mostrado eficaz em uma variedade de tarefas de NLP, incluindo:

- Classificação de sentimentos
- Tradução de idiomas
- Resposta a perguntas
- Inferência de linguagem natural

Exemplo de prompt para classificação de sentimentos:

```python
prompt = """
Classifique o sentimento das seguintes frases como positivo ou negativo:

Frase: O filme foi incrível!
Sentimento: Positivo

Frase: O serviço no restaurante foi péssimo.
Sentimento: Negativo

Frase: Estou ansioso para as férias.
Sentimento: Positivo

Frase: O produto chegou quebrado.
Sentimento: Negativo

Frase: A apresentação foi muito cansativa.
Sentimento: 
"""

resposta = modelo_linguagem(prompt)
print(resposta)
```

> ✔️ **Destaque**: Este exemplo demonstra como as demonstrações no prompt guiam o modelo a compreender a tarefa de classificação de sentimentos e aplicá-la a uma nova entrada [5].

### Vantagens e Desvantagens

| 👍 Vantagens                               | 👎 Desvantagens                                         |
| ----------------------------------------- | ------------------------------------------------------ |
| Não requer fine-tuning do modelo [6]      | Limitado pelo tamanho do contexto do modelo [7]        |
| Flexibilidade para diferentes tarefas [6] | Pode ser sensível à ordem e seleção dos exemplos [7]   |
| Rápida adaptação a novos domínios [6]     | Potencial para overfitting em exemplos específicos [7] |

### Induction Heads e In-Context Learning

<image: Um diagrama detalhado de uma "induction head" em um transformer, mostrando o mecanismo de prefix matching e copying, com setas indicando o fluxo de informação.>

Um aspecto fascinante do few-shot prompting é sua relação com as "induction heads" nos transformers. Induction heads são circuitos dentro da computação de atenção que desempenham um papel crucial no aprendizado in-context [8].

Funcionamento de uma Induction Head:
1. **Prefix Matching**: Busca padrões repetidos no input.
2. **Copying**: Copia o token que seguiu o padrão anteriormente.

Matematicamente, uma induction head implementa uma regra de completação de padrão:

$$
AB...A \rightarrow B
$$

Onde A e B são tokens ou sequências de tokens [9].

> ❗ **Ponto de Atenção**: Estudos sugerem que a ablação de induction heads reduz significativamente o desempenho do aprendizado in-context, indicando seu papel crucial no few-shot prompting [10].

#### Questões Técnicas/Teóricas

1. Como a arquitetura das induction heads influencia a capacidade de few-shot learning dos transformers?
2. Qual é a relação entre o número de induction heads e a performance em tarefas de few-shot?

### Otimização de Prompts

A eficácia do few-shot prompting pode ser melhorada através da otimização automática de prompts. Técnicas como busca iterativa e expansão de prompts podem ser empregadas para encontrar as melhores demonstrações e formulações [11].

Algoritmo básico de otimização de prompt:

```python
def otimizar_prompt(prompts_iniciais, largura_beam):
    ativos = prompts_iniciais
    while not concluido:
        fronteira = []
        for prompt in ativos:
            filhos = expandir(prompt)
            for filho in filhos:
                fronteira = adicionar_ao_beam(filho, fronteira, largura_beam)
        ativos = fronteira
    return melhor_prompt(ativos)

def adicionar_ao_beam(estado, agenda, largura):
    if len(agenda) < largura:
        agenda.append(estado)
    elif score(estado) > score(pior_de(agenda)):
        agenda.remove(pior_de(agenda))
        agenda.append(estado)
    return agenda
```

> 💡 **Destaque**: A otimização automática de prompts pode levar a melhorias significativas no desempenho do few-shot prompting [11].

### Conclusão

Few-shot prompting representa um avanço significativo na utilização de grandes modelos de linguagem, permitindo adaptação rápida a novas tarefas sem a necessidade de fine-tuning extensivo [1,2]. Esta técnica aproveita a capacidade de aprendizado in-context dos modelos, possivelmente mediada por estruturas como induction heads [8,9,10]. Apesar de suas limitações, como sensibilidade à seleção de exemplos e restrições de tamanho de contexto, o few-shot prompting oferece uma abordagem flexível e poderosa para uma variedade de aplicações em NLP [6,7].

### Questões Avançadas

1. Como o few-shot prompting se compara ao fine-tuning em termos de desempenho e eficiência computacional em tarefas complexas de NLP?
2. Quais são as implicações éticas e de viés ao usar few-shot prompting em aplicações do mundo real, considerando a influência dos exemplos selecionados?
3. Como podemos integrar técnicas de few-shot prompting com outros métodos de aprendizado de máquina para criar sistemas mais robustos e adaptáveis?

### Referências

[1] "Few-shot prompting relies on contextual generation. Given the prompt as context, the language model generates the next token based on its token probability, conditioned on the prompt: P(wi|w<i)." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[2] "A prompt can also contain demonstrations, examples to help make the instructions clearer." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[3] "For this reason we also refer to prompting as in-context-learning—learning that improves model performance or reduces some loss but does not involve gradient-based updates to the model's underlying parameters." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[4] "Prompting relies on contextual generation. Given the prompt as context, the language model generates the next token based on its token probability, conditioned on the prompt: P(wi|w<i)." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[5] "Fig. 12.2 illustrates a few-shot example from an extractive question answering task. The context combines the task definition along with three gold-standard question and answer pairs from the training set." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[6] "The power of this approach is that with suitable additions to the context a single LLM can produce outputs appropriate for many different tasks." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[7] "Why isn't it useful to have more demonstrations? The reason is that the primary benefit in examples is to demonstrate the task to be performed to the LLM and the format of the sequence, not to provide relevant information as to the right answer for any particular question." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[8] "One hypothesis is based on the idea of induction heads (Elhage et al., 2021; Olsson et al., 2022). Induction heads are the name for a circuit, which is a kind of abstract component of a network." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[9] "The function of the induction head is to predict repeated sequences. For example if it sees the pattern AB...A in an input sequence, it predicts that B will follow, instantiating the pattern completion rule AB...A→B." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[10] "Crosbie and Shutova (2022) show that ablating induction heads causes in-context learning performance to decrease." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[11] "Fig. 12.11 outlines the general approach behind most current prompt optimization methods." (Excerpt from Model Alignment, Prompting, and In-Context Learning)