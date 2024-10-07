# Skip-gram com Negative Sampling (SGNS)

O modelo Skip-gram com Negative Sampling (SGNS) é uma técnica poderosa utilizada no processamento de linguagem natural para aprender representações vetoriais (*embeddings*) de palavras a partir de grandes corpora de texto. ==O objetivo é capturar relações semânticas entre palavras, posicionando-as em um espaço vetorial de tal forma que palavras com contextos similares estejam próximas umas das outras.==

## Definição e Explicação Detalhada

A função objetivo do SGNS é expressa como:

$$
J = \sum_{(w, c) \in D} \left[ \log \sigma(\mathbf{v}_w^\top \mathbf{v}_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_c) \right] \right]
$$

Onde:

- **$D$**: Conjunto de pares palavra-contexto observados no corpus.
- **$\sigma(x) = \frac{1}{1 + e^{-x}}$**: Função sigmoide, que mapeia um valor real para o intervalo (0, 1).
- **$\mathbf{v}_w$ e $\mathbf{v}_c$**: Vetores de representação da palavra central $w$ e do contexto $c$, respectivamente.
- **$P_n(w)$**: Distribuição de amostragem negativa, geralmente baseada na frequência das palavras no corpus.
- **$K$**: Número de amostras negativas para cada par positivo.

### Intuição por Trás da Função Objetivo

==A função objetivo do SGNS busca maximizar a similaridade (produto interno) entre uma palavra e seus contextos verdadeiros enquanto minimiza a similaridade entre a palavra e contextos negativos amostrados aleatoriamente.== Isso é alcançado através de dois termos:

1. **Termo Positivo**: $\log \sigma(\mathbf{v}_w^\top \mathbf{v}_c)$
   - ==Incentiva que o produto interno entre a palavra e seu contexto real seja alto.==
   - A função sigmoide ==mapeia esse produto interno para uma probabilidade alta.==

2. **Termo Negativo**: $\sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_c) \right]$
   - ==Penaliza a similaridade entre o contexto e palavras negativas (não relacionadas).==
   - Busca que o produto interno entre palavras negativas e o contexto seja baixo, resultando em uma probabilidade baixa após a aplicação da sigmoide.

## Passo a Passo para Demonstrar e Utilizar a Definição

### Passo 1: Preparação dos Dados

- **Construção do Corpus**: Colete um grande corpus de texto.
- **Definição da Janela de Contexto**: Escolha um tamanho de janela (por exemplo, 2 palavras antes e depois da palavra central).
- **Geração de Pares Positivos**: Para cada palavra central $w$, colete os contextos $c$ dentro da janela, formando pares $(w, c)$.

### Passo 2: Inicialização dos Vetores

- **Vetores de Palavras e Contextos**: Inicialize os vetores $\mathbf{v}_w$ e $\mathbf{v}_c$ com valores aleatórios pequenos ou utilizando uma distribuição específica.

### Passo 3: Definição da Distribuição de Amostragem Negativa

- **Distribuição $P_n(w)$**: Geralmente definida como a frequência das palavras elevada a uma potência (por exemplo, 0.75) para suavizar a distribuição e dar menos peso a palavras muito frequentes.

### Passo 4: Implementação do Algoritmo

Para cada par positivo $(w, c)$:

1. **Cálculo do Termo Positivo**:
   - Compute o produto interno $\mathbf{v}_w^\top \mathbf{v}_c$.
   - Calcule $\log \sigma(\mathbf{v}_w^\top \mathbf{v}_c)$.

2. **Amostragem Negativa**:
   - Amostre $K$ palavras negativas $w_k$ a partir de $P_n(w)$.
   - Para cada $w_k$, compute $\log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_c)$.

3. **Atualização dos Vetores**:
   - Calcule o gradiente da função objetivo em relação aos vetores $\mathbf{v}_w$, $\mathbf{v}_c$ e $\mathbf{v}_{w_k}$.
   - Atualize os vetores utilizando um algoritmo de otimização, como a descida de gradiente estocástica.

### Passo 5: Repetição e Convergência

- **Iterações**: Repita o processo para todas as palavras no corpus por várias épocas.
- **Convergência**: O treinamento continua até que a função objetivo converja ou por um número pré-definido de épocas.

## Exemplo Numérico Intuitivo

Considere um corpus simples: "o gato senta no tapete".

### Passo 1: Geração de Pares Positivos

Com uma janela de contexto de 1:

- **Pares Positivos**:
  - ("gato", "o")
  - ("gato", "senta")
  - ("senta", "gato")
  - ("senta", "no")
  - ("no", "senta")
  - ("no", "tapete")
  - ("tapete", "no")

### Passo 2: Inicialização dos Vetores

Vamos trabalhar com vetores de dimensão 2 para simplificar.

- **Vetores de Palavras**:

| Palavra | $\mathbf{v}_w$ |
| ------- | -------------- |
| o       | [0.5, 0.1]     |
| gato    | [0.2, -0.3]    |
| senta   | [-0.4, 0.2]    |
| no      | [0.3, 0.4]     |
| tapete  | [-0.2, -0.5]   |

### Passo 3: Cálculo do Termo Positivo

Para o par ("gato", "senta"):

- **Produto Interno**: $\mathbf{v}_{gato}^\top \mathbf{v}_{senta} = (0.2)(-0.4) + (-0.3)(0.2) = -0.08$
- **Probabilidade Positiva**: $\sigma(-0.08) = \frac{1}{1 + e^{0.08}} \approx 0.4800$
- **Log-Probabilidade**: $\log(0.4800) \approx -0.7339$

### Passo 4: Amostragem Negativa

Suponha $K = 2$ e que as palavras negativas amostradas são "o" e "tapete".

Para cada palavra negativa $w_k$:

1. **Para "o"**:
   - **Produto Interno**: $\mathbf{v}_{o}^\top \mathbf{v}_{senta} = (0.5)(-0.4) + (0.1)(0.2) = -0.18$
   - **Probabilidade Negativa**: $\sigma(0.18) = \frac{1}{1 + e^{-0.18}} \approx 0.5448$
   - **Log-Probabilidade**: $\log(1 - 0.5448) = \log(0.4552) \approx -0.7885$

2. **Para "tapete"**:
   - **Produto Interno**: $\mathbf{v}_{tapete}^\top \mathbf{v}_{senta} = (-0.2)(-0.4) + (-0.5)(0.2) = 0.08 - 0.10 = -0.02$
   - **Probabilidade Negativa**: $\sigma(0.02) \approx 0.5050$
   - **Log-Probabilidade**: $\log(1 - 0.5050) = \log(0.4950) \approx -0.7028$

### Passo 5: Atualização dos Vetores

- **Gradiente em Relação a $\mathbf{v}_{gato}$**:
  - $\delta = \sigma(\mathbf{v}_{gato}^\top \mathbf{v}_{senta}) - 1 = 0.4800 - 1 = -0.5200$
  - Atualize $\mathbf{v}_{gato} = \mathbf{v}_{gato} - \eta \cdot \delta \cdot \mathbf{v}_{senta}$

- **Gradiente em Relação a $\mathbf{v}_{senta}$**:
  - Similarmente, atualize considerando contribuições dos termos positivo e negativos.

- **Onde** $\eta$ é a taxa de aprendizado.

### Observações

- **Simplificações**: Este exemplo simplifica muitos aspectos para fins ilustrativos.
- **Dimensionalidade**: Em aplicações reais, os vetores têm dimensões altas (por exemplo, 100 ou 300).

## Propriedades Matemáticas do SGNS

### Eficiência Computacional

- **Redução do Custo Computacional**: Ao invés de computar probabilidades sobre todo o vocabulário (como no Softmax), o Negative Sampling considera apenas um número limitado de palavras negativas ($K$), tornando o treinamento muito mais rápido.

### Captura de Semântica e Sintaxe

- **Aprendizado de Representações**: Os embeddings resultantes refletem relações semânticas e sintáticas entre palavras, posicionando palavras similares próximas no espaço vetorial.

### Uso da Distribuição de Amostragem

- **Distribuição $P_n(w)$**: A escolha de $P_n(w)$ é crucial. Uma distribuição comum é elevar a frequência das palavras a uma potência entre 0 e 1 (por exemplo, 0.75) para suavizar a distribuição.

### Função Objetivo Suavizada

- **Propriedades de Convexidade**: Embora a função objetivo não seja convexa, a suavidade da função sigmoide e a natureza estocástica do algoritmo permitem uma convergência eficiente em práticas.

## Conclusão

O SGNS é uma técnica fundamental para o aprendizado de representações de palavras em linguagens naturais. Ao focar em distinguir entre contextos reais e negativos, o modelo captura eficientemente as relações entre palavras sem a necessidade de computar probabilidades sobre todo o vocabulário.

## Referências

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. *arXiv preprint arXiv:1301.3781*.
- Goldberg, Y., & Levy, O. (2014). **Word2Vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method**. *arXiv preprint arXiv:1402.3722*.