Aqui está um resumo detalhado e avançado sobre as limitações dos modelos n-gram:

## Limitações dos Modelos N-Gram

<imagem: Um gráfico mostrando o trade-off entre viés e variância para diferentes valores de n em modelos n-gram>

### Introdução

Os modelos n-gram são uma abordagem fundamental na modelagem de linguagem estatística, baseando-se na suposição de que a probabilidade de uma palavra depende apenas das n-1 palavras anteriores [1]. Embora simples e eficazes em muitos casos, esses modelos apresentam limitações significativas que impactam seu desempenho e aplicabilidade em tarefas de processamento de linguagem natural mais complexas [2].

### Conceitos Fundamentais

| Conceito          | Explicação                                                   |
| ----------------- | ------------------------------------------------------------ |
| **Modelo N-Gram** | Um modelo probabilístico que prevê a ocorrência de uma palavra com base nas n-1 palavras anteriores [3]. |
| **Viés (Bias)**   | Erro introduzido por suposições simplificadoras do modelo [4]. |
| **Variância**     | Sensibilidade do modelo a flutuações nos dados de treinamento [5]. |

> ⚠️ **Nota Importante**: O trade-off entre viés e variância é crucial na compreensão das limitações dos modelos n-gram [6].

### Limitações para Pequenos Valores de N

<imagem: Ilustração de um modelo bigram (n=2) falhando em capturar dependências de longo alcance>

Quando n é muito pequeno, os modelos n-gram sofrem de alto viés, o que leva a várias limitações:

1. **Incapacidade de Capturar Dependências de Longo Alcance**: 
   Modelos com n pequeno não conseguem modelar relações entre palavras distantes na sequência [7]. Por exemplo, em um modelo bigram (n=2), a probabilidade de uma palavra depende apenas da palavra imediatamente anterior:

   $$P(w_i|w_1^{i-1}) \approx P(w_i|w_{i-1})$$

   Isso falha em capturar dependências importantes em frases como "The cat, which was sitting on the mat, slowly walked away", onde "cat" e "walked" estão relacionados, mas distantes [8].

2. **Perda de Contexto Semântico**: 
   A simplificação excessiva do contexto leva à perda de nuances semânticas e pragmáticas importantes [9]. Por exemplo, um modelo trigram não conseguiria diferenciar adequadamente o significado de "bank" em "river bank" e "bank account".

3. **Problemas com Concordância Gramatical**: 
   Modelos com n pequeno frequentemente falham em manter a concordância gramatical em frases longas [10]. Por exemplo, um modelo bigram teria dificuldade em manter a concordância de número entre sujeito e verbo em uma frase longa.

#### Perguntas Teóricas

1. Derive matematicamente a expressão para o viés de um modelo bigram em relação a um modelo de linguagem ideal que captura todas as dependências. Considere uma sequência de m palavras.

2. Como o aumento de n afeta a capacidade do modelo de capturar dependências de longo alcance? Prove matematicamente por que mesmo aumentar n não resolve completamente este problema.

3. Demonstre teoricamente por que um modelo n-gram com n fixo sempre terá um viés intrínseco, independentemente da quantidade de dados de treinamento.

### Limitações para Grandes Valores de N

<imagem: Gráfico mostrando o aumento exponencial do número de parâmetros conforme n aumenta>

Quando n é muito grande, os modelos n-gram enfrentam problemas de alta variância e esparsidade de dados:

1. **Explosão Combinatória de Parâmetros**: 
   O número de parâmetros em um modelo n-gram cresce exponencialmente com n [11]. Para um vocabulário de tamanho V, o número de possíveis n-gramas é V^n. Isso leva a:

   $$\text{Número de Parâmetros} = O(V^n)$$

   Por exemplo, com um vocabulário de 10.000 palavras e n=5, teríamos 10^20 parâmetros potenciais [12].

2. **Esparsidade de Dados**: 
   Com o aumento de n, a maioria das sequências possíveis nunca será observada nos dados de treinamento, levando a estimativas de probabilidade não confiáveis [13]. Este problema é conhecido como "curse of dimensionality" em estatística.

3. **Overfitting**: 
   Modelos com n grande tendem a se ajustar demais aos dados de treinamento, memorizando sequências específicas em vez de aprender padrões generalizáveis [14]. Isso resulta em:

   $$P(w_i|w_{i-n+1}^{i-1}) = \frac{\text{count}(w_{i-n+1}^i)}{\text{count}(w_{i-n+1}^{i-1})} \approx 0 \text{ ou } 1$$

   para muitas sequências nos dados de teste.

4. **Ineficiência Computacional**: 
   O armazenamento e processamento de um grande número de n-gramas torna-se computacionalmente inviável para valores altos de n [15].

> ❗ **Ponto de Atenção**: A esparsidade de dados é um dos maiores desafios em modelos n-gram de alta ordem, levando a estimativas de probabilidade não confiáveis [16].

#### Perguntas Teóricas

1. Derive uma expressão para a probabilidade de observar um n-grama específico em um corpus de tamanho M, assumindo uma distribuição uniforme de palavras. Como essa probabilidade muda com o aumento de n?

2. Prove que, para qualquer corpus finito, existe um valor de n a partir do qual o modelo n-gram terá perplexidade perfeita no conjunto de treinamento, mas péssimo desempenho em dados não vistos.

3. Desenvolva uma análise teórica do trade-off entre viés e variância em modelos n-gram, expressando matematicamente como esses fatores mudam com o aumento de n.

### Técnicas de Suavização e Descontagem

Para mitigar algumas das limitações dos modelos n-gram, técnicas de suavização e descontagem foram desenvolvidas [17]:

1. **Suavização de Lidstone**:
   Adiciona um pseudo-contagem α a todas as contagens de n-gramas [18]:

   $$P_{\text{smooth}}(w_m | w_{m-1}) = \frac{\text{count}(w_{m-1}, w_m) + \alpha}{\sum_{w' \in V} \text{count}(w_{m-1}, w') + V\alpha}$$

2. **Descontagem Absoluta**:
   Subtrai uma quantidade fixa d de cada contagem não-zero [19]:

   $$c^*(i,j) = c(i,j) - d$$

3. **Backoff de Katz**:
   Utiliza n-gramas de ordem inferior quando n-gramas de ordem superior não são observados [20]:

   $$P_{\text{Katz}}(w_i|w_{i-n+1}^{i-1}) = \begin{cases}
   \frac{c^*(w_{i-n+1}^i)}{c(w_{i-n+1}^{i-1})} & \text{se } c(w_{i-n+1}^i) > 0 \\
   \alpha(w_{i-n+1}^{i-1}) P_{\text{Katz}}(w_i|w_{i-n+2}^{i-1}) & \text{caso contrário}
   \end{cases}$$

Estas técnicas ajudam a reduzir o problema de esparsidade, mas não resolvem completamente as limitações fundamentais dos modelos n-gram [21].

> ✔️ **Destaque**: A suavização de Kneser-Ney é considerada o estado da arte para modelagem de linguagem n-gram, oferecendo um equilíbrio eficaz entre suavização e backoff [22].

### Conclusão

Os modelos n-gram, apesar de sua simplicidade e eficácia em certas aplicações, enfrentam limitações significativas tanto para valores pequenos quanto grandes de n [23]. O trade-off entre viés e variância é central para entender essas limitações [24]. Enquanto técnicas de suavização e descontagem oferecem melhorias, elas não resolvem completamente os problemas fundamentais [25]. Estas limitações motivaram o desenvolvimento de modelos de linguagem mais avançados, como os baseados em redes neurais recorrentes e transformers, que são capazes de capturar dependências de longo alcance e lidar melhor com a esparsidade de dados [26].

### Perguntas Teóricas Avançadas

1. Desenvolva uma prova matemática mostrando que, para qualquer n finito, existe sempre uma classe de dependências linguísticas que um modelo n-gram não pode capturar adequadamente. Como isso se relaciona com a hierarquia de Chomsky de gramáticas formais?

2. Analise teoricamente o impacto da Lei de Zipf na eficácia dos modelos n-gram. Como a distribuição de cauda longa das palavras afeta o desempenho desses modelos para diferentes valores de n?

3. Derive uma expressão para a perplexidade esperada de um modelo n-gram em um corpus infinito gerado por uma gramática livre de contexto. Como essa perplexidade se compara à de um modelo de linguagem ideal baseado na própria gramática?

4. Prove que a suavização de Kneser-Ney é ótima sob certas suposições sobre a distribuição de palavras. Quais são essas suposições e como elas se relacionam com as características reais das linguagens naturais?

5. Desenvolva um framework teórico para quantificar a "informação perdida" em um modelo n-gram em comparação com um modelo de linguagem ideal. Como essa medida de informação perdida se relaciona com as métricas padrão de avaliação de modelos de linguagem, como perplexidade?

### Referências

[1] "In probabilistic classification, the problem is to compute the probability of a label, conditioned on the text. Let's now consider the inverse problem: computing the probability of text itself. Specifically, we will consider models that assign probability to a sequence of word tokens, p(w₁, w₂, ..., wₘ), with wₘ ∈ V." *(Trecho de Language Models_143-162.pdf.md)*

[2] "These two problems point to another bias-variance tradeoff (see § 2.2.4). A small n-gram size introduces high bias, and a large n-gram size introduces high variance." *(Trecho de Language Models_143-162.pdf.md)*

[3] "n-gram models, which compute the probability of a sequence as the product of probabilities of subsequences." *(Trecho de Language Models_143-162.pdf.md)*

[4] "When n is too small. Consider the following sentences:" *(Trecho de Language Models_143-162.pdf.md)*

[5] "When n is too big. In this case, it is hard good estimates of the n-gram parameters from our dataset, because of data sparsity." *(Trecho de Language Models_143-162.pdf.md)*

[6] "These two problems point to another bias-variance tradeoff (see § 2.2.4). A small n-gram size introduces high bias, and a large n-gram size introduces high variance." *(Trecho de Language Models_143-162.pdf.md)*

[7] "In each example, the words written in bold depend on each other: the likelihood of their depends on knowing that gorillas is plural, and the likelihood of crashed depends on knowing that the subject is a computer. If the n-grams are not big enough to capture this context, then the resulting language model would offer probabilities that are too low for these sentences, and too high for sentences that fail basic linguistic tests like number agreement." *(Trecho de Language Models_143-162.pdf.md)*

[8] "Gorillas always like to groom their friends." *(Trecho de Language Models_143-162.pdf.md)*

[9] "When n is too small. Consider the following sentences:" *(Trecho de Language Models_143-162.pdf.md)*

[10] "If the n-grams are not big enough to capture this context, then the resulting language model would offer probabilities that are too low for these sentences, and too high for sentences that fail basic linguistic tests like number agreement." *(Trecho de Language Models_143-162.pdf.md)*

[11] "When n is too big. In this case, it is hard good estimates of the n-gram parameters from our dataset, because of data sparsity. To handle the gorilla example, it is necessary to model 6-grams, which means accounting for $V^6$ events. Under a very small vocabulary of $V = 10^4$, this means estimating the probability of $10^{24}$ distinct events." *(Trecho de Language Models_143-162.pdf.md)*

[12] "Under a very small vocabulary of $V = 10^4$, this means estimating the probability of $10^{24}$ distinct events." *(Trecho de Language Models_143-162.pdf.md)*

[13] "When n is too big. In this case, it is hard good estimates of the n-gram parameters from our dataset, because of data sparsity." *(Trecho de Language Models_143-162.pdf.md)*

[14] "When n is too big. In this case, it is hard good estimates of the n-gram parameters from our dataset, because of data sparsity." *(Trecho de Language Models_143-162.pdf.md)*

[15] "When n is too big. In this case, it is hard good estimates of the n-gram parameters from our dataset, because of data sparsity." *(Trecho de Language Models_143-162.pdf.md)*

[16] "When n is too big. In this case, it is hard good estimates of the n-gram parameters from our dataset, because of data sparsity." *(Trecho de Language Models_143-162.pdf.md)*

[17] "Smoothing and discounting" *(Trecho de Language Models_143-162.pdf.md)*

[18] "A major concern in language modeling is to avoid the situation p(w) = 0, which could arise as a result of a single unseen n-gram. A similar problem arose in Naïve Bayes, and the solution was smoothing: adding imaginary "pseudo" counts. The same idea can be applied to n-gram language models, as shown here in the bigram case," *(Trecho de Language Models_143-162.pdf.md)*

[19] "Discounting "borrows" probability mass from observed n-grams and redistributes it. In Lidstone smoothing, the borrowing is done by increasing the denominator of the relative frequency estimates. The borrowed probability mass is then redistributed by increasing the numerator for all n-grams. Another approach would be to borrow the same amount of probability mass from all observed n-grams, and redistribute it among only the unobserved n-grams. This is called absolute discounting." *(Trecho de Language Models_143-162.pdf.md)*

[20] "Discounting reserves some probability mass from the observed data, and we need not redistribute this probability mass equally. Instead, we can backoff to a lower-order language model: if you have trigrams, use trigrams; if you don't have trigrams, use bigrams; if you don't even have bigrams, use unigrams. This is called Katz backoff." *(Trecho de Language Models_143-162.pdf.md)*

[21] "Smoothing and discounting" *(Trecho de Language Models_143-162.pdf.md)*

[22] "Kneser-Ney smoothing is based on absolute discounting, but it redistributes the resulting probability mass in