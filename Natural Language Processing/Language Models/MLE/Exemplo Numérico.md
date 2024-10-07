### Exemplo Numérico Detalhado

Para aprofundar a compreensão da relação entre as contagens de n-gramas e a estimação de probabilidades por Máxima Verossimilhança (MLE), vamos considerar um corpus mais extenso e complexo. Este exemplo ilustrará como as frequências observadas afetam as probabilidades estimadas e destacará potenciais desafios, como a esparsidade de dados.

#### Corpus de Treinamento

Considere o seguinte corpus composto por várias frases:

1. "o gato está no tapete"
2. "o cachorro está no jardim"
3. "o gato e o cachorro são amigos"
4. "o gato gosta de peixe"
5. "o cachorro gosta de osso"
6. "o gato está no jardim"
7. "o cachorro está no tapete"
8. "o peixe está no aquário"
9. "o osso está no prato"

Este corpus contém uma variedade de palavras e combinações, permitindo a análise de bigramas e trigramas.

#### Passo 1: Extração dos N-gramas e Contagem de Ocorrências

##### **1.1 Contagem de Unigramas (palavras individuais)**

Contamos o número de ocorrências de cada palavra no corpus.

| Palavra  | Contagem |
| -------- | -------- |
| o        | 18       |
| gato     | 5        |
| está     | 6        |
| no       | 6        |
| tapete   | 2        |
| cachorro | 5        |
| jardim   | 2        |
| e        | 1        |
| são      | 1        |
| amigos   | 1        |
| gosta    | 2        |
| de       | 2        |
| peixe    | 2        |
| osso     | 2        |
| aquário  | 1        |
| prato    | 1        |

##### **1.2 Contagem de Bigrams (n=2)**

Extraímos todos os bigramas e contamos suas ocorrências.

| Bigrama               | Contagem |
| --------------------- | -------- |
| ("o", "gato")         | 4        |
| ("gato", "está")      | 2        |
| ("está", "no")        | 6        |
| ("no", "tapete")      | 2        |
| ("o", "cachorro")     | 4        |
| ("cachorro", "está")  | 2        |
| ("no", "jardim")      | 2        |
| ("gato", "e")         | 1        |
| ("e", "o")            | 1        |
| ("cachorro", "são")   | 1        |
| ("são", "amigos")     | 1        |
| ("gato", "gosta")     | 1        |
| ("gosta", "de")       | 2        |
| ("de", "peixe")       | 1        |
| ("cachorro", "gosta") | 1        |
| ("de", "osso")        | 1        |
| ("gato", "está")      | 2        |
| ("cachorro", "está")  | 2        |
| ("o", "peixe")        | 1        |
| ("peixe", "está")     | 1        |
| ("no", "aquário")     | 1        |
| ("o", "osso")         | 1        |
| ("osso", "está")      | 1        |
| ("no", "prato")       | 1        |

##### **1.3 Contagem de Trigramas (n=3)**

Extraímos todos os trigramas e contamos suas ocorrências.

| Trigrama                      | Contagem |
| ----------------------------- | -------- |
| ("o", "gato", "está")         | 2        |
| ("gato", "está", "no")        | 2        |
| ("está", "no", "tapete")      | 2        |
| ("o", "cachorro", "está")     | 2        |
| ("cachorro", "está", "no")    | 2        |
| ("está", "no", "jardim")      | 2        |
| ("o", "gato", "e")            | 1        |
| ("gato", "e", "o")            | 1        |
| ("e", "o", "cachorro")        | 1        |
| ("o", "cachorro", "são")      | 1        |
| ("cachorro", "são", "amigos") | 1        |
| ("o", "gato", "gosta")        | 1        |
| ("gato", "gosta", "de")       | 1        |
| ("gosta", "de", "peixe")      | 1        |
| ("o", "cachorro", "gosta")    | 1        |
| ("cachorro", "gosta", "de")   | 1        |
| ("gosta", "de", "osso")       | 1        |
| ("o", "peixe", "está")        | 1        |
| ("peixe", "está", "no")       | 1        |
| ("está", "no", "aquário")     | 1        |
| ("o", "osso", "está")         | 1        |
| ("osso", "está", "no")        | 1        |
| ("está", "no", "prato")       | 1        |

#### Passo 2: Cálculo das Probabilidades Condicionais

##### **2.1 Probabilidades Condicionais de Bigrams**

Usando a fórmula MLE para bigramas:

$$
P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}
$$

Calculamos as probabilidades condicionais para alguns bigramas selecionados.

1. **$P(\text{"gato"} \mid \text{"o"})$**

$$
C(\text{"o"}, \text{"gato"}) = 4 \\
C(\text{"o"}) = 18 \\
P(\text{"gato"} \mid \text{"o"}) = \frac{4}{18} \approx 0{,}222
$$

2. **$P(\text{"cachorro"} \mid \text{"o"})$**

$$
C(\text{"o"}, \text{"cachorro"}) = 4 \\
P(\text{"cachorro"} \mid \text{"o"}) = \frac{4}{18} \approx 0{,}222
$$

3. **$P(\text{"peixe"} \mid \text{"o"})$**

$$
C(\text{"o"}, \text{"peixe"}) = 1 \\
P(\text{"peixe"} \mid \text{"o"}) = \frac{1}{18} \approx 0{,}056
$$

4. **$P(\text{"gosta"} \mid \text{"gato"})$**

$$
C(\text{"gato"}, \text{"gosta"}) = 1 \\
C(\text{"gato"}) = 5 \\
P(\text{"gosta"} \mid \text{"gato"}) = \frac{1}{5} = 0{,}2
$$

5. **$P(\text{"está"} \mid \text{"gato"})$**

$$
C(\text{"gato"}, \text{"está"}) = 2 \\
P(\text{"está"} \mid \text{"gato"}) = \frac{2}{5} = 0{,}4
$$

6. **$P(\text{"e"} \mid \text{"gato"})$**

$$
C(\text{"gato"}, \text{"e"}) = 1 \\
P(\text{"e"} \mid \text{"gato"}) = \frac{1}{5} = 0{,}2
$$

7. **$P(\text{"está"} \mid \text{"cachorro"})$**

$$
C(\text{"cachorro"}, \text{"está"}) = 2 \\
C(\text{"cachorro"}) = 5 \\
P(\text{"está"} \mid \text{"cachorro"}) = \frac{2}{5} = 0{,}4
$$

8. **$P(\text{"gosta"} \mid \text{"cachorro"})$**

$$
C(\text{"cachorro"}, \text{"gosta"}) = 1 \\
P(\text{"gosta"} \mid \text{"cachorro"}) = \frac{1}{5} = 0{,}2
$$

##### **2.2 Probabilidades Condicionais de Trigramas**

Usando a fórmula MLE para trigramas:

$$
P(w_n \mid w_{n-2}, w_{n-1}) = \frac{C(w_{n-2}, w_{n-1}, w_n)}{C(w_{n-2}, w_{n-1})}
$$

Calculamos as probabilidades condicionais para alguns trigramas.

1. **$P(\text{"está"} \mid \text{"o"}, \text{"gato"})$**

$$
C(\text{"o"}, \text{"gato"}, \text{"está"}) = 2 \\
C(\text{"o"}, \text{"gato"}) = 4 \\
P(\text{"está"} \mid \text{"o"}, \text{"gato"}) = \frac{2}{4} = 0{,}5
$$

2. **$P(\text{"e"} \mid \text{"o"}, \text{"gato"})$**

$$
C(\text{"o"}, \text{"gato"}, \text{"e"}) = 1 \\
P(\text{"e"} \mid \text{"o"}, \text{"gato"}) = \frac{1}{4} = 0{,}25
$$

3. **$P(\text{"gosta"} \mid \text{"o"}, \text{"gato"})$**

$$
C(\text{"o"}, \text{"gato"}, \text{"gosta"}) = 1 \\
P(\text{"gosta"} \mid \text{"o"}, \text{"gato"}) = \frac{1}{4} = 0{,}25
$$

4. **$P(\text{"está"} \mid \text{"o"}, \text{"cachorro"})$**

$$
C(\text{"o"}, \text{"cachorro"}, \text{"está"}) = 2 \\
C(\text{"o"}, \text{"cachorro"}) = 4 \\
P(\text{"está"} \mid \text{"o"}, \text{"cachorro"}) = \frac{2}{4} = 0{,}5
$$

5. **$P(\text{"gosta"} \mid \text{"o"}, \text{"cachorro"})$**

$$
C(\text{"o"}, \text{"cachorro"}, \text{"gosta"}) = 1 \\
P(\text{"gosta"} \mid \text{"o"}, \text{"cachorro"}) = \frac{1}{4} = 0{,}25
$$

6. **$P(\text{"são"} \mid \text{"o"}, \text{"cachorro"})$**

$$
C(\text{"o"}, \text{"cachorro"}, \text{"são"}) = 1 \\
P(\text{"são"} \mid \text{"o"}, \text{"cachorro"}) = \frac{1}{4} = 0{,}25
$$

#### Passo 3: Interpretação e Análise dos Resultados

##### **3.1 Probabilidades Condicionais de Bigrams**

- **Após "o"**:

  - A probabilidade de ocorrer "gato" ou "cachorro" após "o" é a mesma ($\approx 0{,}222$).
  - Outras palavras, como "peixe" e "osso", têm probabilidades menores, refletindo suas menores frequências após "o".

- **Após "gato"**:

  - "está" tem a maior probabilidade ($0{,}4$), indicando que "gato está" é uma sequência comum no corpus.
  - "gosta" e "e" têm probabilidades menores, mostrando variação no comportamento após "gato".

##### **3.2 Probabilidades Condicionais de Trigramas**

- **Após "o gato"**:

  - "está" tem a maior probabilidade ($0{,}5$), indicando que "o gato está" é uma sequência frequente.
  - "gosta" e "e" têm probabilidades menores, mas significativas.

- **Após "o cachorro"**:

  - Situação semelhante, com "está" sendo a sequência mais provável, seguido por "gosta" e "são".

##### **3.3 Observações Gerais**

- As probabilidades refletem diretamente as contagens observadas no corpus.
- Sequências com maiores frequências têm probabilidades condicionais maiores.
- A distribuição das probabilidades condicionais mostra como o modelo MLE se baseia nas frequências relativas.

#### Passo 4: Cálculo da Probabilidade de uma Sentença

Vamos calcular a probabilidade da sentença "o gato está no jardim" usando bigramas.

$$
P(\text{"o gato está no jardim"}) = P(\text{"o"}) \times P(\text{"gato"} \mid \text{"o"}) \times P(\text{"está"} \mid \text{"gato"}) \times P(\text{"no"} \mid \text{"está"}) \times P(\text{"jardim"} \mid \text{"no"})
$$

Assumindo que $P(\text{"o"})$ é proporcional à sua frequência relativa:

$$
P(\text{"o"}) = \frac{C(\text{"o"})}{\text{Total de palavras}} = \frac{18}{52} \approx 0{,}346
$$

Usando as probabilidades condicionais calculadas:

- $P(\text{"gato"} \mid \text{"o"}) \approx 0{,}222$
- $P(\text{"está"} \mid \text{"gato"}) = 0{,}4$
- $P(\text{"no"} \mid \text{"está"})$

  Contamos $C(\text{"está"}, \text{"no"}) = 6$ e $C(\text{"está"}) = 6$:

  $$
  P(\text{"no"} \mid \text{"está"}) = \frac{6}{6} = 1
  $$

- $P(\text{"jardim"} \mid \text{"no"})$

  $C(\text{"no"}, \text{"jardim"}) = 2$ e $C(\text{"no"}) = 6$:

  $$
  P(\text{"jardim"} \mid \text{"no"}) = \frac{2}{6} \approx 0{,}333
  $$

Calculando a probabilidade total:

$$
P(\text{"o gato está no jardim"}) \approx 0{,}346 \times 0{,}222 \times 0{,}4 \times 1 \times 0{,}333 \approx 0{,}0103
$$

#### Passo 5: Considerações sobre Esparsidade de Dados

- **N-gramas Não Observados**: Se tentarmos calcular a probabilidade de uma sequência que não ocorre no corpus, como "o gato come peixe", alguns bigramas terão contagem zero, resultando em probabilidade zero.

- **Exemplo**:

  - $C(\text{"gato"}, \text{"come"}) = 0$
  - $P(\text{"come"} \mid \text{"gato"}) = \frac{0}{5} = 0$

- Isso ilustra um dos principais problemas da MLE: atribuir probabilidade zero a eventos não observados, mesmo que sejam linguisticamente plausíveis.

#### Passo 6: Implicações e Necessidade de Suavização

- **Probabilidades Zero**: Eventos não observados recebem probabilidade zero, afetando negativamente a capacidade de generalização do modelo.

- **Superestimação de Probabilidades**: Eventos observados poucas vezes podem ter suas probabilidades superestimadas em relação a eventos não observados.

- **Soluções**: Técnicas de suavização, como Laplace, Lidstone, interpolação linear e Kneser-Ney, são necessárias para ajustar as probabilidades e melhorar a performance do modelo em dados de teste.

#### Passo 7: Aplicação de Suavização (Opcional)

Como exemplo, aplicando a suavização de Laplace (Add-One) para ajustar as probabilidades dos bigramas após "gato":

$$
P_{\text{Laplace}}(w_n \mid \text{"gato"}) = \frac{C(\text{"gato"}, w_n) + 1}{C(\text{"gato"}) + V}
$$

Onde $V$ é o tamanho do vocabulário (número de palavras únicas no corpus). Contamos $V = 16$.

- **$P_{\text{Laplace}}(\text{"come"} \mid \text{"gato"})$**

  $$
  C(\text{"gato"}, \text{"come"}) = 0 \\
  P_{\text{Laplace}}(\text{"come"} \mid \text{"gato"}) = \frac{0 + 1}{5 + 16} = \frac{1}{21} \approx 0{,}0476
  $$

Agora, "come" tem uma probabilidade não zero após "gato", permitindo que o modelo atribua uma probabilidade positiva a sequências não observadas.

### Conclusão do Exemplo

Este exemplo numérico complexo demonstra:

- **Relação Direta entre Contagens e Probabilidades**: As probabilidades estimadas por MLE refletem diretamente as frequências observadas no corpus.

- **Impacto da Frequência nas Estimativas**: N-gramas com contagens maiores recebem probabilidades condicionais maiores.

- **Desafios da MLE**: Problemas como probabilidades zero para n-gramas não observados e superestimação de n-gramas raros.

- **Necessidade de Suavização**: Para melhorar a capacidade de generalização do modelo, técnicas de suavização são essenciais.

Este exemplo ilustra a aplicação prática da MLE em um contexto mais complexo, evidenciando tanto suas vantagens quanto suas limitações, e prepara o terreno para discussões sobre métodos avançados de estimação e aprimoramento de modelos de linguagem.

---

### Referências

[1] **Jurafsky, D., & Martin, J. H.** (2009). *Speech and Language Processing*. Prentice Hall.

[2] **Manning, C. D., & Schütze, H.** (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.

[3] **Chen, S. F., & Goodman, J.** (1996). *An Empirical Study of Smoothing Techniques for Language Modeling*. Proceedings of the 34th Annual Meeting on Association for Computational Linguistics.