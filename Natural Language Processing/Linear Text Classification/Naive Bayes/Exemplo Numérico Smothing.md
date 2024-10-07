# Exemplo Numérico: Aplicação do Laplace Smoothing em Classificação de Texto

Para ilustrar os conceitos explorados anteriormente, vamos considerar um exemplo numérico simples que demonstra o problema das probabilidades zero e como o **Laplace Smoothing** pode resolvê-lo.

## Cenário

Considere um conjunto de treinamento com documentos pertencentes a duas classes:

- **Classe A**: Documentos sobre *Esportes*.
- **Classe B**: Documentos sobre *Política*.

O vocabulário consiste em quatro palavras:

1. $w_1$: "futebol"
2. $w_2$: "eleição"
3. $w_3$: "gol"
4. $w_4$: "governo"

As contagens de palavras nos documentos de treinamento são:

| Palavra | Classe A (Esportes) | Classe B (Política) |
| ------- | ------------------- | ------------------- |
| $w_1$   | 10                  | 0                   |
| $w_2$   | 0                   | 15                  |
| $w_3$   | 5                   | 0                   |
| $w_4$   | 0                   | 5                   |

## Cálculo das Probabilidades sem Smoothing

### Classe A (Esportes)

Total de palavras na Classe A:

$$
N_{\text{A}} = 10 + 0 + 5 + 0 = 15
$$

Probabilidades condicionais:

- $P(w_1 | \text{A}) = \dfrac{10}{15} = 0.6667$
- $P(w_2 | \text{A}) = \dfrac{0}{15} = 0$
- $P(w_3 | \text{A}) = \dfrac{5}{15} = 0.3333$
- $P(w_4 | \text{A}) = \dfrac{0}{15} = 0$

### Classe B (Política)

Total de palavras na Classe B:

$$
N_{\text{B}} = 0 + 15 + 0 + 5 = 20
$$

Probabilidades condicionais:

- $P(w_1 | \text{B}) = \dfrac{0}{20} = 0$
- $P(w_2 | \text{B}) = \dfrac{15}{20} = 0.75$
- $P(w_3 | \text{B}) = \dfrac{0}{20} = 0$
- $P(w_4 | \text{B}) = \dfrac{5}{20} = 0.25$

## Problema das Probabilidades Zero

Considere um documento de teste que contém as palavras:

- "futebol" ($w_1$): aparece 1 vez
- "governo" ($w_4$): aparece 1 vez

Queremos classificar este documento entre as classes A e B.

### Cálculo da Verossimilhança sem Smoothing

**Classe A**:

$$
P(\text{documento} | \text{A}) = P(w_1 | \text{A}) \times P(w_4 | \text{A}) = 0.6667 \times 0 = 0
$$

**Classe B**:

$$
P(\text{documento} | \text{B}) = P(w_1 | \text{B}) \times P(w_4 | \text{B}) = 0 \times 0.25 = 0
$$

Ambas as probabilidades são zero devido às probabilidades condicionais zero.

## Aplicação do Laplace Smoothing

Vamos aplicar o Laplace Smoothing com $\alpha = 1$.

### Ajuste das Contagens com Smoothing

#### Classe A (Esportes)

- $c'(w_1, \text{A}) = 10 + 1 = 11$
- $c'(w_2, \text{A}) = 0 + 1 = 1$
- $c'(w_3, \text{A}) = 5 + 1 = 6$
- $c'(w_4, \text{A}) = 0 + 1 = 1$

Novo total de palavras:

$$
N'_{\text{A}} = 11 + 1 + 6 + 1 = 19
$$

Probabilidades condicionais suavizadas:

- $P'(w_1 | \text{A}) = \dfrac{11}{19} \approx 0.5789$
- $P'(w_2 | \text{A}) = \dfrac{1}{19} \approx 0.0526$
- $P'(w_3 | \text{A}) = \dfrac{6}{19} \approx 0.3158$
- $P'(w_4 | \text{A}) = \dfrac{1}{19} \approx 0.0526$

#### Classe B (Política)

- $c'(w_1, \text{B}) = 0 + 1 = 1$
- $c'(w_2, \text{B}) = 15 + 1 = 16$
- $c'(w_3, \text{B}) = 0 + 1 = 1$
- $c'(w_4, \text{B}) = 5 + 1 = 6$

Novo total de palavras:

$$
N'_{\text{B}} = 1 + 16 + 1 + 6 = 24
$$

Probabilidades condicionais suavizadas:

- $P'(w_1 | \text{B}) = \dfrac{1}{24} \approx 0.0417$
- $P'(w_2 | \text{B}) = \dfrac{16}{24} \approx 0.6667$
- $P'(w_3 | \text{B}) = \dfrac{1}{24} \approx 0.0417$
- $P'(w_4 | \text{B}) = \dfrac{6}{24} = 0.25$

## Cálculo da Verossimilhança com Smoothing

### Classe A (Esportes)

$$
P'(\text{documento} | \text{A}) = P'(w_1 | \text{A}) \times P'(w_4 | \text{A}) = 0.5789 \times 0.0526 \approx 0.0305
$$

### Classe B (Política)

$$
P'(\text{documento} | \text{B}) = P'(w_1 | \text{B}) \times P'(w_4 | \text{B}) = 0.0417 \times 0.25 \approx 0.0104
$$

## Cálculo das Probabilidades Posteriores

Assumindo probabilidades a priori iguais para ambas as classes:

$$
P(\text{A}) = P(\text{B}) = 0.5
$$

Probabilidades posteriores (antes da normalização):

- $P'(\text{A} | \text{documento}) \propto 0.5 \times 0.0305 = 0.0152$
- $P'(\text{B} | \text{documento}) \propto 0.5 \times 0.0104 = 0.0052$

Após a normalização:

- $P'(\text{A} | \text{documento}) = \dfrac{0.0152}{0.0152 + 0.0052} \approx 0.7451$
- $P'(\text{B} | \text{documento}) = \dfrac{0.0052}{0.0152 + 0.0052} \approx 0.2549$

**Decisão**: O documento é classificado como pertencente à **Classe A (Esportes)**.

## Impacto do Hiperparâmetro $\alpha$

Vamos explorar como a variação de $\alpha$ afeta as probabilidades.

### Com $\alpha = 0.5$

#### Recalculando as Contagens

##### Classe A (Esportes)

- $c'(w_1, \text{A}) = 10 + 0.5 = 10.5$
- $c'(w_4, \text{A}) = 0 + 0.5 = 0.5$

Novo total:

$$
N'_{\text{A}} = 10.5 + 0.5 = 11
$$

Probabilidades suavizadas:

- $P'(w_1 | \text{A}) = \dfrac{10.5}{11} \approx 0.9545$
- $P'(w_4 | \text{A}) = \dfrac{0.5}{11} \approx 0.0455$

##### Classe B (Política)

- $c'(w_1, \text{B}) = 0 + 0.5 = 0.5$
- $c'(w_4, \text{B}) = 5 + 0.5 = 5.5$

Novo total:

$$
N'_{\text{B}} = 0.5 + 5.5 = 6
$$

Probabilidades suavizadas:

- $P'(w_1 | \text{B}) = \dfrac{0.5}{6} \approx 0.0833$
- $P'(w_4 | \text{B}) = \dfrac{5.5}{6} \approx 0.9167$

#### Recalculando as Verossimilhanças

**Classe A**:

$$
P'(\text{documento} | \text{A}) = 0.9545 \times 0.0455 \approx 0.0434
$$

**Classe B**:

$$
P'(\text{documento} | \text{B}) = 0.0833 \times 0.9167 \approx 0.0764
$$

#### Probabilidades Posteriores

- $P'(\text{A} | \text{documento}) \propto 0.5 \times 0.0434 = 0.0217$
- $P'(\text{B} | \text{documento}) \propto 0.5 \times 0.0764 = 0.0382$

Normalizando:

- $P'(\text{A} | \text{documento}) = \dfrac{0.0217}{0.0217 + 0.0382} \approx 0.3629$
- $P'(\text{B} | \text{documento}) = \dfrac{0.0382}{0.0217 + 0.0382} \approx 0.6371$

**Decisão**: Com $\alpha = 0.5$, o documento é classificado como pertencente à **Classe B (Política)**.

### Observações

- **Influência de $\alpha$**: Valores menores de $\alpha$ resultam em probabilidades condicionais mais extremas, refletindo mais fortemente as contagens observadas.
- **Trade-off**: Um $\alpha$ muito pequeno pode levar a overfitting, enquanto um $\alpha$ muito grande pode introduzir viés significativo.

## Conclusão do Exemplo

Este exemplo numérico demonstra:

1. **Problema das Probabilidades Zero**: Sem smoothing, não podemos calcular probabilidades significativas, pois muitas delas são zero.
2. **Laplace Smoothing**: Ao adicionar um pseudoconte $\alpha$, evitamos probabilidades zero e obtemos estimativas mais confiáveis.
3. **Impacto de $\alpha$**: A escolha de $\alpha$ afeta diretamente as probabilidades condicionais e, consequentemente, a classificação final.

## Aplicação Prática

Este exemplo reflete cenários reais na classificação de texto, onde o Laplace Smoothing é essencial para lidar com dados esparsos e vocabulários grandes. A compreensão do impacto do hiperparâmetro $\alpha$ permite ajustar o modelo para melhor desempenho em diferentes conjuntos de dados.