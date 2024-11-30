# Derivação Matemática Detalhada do Direct Preference Optimization (DPO)

Vamos derivar passo a passo a função objetivo do Direct Preference Optimization (DPO) a partir dos conceitos fundamentais. A derivação seguirá os seguintes tópicos:

1. Modelo Bradley-Terry
2. Função de Recompensa Latente
3. Reparametrização DPO
4. Cancelamento do termo $Z(x)$
5. Probabilidade de Preferência em Termos da Política
6. Modelo de Preferência DPO
7. Objetivo de Máxima Verossimilhança
8. Perda de Entropia Cruzada Binária
9. Função Objetivo Final DPO

## 1. Modelo Bradley-Terry

O modelo Bradley-Terry é um modelo estatístico para dados de comparações pareadas. Ele modela a probabilidade de um item ser preferido a outro com base em seus escores de habilidade ou qualidade.

**Definição:**

Dado dois itens $i$ e $j$, com escores de habilidade $r_i$ e $r_j$, a probabilidade de $i$ ser preferido a $j$ é:

$$
p(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}}
$$

No contexto de modelos de linguagem, consideramos respostas $y_1$ e $y_2$ para um contexto $x$, e uma função de recompensa latente $r^*(x, y)$ que avalia a qualidade de uma resposta $y$ no contexto $x$.

**Aplicação ao DPO:**

A probabilidade de preferir a resposta $y_1$ à resposta $y_2$, dado o contexto $x$, é modelada como:

$$
p^*(y_1 \succ y_2 | x) = \sigma\left(r^*(x, y_1) - r^*(x, y_2)\right)
$$

Onde $\sigma$ é a função sigmoide:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Observação:**

A diferença $r^*(x, y_1) - r^*(x, y_2)$ captura a vantagem relativa de $y_1$ sobre $y_2$ em termos da recompensa latente.

## 2. Função de Recompensa Latente

A função de recompensa latente $r^*(x, y)$ é uma função hipotética que atribui um valor escalar à qualidade da resposta $y$ no contexto $x$. Embora não observável diretamente, ela representa as preferências subjacentes de um avaliador ideal.

**Características:**

- **Latente:** Não é diretamente observável.
- **Contextual:** Depende tanto do contexto $x$ quanto da resposta $y$.
- **Escalar:** Retorna um valor numérico que permite comparar diferentes respostas.

## 3. Reparametrização DPO

O objetivo é expressar a função de recompensa latente em termos das políticas de linguagem, permitindo a otimização direta da política sem estimar explicitamente $r^*$.

**Definição:**

Assumimos que a função de recompensa latente pode ser reparametrizada como:

$$
r^*(x, y) = \beta \left( \log \pi^*(y | x) - \log \pi_{\text{ref}}(y | x) \right) + \beta \log Z(x)
$$

Onde:

- $\pi^*(y | x)$ é a política ótima que queremos encontrar.
- $\pi_{\text{ref}}(y | x)$ é uma política de referência conhecida.
- $\beta$ é um parâmetro de temperatura que controla a escala da recompensa.
- $Z(x)$ é um termo de normalização dependente de $x$.

**Justificativa:**

A reparametrização relaciona a recompensa latente com a diferença entre as probabilidades logarítmicas sob a política ótima e a política de referência, escalada por $\beta$.

## 4. Cancelamento do Termo $Z(x)$

Ao calcular a diferença de recompensas entre duas respostas, o termo $Z(x)$ se cancela, simplificando a expressão.

**Cálculo:**

Considerando duas respostas $y_1$ e $y_2$:

$$
\begin{align*}
r^*(x, y_1) - r^*(x, y_2) &= \beta \left( \log \pi^*(y_1 | x) - \log \pi_{\text{ref}}(y_1 | x) \right) + \beta \log Z(x) \\
&\quad - \left( \beta \left( \log \pi^*(y_2 | x) - \log \pi_{\text{ref}}(y_2 | x) \right) + \beta \log Z(x) \right) \\
&= \beta \left( \log \pi^*(y_1 | x) - \log \pi^*(y_2 | x) - \log \pi_{\text{ref}}(y_1 | x) + \log \pi_{\text{ref}}(y_2 | x) \right)
\end{align*}
$$
**Resultado Simplificado:**

$$
r^*(x, y_1) - r^*(x, y_2) = \beta \left( \log \frac{\pi^*(y_1 | x)}{\pi^*(y_2 | x)} - \log \frac{\pi_{\text{ref}}(y_1 | x)}{\pi_{\text{ref}}(y_2 | x)} \right)
$$

## 5. Probabilidade de Preferência em Termos da Política

Substituindo a diferença de recompensas na expressão da probabilidade de preferência:

$$
p^*(y_1 \succ y_2 | x) = \sigma\left( \beta \left( \log \frac{\pi^*(y_1 | x)}{\pi^*(y_2 | x)} - \log \frac{\pi_{\text{ref}}(y_1 | x)}{\pi_{\text{ref}}(y_2 | x)} \right) \right)
$$

**Simplificando:**

Podemos reorganizar a expressão:

$$
\begin{align*}
p^*(y_1 \succ y_2 | x) &= \sigma\left( \beta \left( \log \pi^*(y_1 | x) - \log \pi^*(y_2 | x) - \log \pi_{\text{ref}}(y_1 | x) + \log \pi_{\text{ref}}(y_2 | x) \right) \right) \\
&= \sigma\left( \beta \left( \left( \log \frac{\pi^*(y_1 | x)}{\pi_{\text{ref}}(y_1 | x)} \right) - \left( \log \frac{\pi^*(y_2 | x)}{\pi_{\text{ref}}(y_2 | x)} \right) \right) \right)
\end{align*}
$$
**Resultado Final:**

$$
p^*(y_1 \succ y_2 | x) = \sigma\left( \beta \left( \Delta(y_1, x) - \Delta(y_2, x) \right) \right)
$$

Onde:

$$
\Delta(y, x) = \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)}
$$

## 6. Modelo de Preferência DPO

Para tornar o modelo operacional, substituímos a política ótima $\pi^*$ por uma política parametrizada $\pi_\theta$, que queremos otimizar.

**Modelo de Preferência:**

$$
p_\theta(y_1 \succ y_2 | x) = \sigma\left( \beta \left( \Delta_\theta(y_1, x) - \Delta_\theta(y_2, x) \right) \right)
$$

Onde:

$$
\Delta_\theta(y, x) = \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}
$$

**Interpretação:**

- A probabilidade de preferir $y_1$ a $y_2$ é modelada em termos das diferenças nas probabilidades logarítmicas ajustadas pela política parametrizada $\pi_\theta$.

## 7. Objetivo de Máxima Verossimilhança

Para aprender os parâmetros $\theta$, definimos um objetivo de máxima verossimilhança baseado nas preferências observadas.

**Função Objetivo:**

Queremos maximizar a probabilidade das preferências observadas no conjunto de dados $D$:

$$
\max_{\theta} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log p_\theta(y_w \succ y_l | x) \right]
$$

Onde:

- $(x, y_w, y_l)$ são exemplos no conjunto de dados onde $y_w$ é a resposta preferida (ganhadora) e $y_l$ é a não preferida (perdedora).

## 8. Perda de Entropia Cruzada Binária

A maximização da log-verossimilhança é equivalente à minimização da perda de entropia cruzada binária.

**Perda para um Único Exemplo:**

$$
\mathcal{L}_{\text{BCE}}(\theta) = -\log p_\theta(y_w \succ y_l | x)
$$

Substituindo a expressão de $p_\theta$:

$$
\mathcal{L}_{\text{BCE}}(\theta) = -\log \sigma\left( \beta \left( \Delta_\theta(y_w, x) - \Delta_\theta(y_l, x) \right) \right)
$$

**Perda Total:**

A perda total é a expectativa sobre o conjunto de dados:

$$
\mathcal{L}_{\text{Total}}(\theta) = \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \mathcal{L}_{\text{BCE}}(\theta) \right]
$$

## 9. Função Objetivo Final DPO

Consolidando as expressões anteriores, a função objetivo final para o DPO é:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( \beta \left( \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right) \right]
$$

**Objetivo de Minimização:**

O objetivo é minimizar $\mathcal{L}_{\text{DPO}}(\theta)$ em relação a $\theta$.

**Resumo dos Componentes:**

- **$\pi_\theta(y | x)$:** Política parametrizada que estamos otimizando.
- **$\pi_{\text{ref}}(y | x)$:** Política de referência conhecida (por exemplo, o modelo antes do ajuste fino).
- **$\beta$:** Parâmetro de temperatura que controla a escala da diferença.
- **$\sigma(z)$:** Função sigmoide que mapeia a diferença para uma probabilidade entre 0 e 1.
- **$(x, y_w, y_l) \sim D$:** Amostras do conjunto de dados de preferências, onde $y_w$ é preferido a $y_l$ no contexto $x$.

**Interpretação Intuitiva:**

- O modelo ajusta $\pi_\theta$ para que a probabilidade da resposta preferida $y_w$ seja maior em comparação com a não preferida $y_l$, considerando também a política de referência $\pi_{\text{ref}}$.
- O termo $\log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}$ captura o quanto a nova política está desviando da política de referência em favor de respostas preferidas.

**Procedimento de Otimização:**

1. **Inicialização:** Começar com uma política inicial $\pi_\theta$, possivelmente igual a $\pi_{\text{ref}}$.
2. **Cálculo da Perda:** Para cada par de preferência no conjunto de dados, calcular $\mathcal{L}_{\text{BCE}}(\theta)$.
3. **Atualização dos Parâmetros:** Usar algoritmos de otimização baseados em gradiente (por exemplo, Adam) para atualizar $\theta$ minimizando $\mathcal{L}_{\text{DPO}}(\theta)$.
4. **Iteração:** Repetir o processo até a convergência ou até que um critério de parada seja satisfeito.

**Vantagens do DPO:**

- **Eficiência Computacional:** Evita a necessidade de estimar uma função de recompensa explícita.
- **Simplicidade Conceitual:** Transforma o problema em uma tarefa de classificação binária padrão.
- **Estabilidade:** A ancoragem na política de referência previne grandes desvios indesejados.

**Considerações Finais:**

- **Escolha de $\beta$:** Deve ser ajustado com cuidado; valores muito altos podem levar a instabilidades.
- **Qualidade dos Dados:** A eficácia do DPO depende da qualidade e representatividade das preferências no conjunto de dados.
- **Regularização:** Pode ser necessário adicionar termos de regularização para evitar sobreajuste.
