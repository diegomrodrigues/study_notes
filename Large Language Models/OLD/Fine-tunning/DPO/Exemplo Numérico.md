# Exemplo Numérico do Direct Preference Optimization (DPO)

Vamos ilustrar o processo do DPO com um exemplo numérico simples. Este exemplo demonstrará como calcular a função objetivo do DPO para um único par de preferências.

## Contexto e Configuração

- **Contexto ($x$):** Suponha que o contexto seja uma pergunta simples: "Qual é a capital da França?"
- **Respostas:**
  - **Resposta preferida ($y_w$):** "A capital da França é Paris."
  - **Resposta não preferida ($y_l$):** "A capital da França é Berlim."

## Políticas e Probabilidades

### Política de Referência ($\pi_{\text{ref}}$)

A política de referência representa o modelo antes do ajuste fino. Suponha que as probabilidades condicionais dadas pela política de referência sejam:

- $\pi_{\text{ref}}(y_w | x) = 0.6$
- $\pi_{\text{ref}}(y_l | x) = 0.4$

### Política Parametrizada ($\pi_\theta$)

A política parametrizada é o modelo que estamos treinando para alinhar com as preferências. Suponha que as probabilidades condicionais atuais sejam:

- $\pi_\theta(y_w | x) = 0.5$
- $\pi_\theta(y_l | x) = 0.5$

### Parâmetro de Temperatura ($\beta$)

Escolhemos um valor para $\beta$ (por exemplo, $\beta = 1$):

$$
\beta = 1
$$

## Cálculo da Função Objetivo do DPO

Vamos calcular a perda $\mathcal{L}_{\text{DPO}}(\theta)$ para este exemplo.

### Passo 1: Calcular os Log-Razões das Probabilidades

Primeiro, calculamos os log-ratios para cada resposta:

#### Para a Resposta Preferida ($y_w$):

$$
\Delta_\theta(y_w, x) = \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} = \log \frac{0.5}{0.6}
$$

Calculando o valor numérico:

$$
\Delta_\theta(y_w, x) = \log \left( \frac{0.5}{0.6} \right) = \log(0.8333) \approx -0.182
$$

#### Para a Resposta Não Preferida ($y_l$):

$$
\Delta_\theta(y_l, x) = \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} = \log \frac{0.5}{0.4}
$$

Calculando o valor numérico:

$$
\Delta_\theta(y_l, x) = \log \left( \frac{0.5}{0.4} \right) = \log(1.25) \approx 0.223
$$

### Passo 2: Calcular a Diferença dos Log-Razões

Calculamos a diferença $\Delta_\theta(y_w, x) - \Delta_\theta(y_l, x)$:

$$
\Delta = \Delta_\theta(y_w, x) - \Delta_\theta(y_l, x) = (-0.182) - 0.223 = -0.405
$$

### Passo 3: Multiplicar pelo Parâmetro de Temperatura ($\beta$)

Como $\beta = 1$, a multiplicação não altera o valor:

$$
\beta \cdot \Delta = 1 \times (-0.405) = -0.405
$$

### Passo 4: Calcular a Função Sigmoide

Aplicamos a função sigmoide ao valor obtido:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Então:

$$
p_\theta(y_w \succ y_l | x) = \sigma(-0.405) = \frac{1}{1 + e^{0.405}}
$$

Calculando o valor numérico:

$$
e^{0.405} \approx 1.5 \\
p_\theta(y_w \succ y_l | x) = \frac{1}{1 + 1.5} = \frac{1}{2.5} = 0.4
$$

### Passo 5: Calcular a Perda de Entropia Cruzada Binária

Finalmente, calculamos a perda:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log p_\theta(y_w \succ y_l | x) = -\log(0.4)
$$

Calculando o valor numérico:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log(0.4) \approx -(-0.916) = 0.916
$$

## Interpretação dos Resultados

- **Probabilidade de Preferência:** ==O modelo atual atribui uma probabilidade de 40% de que a resposta preferida ($y_w$) seja melhor que a não preferida ($y_l$)==, o que está abaixo dos 50%. Isso indica que o modelo não está alinhado com as preferências observadas.
- **Perda Alta:** ==A perda $\mathcal{L}_{\text{DPO}}(\theta) \approx 0.916$ é relativamente alta, refletindo o baixo desempenho do modelo em prever a preferência correta.==

## Passo Adicional: Atualizar a Política Parametrizada

Para melhorar o modelo, devemos ajustar os parâmetros $\theta$ para aumentar $\pi_\theta(y_w | x)$ e/ou diminuir $\pi_\theta(y_l | x)$.

### Suponha que Após Atualização:

Após algumas iterações de treinamento, as probabilidades atualizadas são:

- $\pi_\theta(y_w | x) = 0.7$
- $\pi_\theta(y_l | x) = 0.3$

Repetimos os cálculos com os novos valores.

### Recalculando os Log-Razões

#### Para $y_w$:

$$
\Delta_\theta(y_w, x) = \log \frac{0.7}{0.6} = \log(1.1667) \approx 0.154
$$

#### Para $y_l$:

$$
\Delta_\theta(y_l, x) = \log \frac{0.3}{0.4} = \log(0.75) \approx -0.288
$$

### Diferença dos Log-Razões:

$$
\Delta = 0.154 - (-0.288) = 0.154 + 0.288 = 0.442
$$

### Multiplicação pelo Parâmetro $\beta$:

$$
\beta \cdot \Delta = 1 \times 0.442 = 0.442
$$

### Calcular a Função Sigmoide:

$$
p_\theta(y_w \succ y_l | x) = \sigma(0.442) = \frac{1}{1 + e^{-0.442}}
$$

Calculando o valor numérico:

$$
e^{-0.442} \approx 0.643 \\
p_\theta(y_w \succ y_l | x) = \frac{1}{1 + 0.643} = \frac{1}{1.643} \approx 0.608
$$

### Calcular a Perda:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log(0.608) \approx -(-0.498) = 0.498
$$

## Nova Interpretação

- **Probabilidade de Preferência Aumentada:** Agora, o modelo atribui uma probabilidade de aproximadamente 60.8% de que $y_w$ seja preferido a $y_l$, mostrando uma melhoria.
- **Perda Reduzida:** ==A perda diminuiu para $\mathcal{L}_{\text{DPO}}(\theta) \approx 0.498$, indicando que o modelo está melhor alinhado com as preferências.==

## Conclusão do Exemplo

Este exemplo numérico demonstra como o DPO ajusta a política $\pi_\theta$ para melhor refletir as preferências observadas:

- **Inicialmente**, o modelo não refletia corretamente a preferência, resultando em uma perda alta.
- **Após o treinamento**, as probabilidades foram ajustadas, aumentando a probabilidade atribuída à resposta preferida e reduzindo a perda.

## Observações Finais

- **Importância de $\beta$:** Neste exemplo, $\beta = 1$. Alterar $\beta$ afetaria a escala da diferença e, consequentemente, a probabilidade final.
- **Ancoragem na Política de Referência:** A comparação com $\pi_{\text{ref}}$ garante que as atualizações da política não se desviem drasticamente do comportamento original.
- **Generalização:** Embora este exemplo seja simplificado, o mesmo processo se aplica a modelos de linguagem maiores e conjuntos de dados de preferências reais.

## Resumo das Equações Utilizadas

1. **Log-Razão da Resposta:**

   $$
   \Delta_\theta(y, x) = \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}
   $$

2. **Diferença das Log-Razões:**

   $$
   \Delta = \Delta_\theta(y_w, x) - \Delta_\theta(y_l, x)
   $$

3. **Probabilidade de Preferência:**

   $$
   p_\theta(y_w \succ y_l | x) = \sigma\left( \beta \cdot \Delta \right)
   $$

4. **Perda de Entropia Cruzada Binária:**

   $$
   \mathcal{L}_{\text{DPO}}(\theta) = -\log p_\theta(y_w \succ y_l | x)
   $$

## Referências

- Função Sigmoide:

  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$

- Função Logarítmica Natural:

  $$
  \log_e (x) = \ln(x)
  $$

Este exemplo numérico ilustra como o DPO pode ser aplicado na prática para ajustar um modelo de linguagem de acordo com preferências específicas, usando cálculos explícitos e passo a passo.