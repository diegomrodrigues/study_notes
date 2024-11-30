# Exemplo Numérico do Teorema 1

O **Teorema 1** afirma que, sob condições suaves, todas as classes de recompensa consistentes com os modelos Plackett-Luce (e Bradley-Terry em particular) podem ser representadas com a reparametrização:

$$
r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
$$
para algum modelo $\pi(y|x)$ e um dado modelo de referência $\pi_{\text{ref}}(y|x)$.

Vamos ilustrar este teorema com um exemplo numérico passo a passo.

---

## Passo 1: Definir o Prompt $x$ e as Respostas Possíveis $y$

Considere o seguinte prompt:

- $x$: "Resolva 2 + 2"

Respostas possíveis:

- $y_1$: "A resposta é 4."
- $y_2$: "Não sei."

## Passo 2: Definir a Política de Referência $\pi_{\text{ref}}(y|x)$

Suponha que a política de referência atribui probabilidades iguais a ambas as respostas:

$$
\pi_{\text{ref}}(y_1|x) = 0{,}5
$$
$$
\pi_{\text{ref}}(y_2|x) = 0{,}5
$$

## Passo 3: Definir a Política Alvo $\pi(y|x)$

Assuma que a política alvo favorece a resposta correta:

$$
\pi(y_1|x) = 0{,}8
$$
$$
\pi(y_2|x) = 0{,}2
$$

## Passo 4: Escolher um Valor para $\beta$

Vamos definir:

$$
\beta = 1
$$

## Passo 5: Calcular a Função de Recompensa $r(x, y)$

Utilizando a reparametrização do teorema:

$$
r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
$$

Para $y_1$:

$$
\begin{align*}
r(x, y_1) &= 1 \times \log \left( \frac{0{,}8}{0{,}5} \right) \\
&= \log(1{,}6) \\
&\approx 0{,}470
\end{align*}
$$

Para $y_2$:

$$
\begin{align*}
r(x, y_2) &= 1 \times \log \left( \frac{0{,}2}{0{,}5} \right) \\
&= \log(0{,}4) \\
&\approx -0{,}916
\end{align*}
$$

## Passo 6: Calcular a Função de Partição $Z(x)$

A função de partição é definida como:

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

Como $\beta = 1$:

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \exp\left( r(x, y) \right)
$$

Calculando $\exp\left( r(x, y) \right)$ para cada $y$:

Para $y_1$:

$$
\exp\left( r(x, y_1) \right) = \exp(0{,}470) \approx 1{,}6
$$

Para $y_2$:

$$
\exp\left( r(x, y_2) \right) = \exp(-0{,}916) \approx 0{,}4
$$

Agora, calculamos $Z(x)$:

$$
\begin{align*}
Z(x) &= \pi_{\text{ref}}(y_1|x) \times \exp\left( r(x, y_1) \right) + \pi_{\text{ref}}(y_2|x) \times \exp\left( r(x, y_2) \right) \\
&= 0{,}5 \times 1{,}6 + 0{,}5 \times 0{,}4 \\
&= 0{,}8 + 0{,}2 \\
&= 1{,}0
\end{align*}
$$

## Passo 7: Calcular a Política Ótima $\pi^*(y|x)$

De acordo com o teorema, a política ótima é dada por:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

Como $Z(x) = 1$ e $\beta = 1$:

$$
\pi^*(y|x) = \pi_{\text{ref}}(y|x) \exp\left( r(x, y) \right)
$$

Calculando para $y_1$:

$$
\begin{align*}
\pi^*(y_1|x) &= 0{,}5 \times 1{,}6 \\
&= 0{,}8
\end{align*}
$$

Para $y_2$:

$$
\begin{align*}
\pi^*(y_2|x) &= 0{,}5 \times 0{,}4 \\
&= 0{,}2
\end{align*}
$$

## Conclusão

Observamos que a política ótima $\pi^*(y|x)$ coincide com a política alvo $\pi(y|x)$ definida no **Passo 3**:

$$
\pi^*(y_1|x) = \pi(y_1|x) = 0{,}8
$$
$$
\pi^*(y_2|x) = \pi(y_2|x) = 0{,}2
$$

Este exemplo numérico demonstra como a reparametrização $r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ permite representar a função de recompensa de tal forma que a otimização direta da política correspondente resulta na política alvo desejada. Assim, ilustramos o **Teorema 1** de forma prática e numérica.

---

Este exemplo confirma que a reparametrização proposta no teorema é válida e eficaz para representar classes de recompensa consistentes com os modelos Plackett-Luce e Bradley-Terry.