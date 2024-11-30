Vou criar um exemplo mais complexo de um sistema de terceira ordem com interações mais ricas.

# Sistema Dinâmico de Terceira Ordem com Feedbacks Não-Lineares

## 1. Especificação do Sistema

Considere o seguinte sistema de terceira ordem:

$$ y_t = 0.7y_{t-1} + 0.4y_{t-2} - 0.2y_{t-3} + w_t $$

onde:
- $\phi_1 = 0.7$ (efeito positivo do primeiro lag)
- $\phi_2 = 0.4$ (efeito positivo do segundo lag)
- $\phi_3 = -0.2$ (efeito negativo do terceiro lag)

## 2. Representação em Forma de Estado

O sistema pode ser escrito na forma matricial:

$$ \begin{bmatrix} 
y_t \\
y_{t-1} \\
y_{t-2}
\end{bmatrix} = 
\begin{bmatrix}
0.7 & 0.4 & -0.2 \\
1.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0
\end{bmatrix}
\begin{bmatrix}
y_{t-1} \\
y_{t-2} \\
y_{t-3}
\end{bmatrix} +
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}w_t $$

## 3. Análise dos Autovalores

Os autovalores $\lambda$ da matriz de transição satisfazem a equação característica:

$$ \lambda^3 - 0.7\lambda^2 - 0.4\lambda + 0.2 = 0 $$

Resolvendo numericamente, obtemos:
- $\lambda_1 \approx 0.893$ (autovalor dominante real)
- $\lambda_2 \approx -0.224 + 0.316i$ (autovalor complexo)
- $\lambda_3 \approx -0.224 - 0.316i$ (autovalor complexo conjugado)

## 4. Dinâmica do Sistema

### 4.1 Multiplicadores Dinâmicos

Para horizonte $j$, o multiplicador dinâmico é:

$$ \frac{\partial y_{t+j}}{\partial w_t} = c_1\lambda_1^j + c_2\lambda_2^j + c_3\lambda_3^j $$

onde $c_i$ são constantes determinadas pelos autovetores.

### 4.2 Padrões de Choque

O sistema foi submetido a uma sequência de choques:
- $w_3 = 0.8$ (choque forte positivo)
- $w_7 = -0.5$ (choque médio negativo)
- $w_{11} = 0.3$ (choque fraco positivo)
- $w_{15} = -0.2$ (choque fraco negativo)

## 5. Características do Sistema

1. **Comportamento Oscilatório**:
   - Os autovalores complexos conjugados induzem oscilações
   - Período ≈ $\frac{2\pi}{\text{arg}(\lambda_2)} \approx 4.5$ períodos

2. **Convergência**:
   - $|\lambda_1| < 1$ garante estabilidade
   - Taxa de convergência dominada por $\lambda_1^t$

3. **Interação de Lags**:
   - Efeito positivo inicial ($\phi_1 > 0$)
   - Reforço intermediário ($\phi_2 > 0$)
   - Reversão tardia ($\phi_3 < 0$)

## 6. Análise das Trajetórias

### Trajetória com Choques (linha azul):
- Mostra a resposta completa do sistema aos choques
- Exibe oscilações complexas devido à interação entre:
  * Dinâmica natural (autovalores)
  * Choques externos ($w_t$)

### Trajetória sem Choques (linha verde tracejada):
- Revela a dinâmica intrínseca do sistema
- Converge gradualmente para zero
- Mantém padrão oscilatório devido aos autovalores complexos

### Trajetória Teórica (linha laranja pontilhada):
- Baseada na decomposição espectral
- Representa o comportamento médio esperado
- Suaviza as oscilações de alta frequência

## 7. Conclusões

1. **Complexidade Dinâmica**:
   - Sistema exibe tanto comportamento oscilatório quanto convergente
   - Interação rica entre diferentes horizontes temporais

2. **Estabilidade**:
   - Sistema é estável mas com ajuste não-monotônico
   - Choques têm efeitos persistentes mas eventualmente se dissipam

3. **Previsibilidade**:
   - Maior incerteza em horizontes intermediários
   - Convergência previsível no longo prazo

Este exemplo ilustra como sistemas de ordem superior podem gerar dinâmicas mais ricas e complexas, combinando múltiplos canais de transmissão temporal com diferentes padrões de resposta a choques.

Vou explorar o conceito de multiplicadores dinâmicos usando um exemplo detalhado e mostrar como eles se relacionam com a estrutura do sistema.

# Multiplicadores Dinâmicos em Sistemas de Ordem Superior

## 1. Definição Formal

Para um sistema de ordem $p$, o multiplicador dinâmico mede o impacto de uma perturbação no tempo $t$ sobre o valor da variável no tempo $t+j$:

$$ \frac{\partial y_{t+j}}{\partial w_t} = f_{11}^{(j)} $$

onde $f_{11}^{(j)}$ é o elemento $(1,1)$ da matriz $F^j$.

## 2. Exemplo Analítico

Considere um sistema de segunda ordem:

$$ y_t = 0.6y_{t-1} + 0.3y_{t-2} + w_t $$

A matriz de transição $F$ é:

$$ F = \begin{bmatrix} 
0.6 & 0.3 \\
1.0 & 0.0
\end{bmatrix} $$

## 3. Cálculo dos Multiplicadores

### Horizonte j = 1:
$$ F^1 = \begin{bmatrix} 
0.6 & 0.3 \\
1.0 & 0.0
\end{bmatrix} $$

Portanto: $\frac{\partial y_{t+1}}{\partial w_t} = 0.6$

### Horizonte j = 2:
$$ F^2 = \begin{bmatrix} 
0.6 & 0.3 \\
1.0 & 0.0
\end{bmatrix} \begin{bmatrix} 
0.6 & 0.3 \\
1.0 & 0.0
\end{bmatrix} = \begin{bmatrix} 
0.36 + 0.3 & 0.18 \\
0.6 & 0.3
\end{bmatrix} $$

Portanto: $\frac{\partial y_{t+2}}{\partial w_t} = 0.66$

## 4. Decomposição do Efeito

Para $j = 2$, o multiplicador 0.66 pode ser decomposto em:
- Efeito direto: $(0.6)^2 = 0.36$
- Efeito indireto: $0.3 = 0.3$

## 5. Interpretação Geométrica

A propagação do choque segue um padrão determinado pelos autovalores de $F$. Para este sistema:

$$ \text{det}(F - \lambda I) = \begin{vmatrix} 
0.6 - \lambda & 0.3 \\
1.0 & -\lambda
\end{vmatrix} = \lambda^2 - 0.6\lambda - 0.3 = 0 $$

Os autovalores são:
$$ \lambda_{1,2} = \frac{0.6 \pm \sqrt{0.36 + 1.2}}{2} $$

Isto resulta em:
- $\lambda_1 \approx 0.927$ (autovalor dominante)
- $\lambda_2 \approx -0.327$ (autovalor secundário)

## 6. Padrão de Resposta

Para qualquer horizonte $j$, o multiplicador dinâmico pode ser expresso como:

$$ f_{11}^{(j)} = c_1\lambda_1^j + c_2\lambda_2^j $$

onde $c_1$ e $c_2$ são constantes determinadas pelos autovetores.

## 7. Comportamento Assintótico

Como $|\lambda_1| < 1$ e $|\lambda_2| < 1$:

$$ \lim_{j \to \infty} \frac{\partial y_{t+j}}{\partial w_t} = \lim_{j \to \infty} f_{11}^{(j)} = 0 $$

Este resultado demonstra que o sistema é estável e que os choques eventualmente se dissipam.

## 8. Exemplo Numérico

Para os primeiros períodos:

| Horizonte (j) | Multiplicador ($f_{11}^{(j)}$) | Decomposição        |
| ------------- | ------------------------------ | ------------------- |
| 0             | 1.000                          | Impacto direto      |
| 1             | 0.600                          | $\phi_1$            |
| 2             | 0.660                          | $\phi_1^2 + \phi_2$ |
| 3             | 0.636                          | Combinação complexa |
| 4             | 0.592                          | Decaimento gradual  |

## 9. Propriedades Fundamentais

1. **Monotonicidade**: O sistema não é monotônico devido ao autovalor negativo

2. **Persistência**: O decaimento é mais lento que exponencial devido à interação dos lags

3. **Oscilação**: O sistema exibe pequenas oscilações devido ao autovalor negativo

## 10. Conclusão

Os multiplicadores dinâmicos fornecem uma maneira sistemática de entender como choques se propagam através do sistema. A decomposição espectral permite:

1. Identificar os canais de transmissão
2. Quantificar a persistência dos choques
3. Prever o padrão de resposta em diferentes horizontes

A estrutura dos multiplicadores é determinada por:
- Os coeficientes originais do sistema ($\phi_1$ e $\phi_2$)
- Os autovalores da matriz de transição
- A interação entre diferentes horizontes temporais

Isto fornece uma caracterização completa da dinâmica do sistema e sua resposta a perturbações.



Vou criar um exemplo numérico detalhado para ilustrar como os choques se propagam em um sistema dinâmico.

# Exemplo: Propagação de Choques em um Sistema de Segunda Ordem

## 1. Definição do Sistema

Considere o sistema:

$$ y_t = 0.8y_{t-1} + 0.3y_{t-2} + w_t $$

A matriz de transição $F$ é:

$$ F = \begin{bmatrix}
0.8 & 0.3 \\
1.0 & 0.0
\end{bmatrix} $$

## 2. Cálculo dos Autovalores

A equação característica é:
$$ \det(F - \lambda I) = \begin{vmatrix}
0.8 - \lambda & 0.3 \\
1.0 & -\lambda
\end{vmatrix} = \lambda^2 - 0.8\lambda - 0.3 = 0 $$

Resolvendo:
$$ \lambda = \frac{0.8 \pm \sqrt{0.64 + 1.2}}{2} = \frac{0.8 \pm \sqrt{1.84}}{2} $$

Obtemos:
- $\lambda_1 \approx 1.077$ (autovalor dominante)
- $\lambda_2 \approx -0.277$ (autovalor secundário)

## 3. Cálculo dos Autovetores

Para $\lambda_1 = 1.077$:
$$ (F - \lambda_1 I)v_1 = 0 $$
$$ \begin{bmatrix}
-0.277 & 0.3 \\
1.0 & -1.077
\end{bmatrix} \begin{bmatrix}
v_{11} \\
v_{12}
\end{bmatrix} = \begin{bmatrix}
0 \\
0
\end{bmatrix} $$

Obtemos $v_1 = \begin{bmatrix} 1.077 \\ 1 \end{bmatrix}$

Similarmente para $\lambda_2 = -0.277$:
$v_2 = \begin{bmatrix} -0.277 \\ 1 \end{bmatrix}$

## 4. Decomposição Espectral

A matriz $T$ de autovetores é:
$$ T = \begin{bmatrix}
1.077 & -0.277 \\
1.000 & 1.000
\end{bmatrix} $$

$T^{-1}$ é:
$$ T^{-1} = \frac{1}{1.354} \begin{bmatrix}
1.000 & 0.277 \\
-1.000 & 1.077
\end{bmatrix} $$

## 5. Cálculo dos Coeficientes $c_i$

$$ c_1 = T_{11}(T^{-1})_{11} = 1.077 \times \frac{1.000}{1.354} \approx 0.795 $$
$$ c_2 = T_{12}(T^{-1})_{21} = -0.277 \times \frac{-1.000}{1.354} \approx 0.205 $$

## 6. Multiplicadores Dinâmicos

Para qualquer horizonte $j$:
$$ f_{11}^{(j)} = 0.795(1.077)^j + 0.205(-0.277)^j $$

Vamos calcular os primeiros períodos:

| j    | $0.795(1.077)^j$ | $0.205(-0.277)^j$ | $f_{11}^{(j)}$ |
| ---- | ---------------- | ----------------- | -------------- |
| 0    | 0.795            | 0.205             | 1.000          |
| 1    | 0.857            | -0.057            | 0.800          |
| 2    | 0.923            | 0.016             | 0.939          |
| 3    | 0.994            | -0.004            | 0.990          |
| 4    | 1.071            | 0.001             | 1.072          |

## 7. Análise da Propagação do Choque

1. **Impacto Imediato ($j=0$)**:
   - Multiplicador = 1.000
   - Choque afeta $y_t$ um-para-um

2. **Primeiro Período ($j=1$)**:
   - Multiplicador = 0.800
   - Corresponde ao coeficiente $\phi_1$

3. **Segundo Período ($j=2$)**:
   - Multiplicador = 0.939
   - Combinação de $\phi_1^2 + \phi_2$
   - Aumento devido ao efeito do segundo lag

4. **Períodos Posteriores**:
   - Sistema diverge lentamente
   - $|\lambda_1| > 1$ implica instabilidade
   - Oscilações amortecidas devido a $\lambda_2 < 0$

## 8. Padrões de Resposta

1. **Componente Explosiva**:
   - $\lambda_1 = 1.077$ > 1
   - Causa crescimento exponencial
   - Domina no longo prazo

2. **Componente Oscilatória**:
   - $\lambda_2 = -0.277$
   - Causa oscilações
   - Decai rapidamente

## 9. Interpretação Econômica

1. **Curto Prazo**:
   - Resposta inicial forte (multiplicador = 1)
   - Seguida por declínio temporário

2. **Médio Prazo**:
   - Oscilações devido ao autovalor negativo
   - Interação complexa entre lags

3. **Longo Prazo**:
   - Sistema é instável
   - Choques são amplificados
   - Divergência exponencial

## 10. Conclusões

Este exemplo ilustra como:
1. A estrutura dos autovalores determina a dinâmica
2. Autovalores mistos (>1 e <1) geram comportamento complexo
3. A decomposição espectral permite análise detalhada
4. Sistemas instáveis podem ter dinâmica rica no curto prazo

As implicações práticas incluem:
- Necessidade de estabilização se $|\lambda_1| > 1$
- Importância do monitoramento de curto prazo
- Possibilidade de ciclos no médio prazo



Vou criar um exemplo numérico detalhado para ilustrar a estrutura das potências da matriz F e como isso afeta os multiplicadores dinâmicos.

# Exemplo Numérico: Potências da Matriz F e Multiplicadores Dinâmicos

## 1. Sistema Base

Considere um sistema de ordem 3:

$$ y_t = 0.7y_{t-1} + 0.4y_{t-2} - 0.2y_{t-3} + w_t $$

A matriz F é:
$$ F = \begin{bmatrix}
0.7 & 0.4 & -0.2 \\
1.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0
\end{bmatrix} $$

## 2. Cálculo de F²

Vamos calcular F² = F × F passo a passo:

1) **Primeira linha de F²**:
   ```
   [0.7  0.4  -0.2] × [0.7   0.4   -0.2]
   [1.0  0.0   0.0]   [1.0   0.0    0.0]
   [0.0  1.0   0.0]   [0.0   1.0    0.0]
   ```

   $f_{11}^{(2)} = (0.7)(0.7) + (0.4)(1.0) + (-0.2)(0.0) = 0.49 + 0.4 = 0.89$
   
   $f_{12}^{(2)} = (0.7)(0.4) + (0.4)(0.0) + (-0.2)(1.0) = 0.28 - 0.2 = 0.08$
   
   $f_{13}^{(2)} = (0.7)(-0.2) + (0.4)(0.0) + (-0.2)(0.0) = -0.14$

2) **Segunda linha de F²**:
   $f_{21}^{(2)} = 0.7$
   $f_{22}^{(2)} = 0.4$
   $f_{23}^{(2)} = -0.2$

3) **Terceira linha de F²**:
   $f_{31}^{(2)} = 1.0$
   $f_{32}^{(2)} = 0.0$
   $f_{33}^{(2)} = 0.0$

Portanto:
$$ F^2 = \begin{bmatrix}
0.89 & 0.08 & -0.14 \\
0.70 & 0.40 & -0.20 \\
1.00 & 0.00 & 0.00
\end{bmatrix} $$

## 3. Verificação do Teorema 4

Para o elemento $(1,1)$:
- $\phi_1^2 = (0.7)^2 = 0.49$
- $\phi_2 = 0.4$
- $f_{11}^{(2)} = \phi_1^2 + \phi_2 = 0.49 + 0.4 = 0.89$

## 4. Cálculo de F³

Calculando F³ = F² × F:

$$ F^3 = \begin{bmatrix}
0.863 & 0.276 & -0.098 \\
0.890 & 0.080 & -0.140 \\
0.700 & 0.400 & -0.200
\end{bmatrix} $$

## 5. Análise dos Multiplicadores Dinâmicos

| Horizonte (j) | $f_{11}^{(j)}$ | Decomposição        |
| ------------- | -------------- | ------------------- |
| 1             | 0.700          | $\phi_1$            |
| 2             | 0.890          | $\phi_1^2 + \phi_2$ |
| 3             | 0.863          | Combinação complexa |

## 6. Interpretação Econômica

### 6.1 Primeiro Período (j=1)
- Impacto direto: 0.7
- Representa efeito imediato via $\phi_1$

### 6.2 Segundo Período (j=2)
- Efeito direto: $(0.7)^2 = 0.49$
- Efeito indireto: $0.4$
- Total: $0.89$

### 6.3 Terceiro Período (j=3)
- Combinação de efeitos diretos e indiretos
- Inclui feedback do termo negativo $\phi_3$

## 7. Estrutura de Propagação

1. **Canais de Transmissão**:
   - Via primeiro lag: $\phi_1^j$
   - Via segundo lag: $\phi_2$ (com defasagem)
   - Via terceiro lag: $-\phi_3$ (com defasagem maior)

2. **Padrão de Oscilação**:
   ```
   j=1: 0.700 (↑)
   j=2: 0.890 (↑)
   j=3: 0.863 (↓)
   ```

## 8. Decomposição dos Efeitos

Para j=2, podemos decompor o multiplicador:

1. **Efeito Direto**: 
   $\phi_1^2 = 0.49$ (via primeiro lag ao quadrado)

2. **Efeito Indireto**: 
   $\phi_2 = 0.40$ (via segundo lag)

3. **Efeito Total**:
   $f_{11}^{(2)} = 0.89$

## 9. Implicações Práticas

1. **Previsão**:
   - Horizonte curto: dominado por $\phi_1$
   - Horizonte médio: interação complexa
   - Horizonte longo: convergência cíclica

2. **Política**:
   - Efeitos de intervenções são não-lineares
   - Timing é crucial devido aos lags
   - Feedback negativo pode causar reversões

## 10. Conclusões

1. **Estrutura Temporal**:
   - Multiplicadores seguem padrão complexo
   - Combinação de efeitos diretos e indiretos
   - Não-linearidades emergem após j=2

2. **Implicações Analíticas**:
   - F² captura interações de segunda ordem
   - Teorema 4 simplifica análise
   - Estrutura permite decomposição clara

3. **Aplicações**:
   - Previsão de impactos
   - Calibração de políticas
   - Análise de estabilidade

Este exemplo ilustra como a estrutura das potências de F determina a propagação de choques e como o Teorema 4 fornece uma ferramenta analítica poderosa para entender essa dinâmica.