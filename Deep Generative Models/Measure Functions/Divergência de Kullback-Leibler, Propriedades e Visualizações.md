## Divergência de Kullback-Leibler: Propriedades e Visualizações

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820084939379.png" alt="image-20240820084939379" style="zoom:80%;" />

### Introdução

A divergência de Kullback-Leibler (KL) é uma medida fundamental na teoria da informação e estatística para quantificar a diferença entre duas distribuições de probabilidade [1]. Embora não seja uma métrica no sentido matemático estrito, ela fornece insights valiosos sobre a relação entre distribuições e tem aplicações amplas em aprendizado de máquina e inferência estatística.

### Definição Matemática

Para distribuições discretas P e Q, a divergência KL é definida como:

$$
D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

Para distribuições contínuas, usamos a integral:

$$
D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx
$$

onde p(x) e q(x) são as densidades de probabilidade de P e Q, respectivamente [1].

### Propriedades Fundamentais

Vamos explorar as propriedades principais da divergência KL, demonstrando cada uma matematicamente e propondo visualizações didáticas.

#### 1. Não-negatividade

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820152008267.png" alt="image-20240820152008267" style="zoom:67%;" />

**Propriedade**: $D_{KL}(P||Q) \geq 0$ para todas as distribuições P e Q [2].

**Demonstração**:
Usando a desigualdade de Jensen, temos:
$$
\begin{aligned}
D_{KL}(P||Q) &= \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \\
&= -\sum_{x} P(x) \log \frac{Q(x)}{P(x)} \\
&\geq -\log \sum_{x} P(x) \frac{Q(x)}{P(x)} \quad \text{(pela desigualdade de Jensen)} \\
&= -\log \sum_{x} Q(x) \\
&= -\log 1 = 0
\end{aligned}
$$

#### 2. Igualdade

**Propriedade**: $D_{KL}(P||Q) = 0$ se e somente se P = Q quase em toda parte [2].

**Demonstração**:
A igualdade na desigualdade de Jensen ocorre se e somente se o argumento é constante. Isso significa:

$$
\frac{Q(x)}{P(x)} = c \quad \text{(constante)}
$$

Como ambas P e Q são distribuições de probabilidade, devem somar 1:

$$
\sum_{x} P(x) = \sum_{x} Q(x) = 1
$$

Isso implica que c = 1, e portanto, P(x) = Q(x) para todo x.

#### 3. Assimetria

**Propriedade**: Em geral, $D_{KL}(P||Q) \neq D_{KL}(Q||P)$ [7].

**Demonstração**:
Considere um exemplo simples:
P: [0.5, 0.5]
Q: [0.9, 0.1]

$$
\begin{aligned}
D_{KL}(P||Q) &= 0.5 \log \frac{0.5}{0.9} + 0.5 \log \frac{0.5}{0.1} \approx 0.510 \\
D_{KL}(Q||P) &= 0.9 \log \frac{0.9}{0.5} + 0.1 \log \frac{0.1}{0.5} \approx 0.247
\end{aligned}
$$

#### 4. Convexidade

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820152247384.png" alt="image-20240820152247384" style="zoom: 80%;" />

**Propriedade**: A divergência KL é convexa em ambos os argumentos.

**Demonstração**:
Para demonstrar a convexidade em P, considere P1, P2, e 0 ≤ λ ≤ 1:
$$
\begin{aligned}
&D_{KL}(\lambda P_1 + (1-\lambda)P_2 || Q) \\
&= \sum_{x} (\lambda P_1(x) + (1-\lambda)P_2(x)) \log \frac{\lambda P_1(x) + (1-\lambda)P_2(x)}{Q(x)} \\
&\leq \lambda \sum_{x} P_1(x) \log \frac{P_1(x)}{Q(x)} + (1-\lambda) \sum_{x} P_2(x) \log \frac{P_2(x)}{Q(x)} \\
&= \lambda D_{KL}(P_1||Q) + (1-\lambda) D_{KL}(P_2||Q)
\end{aligned}
$$

A desigualdade vem da concavidade do logaritmo.

#### 5. Invariância de Transformação

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820152529735.png" alt="image-20240820152529735" style="zoom: 80%;" />

**Propriedade**: Para uma transformação invertível T, $D_{KL}(P||Q) = D_{KL}(T(P)||T(Q))$

**Demonstração**:
Seja y = T(x). Então:

$$
\begin{aligned}
D_{KL}(T(P)||T(Q)) &= \int p_Y(y) \log \frac{p_Y(y)}{q_Y(y)} dy \\
&= \int p_X(T^{-1}(y)) \left|\frac{dT^{-1}(y)}{dy}\right| \log \frac{p_X(T^{-1}(y)) \left|\frac{dT^{-1}(y)}{dy}\right|}{q_X(T^{-1}(y)) \left|\frac{dT^{-1}(y)}{dy}\right|} dy \\
&= \int p_X(x) \log \frac{p_X(x)}{q_X(x)} dx \\
&= D_{KL}(P||Q)
\end{aligned}
$$

### Interpretação em Teoria da Informação

![image-20240820152729311](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820152729311.png)

A divergência KL tem uma interpretação profunda em teoria da informação como o número esperado de bits extras necessários para codificar amostras de P usando um código otimizado para Q [3].

**Demonstração**:
Seja L(x) o comprimento do código para x. O comprimento médio do código é:

$$
\begin{aligned}
E_{P}[L] &= \sum_{x} P(x)L(x) \\
&= \sum_{x} P(x)(-\log Q(x)) \\
&= -\sum_{x} P(x)\log Q(x) \\
&= -\sum_{x} P(x)\log P(x) + \sum_{x} P(x)\log \frac{P(x)}{Q(x)} \\
&= H(P) + D_{KL}(P||Q)
\end{aligned}
$$

Onde H(P) é a entropia de P.

### Aplicações em Aprendizado de Máquina

1. **Estimativa de Densidade**: Minimizar D(P||Q) é equivalente a maximizar a verossimilhança de Q dado amostras de P [4].

2. **Inferência Variacional**: Em métodos variacionais, a divergência KL é usada para aproximar distribuições posteriores complexas [4].

### Limitações e Considerações Práticas

1. **Sensibilidade a Eventos Raros**: D(P||Q) pode ser muito sensível a eventos com baixa probabilidade em Q mas alta em P [7].

2. **Indefinição**: D(P||Q) é indefinida se Q(x) = 0 e P(x) > 0 para algum x [7].

![image-20240820153027447](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820153027447.png)

#### Questões Técnicas/Teóricas

1. Como a divergência KL se comporta quando comparamos duas distribuições normais com médias diferentes mas variâncias iguais?

2. Em um contexto de aprendizado de máquina, como você interpretaria geometricamente a minimização da divergência KL entre a distribuição do modelo e a distribuição dos dados?

### Implementação em Python

Aqui está um exemplo de como calcular a divergência KL para distribuições normais em Python:

```python
import numpy as np
from scipy.stats import norm

def kl_divergence_norm(m1, s1, m2, s2):
    return np.log(s2/s1) + (s1**2 + (m1-m2)**2)/(2*s2**2) - 0.5

# Exemplo
m1, s1 = 0, 1  # Média e desvio padrão da primeira distribuição
m2, s2 = 1, 2  # Média e desvio padrão da segunda distribuição

print(f"KL(N(0,1) || N(1,2)) = {kl_divergence_norm(m1, s1, m2, s2):.4f}")
print(f"KL(N(1,2) || N(0,1)) = {kl_divergence_norm(m2, s2, m1, s1):.4f}")
```

Este código demonstra a assimetria da divergência KL para distribuições normais.

### Conclusão

A divergência de Kullback-Leibler é uma ferramenta poderosa na teoria da informação e aprendizado de máquina. Suas propriedades matemáticas, como não-negatividade, convexidade e invariância de transformação, a tornam útil em uma variedade de contextos. No entanto, sua assimetria e sensibilidade a eventos raros devem ser consideradas cuidadosamente em aplicações práticas.

A compreensão profunda da divergência KL e suas propriedades é essencial para cientistas de dados e pesquisadores em aprendizado de máquina, pois ela fundamenta muitos métodos de inferência e otimização.

### Questões Avançadas

1. Como a escolha entre minimizar D(P||Q) versus D(Q||P) afeta o comportamento de um algoritmo de aprendizado variacional? Dê exemplos concretos.

2. Derive a forma fechada da divergência KL entre duas distribuições gaussianas multivariadas. Como esta forma se relaciona com a distância de Mahalanobis?

3. Discuta as implicações da convexidade da divergência KL no contexto de otimização em aprendizado de máquina. Como isso afeta a escolha de algoritmos de otimização?

### Referências

[1] "The Kullback-Leibler divergence (KL-divergence) between two distributions p and q is defined as D(p∥q) = Sum_x p(x) log (p(x)/q(x))." (Trecho de cs236_lecture4.pdf)

[2] "D(p ∥ q) ≥ 0 for all p, q, with equality if and only if p = q. Proof: E_x~p[-log(q(x)/p(x))] ≥ -log(E_x~p[q(x)/p(x)]) = -log(Sum_x p(x) q(x)/p(x)) = 0" (Trecho de cs236_lecture4.pdf)

[3] "Measures the expected number of extra bits required to describe samples from p(x) using a compression code based on q instead of p" (Trecho de cs236_lecture4.pdf)

[4] "We want to learn the full distribution so that later we can answer any probabilistic inference query" (Trecho de cs236_lecture4.pdf)

[5] "In this setting we can view the learning problem as density estimation" (Trecho de cs236_lecture4.pdf)

[6] "How should we measure distance between distributions?" (Trecho de cs236_lecture4.pdf)

[7] "Notice that KL-divergence is asymmetric, i.e., D(p∥q) ̸= D(q∥p)" (Trecho de cs236_lecture4.pdf)

[8] "We want to construct P_θ as "close" as possible to P_data (recall we assume we are given a dataset D of samples from P_data)" (Trecho de cs236_lecture4.pdf)