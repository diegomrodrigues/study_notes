## Change of Variables Formula: Fundamento Matemático dos Normalizing Flows

![image-20240902094417386](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902094417386.png)

### Introdução

A **fórmula de mudança de variáveis** é um conceito fundamental na teoria da probabilidade e estatística, desempenhando um ==papel crucial no desenvolvimento e compreensão dos modelos de fluxo normalizador (normalizing flows) [1].== Esta fórmula fornece o arcabouço matemático necessário para ==transformar distribuições de probabilidade através de funções invertíveis, permitindo a construção de modelos generativos complexos a partir de distribuições simples [2].==

No contexto dos normalizing flows, a fórmula de mudança de variáveis é a pedra angular que permite a transformação de uma distribuição de base simples (como uma gaussiana) em distribuições de dados complexas e multidimensionais [3]. Este resumo explorará em profundidade a formulação matemática, derivação e aplicações da fórmula de mudança de variáveis, tanto no caso unidimensional quanto no caso geral multidimensional.

### Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Transformação Invertível**  | ==Uma função $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$ que possui uma inversa única $f^{-1}$, permitindo a transformação bidirecional entre espaços [4].== |
| **Jacobiano**                 | ==Matriz de derivadas parciais de primeira ordem de uma função vetorial==, crucial para o cálculo da mudança de densidade [5]. |
| **Determinante do Jacobiano** | ==Medida da mudança local de volume induzida pela transformação==, elemento chave na fórmula de mudança de variáveis [6]. |

> ⚠️ **Nota Importante**: A compreensão profunda da fórmula de mudança de variáveis é essencial para o desenvolvimento e implementação eficaz de modelos de fluxo normalizador.

### Fórmula de Mudança de Variáveis: Caso Unidimensional

![image-20240902095109964](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902095109964.png)

No caso unidimensional, consideramos uma variável aleatória $Z$ com densidade $p_Z(z)$ e uma transformação invertível $X = f(Z)$ com inversa $Z = f^{-1}(X) = h(X)$ [7]. A fórmula de mudança de variáveis para este caso é dada por:
$$
p_X(x) = p_Z(h(x)) \left|\frac{dh(x)}{dx}\right|
$$

Onde:
- $p_X(x)$ é a densidade da variável transformada $X$
- $p_Z(z)$ é a densidade da variável original $Z$
- $h(x) = f^{-1}(x)$ é a função inversa da transformação
- ==$\left|\frac{dh(x)}{dx}\right|$ é o valor absoluto da derivada de $h$ com respeito a $x$==

#### Derivação Informal

Para derivar informalmente esta fórmula, consideremos um pequeno intervalo $\Delta z$ em torno de $z$ e o correspondente intervalo $\Delta x$ em torno de $x = f(z)$ [8]:

1. A probabilidade contida em $\Delta z$ deve ser igual à probabilidade contida em $\Delta x$:
   
   $p_Z(z)\Delta z \approx p_X(x)\Delta x$

2. Pela definição de derivada, temos:
   
   $\Delta x \approx \frac{df(z)}{dz}\Delta z = \frac{1}{\frac{dh(x)}{dx}}\Delta z$

3. Substituindo e rearranjando:
   
   $p_X(x) \approx p_Z(z)\left|\frac{dh(x)}{dx}\right|$

4. No limite quando $\Delta z \rightarrow 0$, obtemos a fórmula exata.

> ✔️ **Ponto de Destaque**: ==O termo $\left|\frac{dh(x)}{dx}\right|$ captura a "distorção" local introduzida pela transformação na densidade de probabilidade.==

#### Questões Técnicas/Teóricas

1. Como a fórmula de mudança de variáveis unidimensional se relaciona com o conceito de conservação de probabilidade?
2. Dada uma distribuição uniforme $U(0,1)$ e a transformação $X = -\ln(1-Z)$, derive a densidade de $X$ usando a fórmula de mudança de variáveis.

### Fórmula de Mudança de Variáveis: Caso Geral

![image-20240902095607964](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902095607964.png)

No caso geral multidimensional, consideramos uma transformação invertível $X = f(Z)$ onde $Z$ e $X$ são vetores aleatórios em $\mathbb{R}^n$ [9]. A fórmula de mudança de variáveis neste caso é dada por:
$$
p_X(x) = p_Z(f^{-1}(x)) \left|\det\left(\frac{\partial f^{-1}(x)}{\partial x}\right)\right|
$$

Onde:
- $p_X(x)$ é a densidade do vetor transformado $X$
- $p_Z(z)$ é a densidade do vetor original $Z$
- $f^{-1}(x)$ é a função inversa da transformação
- ==$\frac{\partial f^{-1}(x)}{\partial x}$ é a matriz Jacobiana de $f^{-1}$==
- $\det(\cdot)$ denota o determinante da matriz

#### Derivação

A derivação da fórmula geral segue princípios similares ao caso unidimensional, mas requer o uso de cálculo multivariável e álgebra linear [10]:

1. ==Considere um pequeno volume $dV_z$ no espaço de $Z$ e o correspondente volume $dV_x$ no espaço de $X$.==

2. ==A relação entre estes volumes é dada pelo determinante do Jacobiano:==
   
   $dV_x = \left|\det\left(\frac{\partial f(z)}{\partial z}\right)\right| dV_z$

3. A conservação da probabilidade implica:
   
   $p_Z(z)dV_z = p_X(x)dV_x$

4. Substituindo e rearranjando:
   
   $p_X(x) = p_Z(z) \left|\det\left(\frac{\partial f(z)}{\partial z}\right)\right|^{-1}$

5. Expressando em termos de $x$ e usando a regra da cadeia para inversos:
   
   $p_X(x) = p_Z(f^{-1}(x)) \left|\det\left(\frac{\partial f^{-1}(x)}{\partial x}\right)\right|$

> ❗ **Ponto de Atenção**: ==O cálculo do determinante do Jacobiano pode ser computacionalmente custoso para dimensões elevadas==, motivando o desenvolvimento de arquiteturas especiais em normalizing flows.

#### Propriedades Importantes

1. **Composição de Transformações**: Para uma sequência de transformações $f_1, f_2, ..., f_M$, a densidade final é dada por [11]:

   $$
   p_X(x) = p_Z(f_M^{-1}(...f_2^{-1}(f_1^{-1}(x))...)) \prod_{m=1}^M \left|\det\left(\frac{\partial f_m^{-1}(z_m)}{\partial z_m}\right)\right|
   $$

2. **Transformações Triangulares**: ==Se o Jacobiano é triangular, seu determinante é simplesmente o produto dos elementos diagonais==, reduzindo significativamente o custo computacional [12].

#### Questões Técnicas/Teóricas

1. Como o teorema da função implícita se relaciona com a fórmula de mudança de variáveis no caso multidimensional?
2. Dada uma transformação linear $X = AZ + b$, onde $A$ é uma matriz invertível e $b$ é um vetor constante, derive a expressão para $p_X(x)$ em termos de $p_Z(z)$.

### Aplicações em Normalizing Flows

[![](https://mermaid.ink/img/pako:eNqVlNtO4zAQhl_F8kUFUqja9EglVgJKgdITtJyW7IWpTbGU2JETs22Bh1ntBQ-wj9AX24nDNkkpC-QmyT-f_4xnJn7EY0kZbuCJIv49GjUdgeAK9G0sOHjgEkEUarnyp4PjaHTt3ji4yYNQ8VvNFy-L3xLtkYChuYN_oK2tb2gPgJEiIriTyiMxEXsBkfjsGXj_rVuLC-Ki6Rq4CXCbjOUtJ0KidoZoGuIACMrCp_ZTJtgCubX4ozztEkQZ6mpKxOLFPF8QxRe_HhgPMkt2UQ4dGM9WyifOORaYoI54U7V9qX2XiwnqkBlT6cIdflS4IwCGsDhE86KF5nYmnyODHAMyL6KoQiFThMo1THtN_WGfK34nUUm0iMOBFf5Lop0gx0bo3ESbEmMSstUWtmMgETpG6H6up71MN0cKbhPoz7pP9BKhv-ww2kFs6m8E2tsINjez1mZVPxEGXx-AQxiAvvEZpHziDf5vAM4Y7LR3MUi3_vSj1p9FRSYeoSaj3bH0XeIxEUpUzCR1Zujhu3S2x0NDj4DO5_OZyMhEzt_16WXoc0NffK6tl9m2yhCIHTRQkmowpjJAy2iwsrccZJyD3HKvX7xMglfptqfNXkUuKH_gVJOVLl4ao6tEuP76JJxCQlfG5zrlE9dkOQnYwh6D_41TOFMfI9nB4T3zmIMbKMr9jmg3jIbiGVAC6Q9nYowbodLMwkrqyT1u3BE3gDftU_jbmpzATHlLlVEeStWNT21zeFvYJ-K7lAkD77jxiKe4sVUq1vOlcrVUs227ZldqNQvPIrlSzVfLlWqtXCtU7HJ1-9nCc2NRzFfqle1qYbtetkuFQrlYev4LwHLQZg?type=png)](https://mermaid.live/edit#pako:eNqVlNtO4zAQhl_F8kUFUqja9EglVgJKgdITtJyW7IWpTbGU2JETs22Bh1ntBQ-wj9AX24nDNkkpC-QmyT-f_4xnJn7EY0kZbuCJIv49GjUdgeAK9G0sOHjgEkEUarnyp4PjaHTt3ji4yYNQ8VvNFy-L3xLtkYChuYN_oK2tb2gPgJEiIriTyiMxEXsBkfjsGXj_rVuLC-Ki6Rq4CXCbjOUtJ0KidoZoGuIACMrCp_ZTJtgCubX4ozztEkQZ6mpKxOLFPF8QxRe_HhgPMkt2UQ4dGM9WyifOORaYoI54U7V9qX2XiwnqkBlT6cIdflS4IwCGsDhE86KF5nYmnyODHAMyL6KoQiFThMo1THtN_WGfK34nUUm0iMOBFf5Lop0gx0bo3ESbEmMSstUWtmMgETpG6H6up71MN0cKbhPoz7pP9BKhv-ww2kFs6m8E2tsINjez1mZVPxEGXx-AQxiAvvEZpHziDf5vAM4Y7LR3MUi3_vSj1p9FRSYeoSaj3bH0XeIxEUpUzCR1Zujhu3S2x0NDj4DO5_OZyMhEzt_16WXoc0NffK6tl9m2yhCIHTRQkmowpjJAy2iwsrccZJyD3HKvX7xMglfptqfNXkUuKH_gVJOVLl4ao6tEuP76JJxCQlfG5zrlE9dkOQnYwh6D_41TOFMfI9nB4T3zmIMbKMr9jmg3jIbiGVAC6Q9nYowbodLMwkrqyT1u3BE3gDftU_jbmpzATHlLlVEeStWNT21zeFvYJ-K7lAkD77jxiKe4sVUq1vOlcrVUs227ZldqNQvPIrlSzVfLlWqtXCtU7HJ1-9nCc2NRzFfqle1qYbtetkuFQrlYev4LwHLQZg)

A fórmula de mudança de variáveis é o fundamento teórico dos normalizing flows, permitindo a construção de modelos generativos complexos através de uma série de transformações invertíveis [13]. Algumas aplicações notáveis incluem:

1. **Planar Flows**: ==Transformações da forma $f(z) = z + uh(w^Tz + b)$, onde o Jacobiano tem uma estrutura especial que permite cálculo eficiente [14].==

2. **Coupling Layers**: ==Dividem o vetor de entrada em duas partes, aplicando transformações que garantem Jacobianos triangulares [15]:==
   $$
   x_A = z_A, \quad x_B = \exp(s(z_A)) \odot z_B + t(z_A)
   $$
   
3. **Autoregressive Flows**: Exploram a estrutura autoregressiva para criar transformações com Jacobianos triangulares [16]:

   $$
   x_i = h(z_i, g_i(x_{1:i-1}))
   $$

> 💡 **Insight**: A escolha da arquitetura do flow é frequentemente guiada pelo trade-off entre expressividade da transformação e eficiência no cálculo do determinante do Jacobiano.

#### Implementação em PyTorch

Aqui está um exemplo simplificado de como implementar uma camada de coupling flow em PyTorch:

```python
import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1
        s, t = torch.chunk(self.nn(x1), 2, dim=-1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=-1)
    
    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=-1)
        x1 = y1
        s, t = torch.chunk(self.nn(y1), 2, dim=-1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=-1)
    
    def log_det_jacobian(self, x):
        x1, _ = torch.chunk(x, 2, dim=-1)
        s, _ = torch.chunk(self.nn(x1), 2, dim=-1)
        return torch.sum(s, dim=-1)
```

Este exemplo demonstra como a fórmula de mudança de variáveis é aplicada na prática, com o método `log_det_jacobian` calculando o logaritmo do determinante do Jacobiano de forma eficiente.

#### Questões Técnicas/Teóricas

1. Como a estrutura de coupling layers garante a invertibilidade da transformação e facilita o cálculo do determinante do Jacobiano?
2. Discuta as vantagens e desvantagens de usar flows autoregressivos versus coupling flows em termos de expressividade e eficiência computacional.

### Desafios e Avanços Recentes

1. **Escalabilidade**: Para dados de alta dimensão, o cálculo do determinante do Jacobiano pode se tornar proibitivo. Técnicas como flows contínuos e estimadores de traço estocásticos têm sido propostas para mitigar este problema [17].

2. **Expressividade vs. Tratabilidade**: Existe um trade-off entre a complexidade das transformações e a facilidade de calcular seus Jacobianos. Pesquisas recentes focam em encontrar o equilíbrio ideal [18].

3. **Flows Condicionais**: Extensões da fórmula de mudança de variáveis para flows condicionais, permitindo a modelagem de distribuições condicionais complexas [19].

### Conclusão

A fórmula de mudança de variáveis é o alicerce matemático sobre o qual os normalizing flows são construídos. Sua compreensão profunda é essencial para o desenvolvimento, implementação e inovação neste campo em rápida evolução. Desde sua formulação básica unidimensional até suas aplicações complexas em espaços multidimensionais, esta fórmula fornece o framework necessário para transformar distribuições simples em modelos generativos poderosos e flexíveis.

À medida que o campo avança, espera-se que novas técnicas e formulações surjam, possivelmente redefinindo ou estendendo a aplicação da fórmula de mudança de variáveis. Cientistas de dados e pesquisadores em aprendizado de máquina devem, portanto, manter um entendimento sólido deste conceito fundamental, preparando-se para suas evoluções futuras e aplicações inovadoras.

### Questões Avançadas

1. Como você derivaria a fórmula de mudança de variáveis para o caso de transformações estocásticas, onde a função de transformação inclui um componente aleatório?

2. Discuta as implicações da fórmula de mudança de variáveis na construção de priors informativas em inferência bayesiana usando normalizing flows.

3. Proponha e analise teoricamente uma nova arquitetura de flow que otimize o trade-off entre expressividade da transformação e eficiência no cálculo do determinante do Jacobiano.

4. Como a fórmula de mudança de variáveis poderia ser estendida ou modificada para lidar com espaços de dimensão infinita, como em processos estocásticos contínuos?

5. Analise as conexões entre a fórmula de mudança de variáveis em normalizing flows e o conceito de transporte ótimo em teoria da medida. Como essas conexões poderiam ser exploradas para desenvolver flows mais eficientes?

### Referências

[1] "Change of variables (1D case): If $X = f(Z)$ and $f(\cdot)$ is monotone with inverse $Z = f^{-1}(X) = h(X)$, then: $p_X(x) = p_Z(h(x))|h'(x)|$" (Trecho de Normalizing Flow Models - Lecture Notes)

[2] "Key idea behind flow models: Map simple distributions (easy to sample and evaluate densities) to complex distributions through an invertible transformation." (Trecho de Normalizing Flow Models - Lecture Notes)

[3] "Even though $p(z)$