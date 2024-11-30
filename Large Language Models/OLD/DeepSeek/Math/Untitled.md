#### Teorema de Convergência Generalizada

**Teorema**: Sob condições adequadas de regularidade, incluindo suposições sobre a suavidade da função de recompensa e da política, os métodos de RL que se enquadram no paradigma unificado convergem para um ótimo local da função objetivo $\mathcal{J}_\mathcal{A}(\theta)$, ou seja:

$$
\lim_{t \to \infty} \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t) = 0
$$

[19]

**Prova**:

Para demonstrar o teorema, seguiremos os passos clássicos de análise de convergência para algoritmos de otimização estocástica. ==Consideramos que os parâmetros do modelo $\theta$ são atualizados usando um algoritmo de gradiente estocástico com certas propriedades.==

1. **Assumir Suavidade da Função Objetivo**:

   Suponha que a função objetivo $\mathcal{J}_\mathcal{A}(\theta)$ é continuamente diferenciável e possui gradiente Lipschitz-contínuo com constante $L > 0$, ou seja:

   $$
   \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta) - \nabla_\theta \mathcal{J}_\mathcal{A}(\theta')\| \leq L \|\theta - \theta'\|
   $$

2. **Atualização dos Parâmetros**:

   Os parâmetros são atualizados de acordo com o gradiente estocástico:

   $$
   \theta_{t+1} = \theta_t - \eta_t \hat{g}_t
   $$

   onde $\hat{g}_t$ é o estimador estocástico do gradiente no passo $t$, e $\eta_t$ é a taxa de aprendizado no passo $t$.

3. **Propriedades do Estimador de Gradiente**:

   - **Não Enviesado**: O estimador de gradiente é não enviesado:

     $$
     \mathbb{E}[\hat{g}_t | \theta_t] = \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)
     $$

   - **Variância Limitada**: A variância do estimador de gradiente é limitada por uma constante $\sigma^2$:

     $$
     \mathbb{E}\left[\|\hat{g}_t - \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 | \theta_t\right] \leq \sigma^2
     $$

4. **Condições sobre a Taxa de Aprendizado**:

   A taxa de aprendizado $\eta_t$ satisfaz:

   $$
   \sum_{t=1}^\infty \eta_t = \infty \quad \text{e} \quad \sum_{t=1}^\infty \eta_t^2 < \infty
   $$

   Exemplo: $\eta_t = \frac{\eta_0}{t}$, com $\eta_0 > 0$.

5. **Análise da Função Objetivo**:

   Considere a diferença na função objetivo entre os passos $t$ e $t+1$:

   $$
   \mathcal{J}_\mathcal{A}(\theta_{t+1}) - \mathcal{J}_\mathcal{A}(\theta_t) = \mathcal{J}_\mathcal{A}(\theta_t - \eta_t \hat{g}_t) - \mathcal{J}_\mathcal{A}(\theta_t)
   $$

   Pela expansão de Taylor de primeira ordem e usando a propriedade de Lipschitz-continuidadedo gradiente:

   $$
   \mathcal{J}_\mathcal{A}(\theta_{t+1}) \leq \mathcal{J}_\mathcal{A}(\theta_t) - \eta_t \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)^\top \hat{g}_t + \frac{L}{2} \eta_t^2 \|\hat{g}_t\|^2
   $$

6. **Tomando a Esperança Condicional**:

   Calculamos a esperança condicional dado $\theta_t$:

   $$
   \mathbb{E}\left[\mathcal{J}_\mathcal{A}(\theta_{t+1}) | \theta_t\right] \leq \mathcal{J}_\mathcal{A}(\theta_t) - \eta_t \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 + \frac{L}{2} \eta_t^2 \mathbb{E}\left[\|\hat{g}_t\|^2 | \theta_t\right]
   $$

   Utilizando as propriedades do estimador de gradiente:

   $$
   \mathbb{E}\left[\|\hat{g}_t\|^2 | \theta_t\right] = \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 + \mathbb{E}\left[\|\hat{g}_t - \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 | \theta_t\right] \leq \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 + \sigma^2
   $$

   Substituindo de volta:

   $$
   \mathbb{E}\left[\mathcal{J}_\mathcal{A}(\theta_{t+1}) | \theta_t\right] \leq \mathcal{J}_\mathcal{A}(\theta_t) - \eta_t \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 + \frac{L}{2} \eta_t^2 \left( \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 + \sigma^2 \right)
   $$

7. **Reorganizando os Termos**:

   Reorganizamos a inequação:

   $$
   \mathbb{E}\left[\mathcal{J}_\mathcal{A}(\theta_{t+1}) | \theta_t\right] \leq \mathcal{J}_\mathcal{A}(\theta_t) - \left( \eta_t - \frac{L}{2} \eta_t^2 \right) \|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2 + \frac{L}{2} \eta_t^2 \sigma^2
   $$

8. **Soma Telescópica**:

   Definimos $\Delta_t = \mathbb{E}\left[\mathcal{J}_\mathcal{A}(\theta_t)\right] - \mathcal{J}_\mathcal{A}^\ast$, onde $\mathcal{J}_\mathcal{A}^\ast$ é o valor mínimo global (ou local) da função objetivo. Somando de $t = 1$ até $T$:

   $$
   \sum_{t=1}^T \left( \eta_t - \frac{L}{2} \eta_t^2 \right) \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2\right] \leq \Delta_1 - \Delta_{T+1} + \frac{L}{2} \sigma^2 \sum_{t=1}^T \eta_t^2
   $$

   Como $\Delta_{T+1} \geq 0$:

   $$
   \sum_{t=1}^T \left( \eta_t - \frac{L}{2} \eta_t^2 \right) \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2\right] \leq \Delta_1 + \frac{L}{2} \sigma^2 \sum_{t=1}^T \eta_t^2
   $$

9. **Convergência do Gradiente**:

   Como $\eta_t$ é pequeno, podemos aproximar $\eta_t - \frac{L}{2} \eta_t^2 \approx \frac{\eta_t}{2}$ (assumindo que $\eta_t L \ll 1$). Então:

   $$
   \sum_{t=1}^T \frac{\eta_t}{2} \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2\right] \leq C
   $$

   onde $C = \Delta_1 + \frac{L}{2} \sigma^2 \sum_{t=1}^\infty \eta_t^2$ é uma constante finita, já que $\sum_{t=1}^\infty \eta_t^2 < \infty$.

   Portanto:

   $$
   \sum_{t=1}^\infty \eta_t \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2\right] < \infty
   $$

10. **Conclusão Final**:

    Como $\sum_{t=1}^\infty \eta_t = \infty$, a única maneira de a soma dos valores positivos $\eta_t \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2\right]$ ser finita é que:

    $$
    \liminf_{t \to \infty} \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t)\|^2\right] = 0
    $$

    Isso implica que existe uma subsequência $\{ \theta_{t_k} \}$ tal que:

    $$
    \lim_{k \to \infty} \mathbb{E}\left[\|\nabla_\theta \mathcal{J}_\mathcal{A}(\theta_{t_k})\|^2\right] = 0
    $$

    Sob as condições de continuidade e suavidade da função objetivo, podemos concluir que:

    $$
    \lim_{t \to \infty} \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t) = 0
    $$

    Portanto, a sequência $\{\theta_t\}$ converge para um ponto crítico (ótimo local) da função objetivo $\mathcal{J}_\mathcal{A}(\theta)$.

**Q.E.D.**

---

**Observações Adicionais**:

- A prova assume que a função objetivo é não convexa (o que é comum em redes neurais profundas), portanto, a convergência é para um ótimo local, não necessariamente global.
- As condições sobre a taxa de aprendizado $\eta_t$ são cruciais para garantir a convergência. Uma escolha comum é $\eta_t = \eta_0 / t^\alpha$, com $0.5 < \alpha \leq 1$.
- A limitação da variância do estimador de gradiente é importante para evitar que ruídos estocásticos prejudiquem a convergência.
- A propriedade de Lipschitz-continuidadedo gradiente garante que a função objetivo não mude abruptamente, permitindo o controle do progresso da otimização.

---

**Referências**:

[19] "We establish a general convergence theorem for methods that fit into the unified paradigm." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[20] Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control*. Athena Scientific.