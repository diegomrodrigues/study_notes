### **Problema 1**

Considere a equação de diferença linear de segunda ordem:

$$
y_t = 1{,}2 \, y_{t-1} - 0{,}32 \, y_{t-2} + w_t
$$

**a) Reescreva esta equação na forma vetorial $\xi_t = F \xi_{t-1} + v_t$, identificando a matriz de transição $F$, o vetor de estado $\xi_t$ e o vetor de entrada $v_t$.**

**b) Calcule os autovalores da matriz $F$.**

**c) Determine se o sistema é estável.**

**d) Suponha que $w_t = 0$ para todo $t$, e as condições iniciais são $y_{-1} = 2$ e $y_{-2} = 1$. Calcule $y_t$ para $t = 0, 1, 2, 3, 4, 5$.**

**e) Analise o comportamento de $y_t$ ao longo do tempo com base nos resultados anteriores.**

---

### **Problema 2**

Para a equação de diferença de terceira ordem:

$$
y_t = 0{,}5 \, y_{t-1} - 0{,}1 \, y_{t-2} + 0{,}05 \, y_{t-3} + w_t
$$

**a) Determine a matriz de transição $F$ correspondente.**

**b) Calcule os autovalores da matriz $F$.**

**c) Verifique se os autovalores satisfazem as condições de estabilidade e conclua sobre a estabilidade do sistema.**

**d) Supondo que $w_t = 0$ para todo $t$ e condições iniciais $y_{-1} = 1$, $y_{-2} = 0$, $y_{-3} = -1$, simule numericamente o sistema e calcule $y_t$ para $t = 0$ a $t = 10$.**

**e) Descreva o comportamento dinâmico do sistema com base nos autovalores e nos valores calculados.**

---

### **Problema 3**

Considere a equação de diferença linear de segunda ordem:

$$
y_t = -0{,}4 \, y_{t-1} + 0{,}08 \, y_{t-2} + w_t
$$

**a) Calcule o multiplicador dinâmico $\frac{\partial y_{t+2}}{\partial w_t}$.**

**b) Utilize a matriz $F$ correspondente e calcule $F^2$ para determinar $f_{11}^{(2)}$, o elemento $(1,1)$ de $F^2$.**

**c) Verifique a relação entre $\frac{\partial y_{t+2}}{\partial w_t}$ e $f_{11}^{(2)}$.**

**d) Suponha que um choque unitário ocorre em $w_0$ (ou seja, $w_0 = 1$ e $w_t = 0$ para $t \neq 0$), e que as condições iniciais são $y_{-1} = y_{-2} = 0$. Calcule $y_t$ para $t = 0$ a $t = 5$.**

**e) Analise como o choque se propaga no sistema e como os multiplicadores dinâmicos influenciam essa propagação.**

---

### **Problema 4**

Um sistema dinâmico é descrito pela equação de diferença de segunda ordem:

$$
y_t = \phi_1 \, y_{t-1} + \phi_2 \, y_{t-2} + w_t
$$

Sabe-se que os autovalores do sistema são $\lambda_1 = 0{,}6$ e $\lambda_2 = -0{,}2$.

**a) Determine os valores de $\phi_1$ e $\phi_2$.**

**b) Verifique se o sistema é estável.**

**c) Suponha que $w_t = 0$ para todo $t$ e condições iniciais $y_{-1} = 1$ e $y_{-2} = 0$. Calcule $y_t$ para $t = 0$ a $t = 5$.**

**d) Analise o comportamento do sistema e explique como os autovalores influenciam esse comportamento.**

---

### **Problema 5**

Considere a equação de diferença linear de terceira ordem:

$$
y_t = 0{,}7 \, y_{t-1} - 0{,}25 \, y_{t-2} + 0{,}05 \, y_{t-3} + w_t
$$

**a) Reescreva a equação na forma vetorial e determine a matriz de transição $F$.**

**b) Calcule os autovalores de $F$ e classifique-os quanto à sua natureza (reais distintos, reais repetidos ou complexos conjugados).**

**c) Determine o multiplicador dinâmico $\frac{\partial y_{t+3}}{\partial w_t}$.**

**d) Supondo que $w_t = 0$ para todo $t$ e condições iniciais $y_{-1} = 2$, $y_{-2} = -1$, $y_{-3} = 0$, simule o sistema e calcule $y_t$ para $t = 0$ a $t = 10$.**

**e) Explique como a estrutura dos autovalores afeta a persistência dos choques no sistema.**

---

Esses problemas exigem a aplicação aprofundada dos conceitos apresentados no texto sobre equações de diferença de ordem $p$, representação vetorial, análise de estabilidade, autovalores e multiplicadores dinâmicos. A resolução passo a passo de cada item permitirá consolidar o entendimento teórico e prático dos sistemas dinâmicos discretos de ordem superior.