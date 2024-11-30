* Tópico: Real NVP (real-valued non-volume-preserving)
  * Contexto: Um tipo de fluxo de normalização de acoplamento que particiona as variáveis latentes e usa transformações invertíveis para modelar distribuições flexíveis.
    * Lista de trechos relevantes:
      - "One solution to this problem is given by a form of normalizing flow model called real NVP... which is short for 'real-valued non-volume-preserving'."
      - "The idea is to partition the latent-variable vector 𝑧 into two parts 𝑧=(𝑧𝐴,𝑧𝐵)... "
      - "For the first part of the output vector, we simply copy the input: 𝑥𝐴=𝑧𝐴."
      - "The second part of the vector undergoes a linear transformation, but now the coefficients in the linear transformation are given by nonlinear functions of 𝑧𝐴: 𝑥𝐵=exp(𝑠(𝑧𝐴,𝑤))⊙𝑧𝐵+𝑏(𝑧𝐴,𝑤)"
      - "The overall transformation is easily invertible..."
      - "The real NVP model belongs to a broad class of normalizing flows called coupling flows..."

* Tópico: Invertible mapping
  * Contexto: A necessidade de mapeamentos bijetivos (invertíveis) entre o espaço latente e o espaço de dados para calcular a função de verossimilhança em fluxos de normalização.
    * Lista de trechos relevantes:
      -  "To calculate the likelihood function for this model, we need the data-space distribution, which depends on the inverse of the neural network function."
      - "This requires that, for every value of 𝑤, the functions 𝑓(𝑧,𝑤) and 𝑔(𝑥,𝑤) are invertible, also called bijective, so that each value of 𝑥 corresponds to a unique value of 𝑧 and vice versa."
      - "One consequence of requiring an invertible mapping is that the dimensionality of the latent space must be the same as that of the data space..."

* Tópico: Jacobian determinant calculation
  * Contexto: A importância e os métodos para calcular o determinante Jacobiano de forma eficiente, especialmente para matrizes triangulares inferiores, em fluxos de normalização.
    * Lista de trechos relevantes:
      - "We can then use the change of variables formula to calculate the data density: 𝑝𝑥(𝑥|𝑤)=𝑝𝑧(𝑔(𝑥,𝑤))|det𝐽(𝑥)|"
      - "Also, in general, the cost of evaluating the determinant of a 𝐷×𝐷 matrix is 𝑂(𝐷3), so we will seek to impose some further restrictions on the model in order that evaluation of the Jacobian matrix determinant is more efficient."
      - "We therefore see that the Jacobian matrix (18.14) is a lower triangular matrix... Consequently, the determinant of the Jacobian is simply given by the product of the elements of exp(−𝑠(𝑧𝐴,𝑤))."

* Tópico: Coupling function (h(zB, g))
  * Contexto: Uma função em fluxos de acoplamento que opera em zB e é eficientemente invertível dado o condicionador g(zA, w).
    * Lista de trechos relevantes:
      - "The real NVP model belongs to a broad class of normalizing flows called coupling flows, in which the linear transformation (18.11) is replaced by a more general form: xB = h(zB, g(zA, w))"
      - "where ℎ(𝑧𝐵,𝑔) is a function of 𝑧𝐵 that is efficiently invertible for any given value of 𝑔 and is called the coupling function."


* Tópico: Conditioner (g(zA, w))
  * Contexto: Uma função, tipicamente uma rede neural, que fornece flexibilidade para a transformação em fluxos de acoplamento.
    * Lista de trechos relevantes:
      - "The function 𝑔(𝑧𝐴,𝑤) is called a conditioner and is typically represented by a neural network."

* Tópico: Layer composition
  * Contexto: Combinar múltiplas camadas de transformações invertíveis para criar fluxos de normalização mais flexíveis.
    * Lista de trechos relevantes:
      - "A clear limitation of this approach is that the value of  𝑧𝐴  is unchanged by the transformation. This is easily resolved by adding another layer in which the roles of  𝑧𝐴  and  𝑧𝐵  are reversed, as illustrated in Figure 18.2. This double-layer structure can then be repeated multiple times to facilitate a very flexible class of generative models."
      - "By composing two layers of the form shown in Figure 18.1, we obtain a more flexible, but still invertible, nonlinear layer."


* Tópico: Masked Autoregressive Flow (MAF)
  * Contexto: Um tipo de fluxo de normalização autoregressivo que usa distribuições condicionais e redes neurais mascaradas para modelar distribuições complexas.
    * Lista de trechos relevantes:
      - "This factorization can be used to construct a class of normalizing flow called a masked autoregressive flow, or MAF... given by 𝑥𝑖=ℎ(𝑧𝑖,𝑔𝑖(𝑥1:𝑖−1,𝑤𝑖))"
      - "Here ℎ(𝑧𝑖,⋅) is the coupling function, which is chosen to be easily invertible with respect to 𝑧𝑖, and 𝑔𝑖 is the conditioner, which is typically represented by a deep neural network."
      - "The term masked refers to the use of a single neural network to implement a set of equations of the form (18.17) along with a binary mask... that force a subset of the network weights to be zero to implement the autoregressive constraint (18.16)."


* Tópico: Inverse Autoregressive Flow (IAF)
  * Contexto: Um tipo de fluxo de normalização autoregressivo que permite amostragem eficiente, mas requer cálculos sequenciais para a função de verossimilhança.
    * Lista de trechos relevantes:
      - "To avoid this inefficient sampling, we can instead define an inverse autoregressive flow, or IAF... given by 𝑥𝑖=ℎ(𝑧𝑖,𝑔̃𝑖(𝑧1:𝑖−1,𝑤𝑖))"
      - "Sampling is now efficient since, for a given choice of z, the evaluation of the elements 𝑥1,…,𝑥𝐷 using (18.19) can be performed in parallel. However, the inverse function, which is needed to evaluate the likelihood, requires a series of calculations... which are intrinsically sequential and therefore slow."

* Tópico: Autoregressive constraint
  * Contexto: A restrição de que cada variável em um fluxo autoregressivo depende apenas das variáveis anteriores na sequência.
    * Lista de trechos relevantes:
      - "We first choose an ordering of the variables... from which we can write, without loss of generality, 𝑝(𝑥1,…,𝑥𝐷)=∏𝑖=1𝐷𝑝(𝑥𝑖|𝑥1:𝑖−1)"
      - "...that force a subset of the network weights to be zero to implement the autoregressive constraint (18.16)."


* Tópico: Neural ODEs
  * Contexto: Redes neurais definidas por equações diferenciais, permitindo uma quantidade infinita de camadas.
    * Lista de trechos relevantes:
      - "This can be thought of as a deep network with an infinite number of layers."
      - "The formulation in (18.22) is known as a neural ordinary differential equation or neural ODE..."


* Tópico: Adjoint sensitivity method
  * Contexto: Um método para calcular gradientes em ODEs neurais, análogo à retropropagação em redes neurais padrão.
    * Lista de trechos relevantes:
      - "Instead, Chen et al. (2018) treat the ODE solver as a black box and use a technique called the adjoint sensitivity method, which can be viewed as the continuous analogue of explicit backpropagation."
      - "To apply backpropagation to neural ODEs, we define a quantity called the adjoint given by 𝑎(𝑡)=𝑑𝐿𝑑𝑧(𝑡)."


* Tópico: Adaptive function evaluation
  * Contexto: Métodos de integração numérica que adaptam os pontos de avaliação da função para eficiência e precisão.
    * Lista de trechos relevantes:
      - "This integral can be evaluated using standard numerical integration packages."
      - "In practice, more powerful numerical integration algorithms can adapt their function evaluation to achieve. In particular, they can adaptively choose values of  𝑡  that typically are not uniformly spaced."


* Tópico: Adjoint method
  * Contexto: O análogo contínuo da retropropagação para treinar ODEs neurais.
    * Lista de trechos relevantes:
      -  "Chen et al. (2018) treat the ODE solver as a black box and use a technique called the adjoint sensitivity method, which can be viewed as the continuous analogue of explicit backpropagation."


* Tópico: Adjoint differential equation
  * Contexto: A equação diferencial que descreve a evolução do adjunto durante a retropropagação em ODEs neurais.
    * Lista de trechos relevantes:
      - "The adjoint satisfies its own differential equation given by 𝑑𝑎(𝑡)𝑑𝑡=−𝑎(𝑡)𝑇∇𝑧𝑓(𝑧(𝑡),𝑤), which is a continuous version of the chain rule of calculus."


* Tópico: Gradient evaluation (for Neural ODEs)
  * Contexto: Calculando gradientes em relação aos parâmetros da rede em ODEs neurais por meio de integração.
    * Lista de trechos relevantes:
      - "The third step in the backpropagation method is to evaluate derivatives of the loss with respect to network parameters by forming appropriate products of activations and gradients... this summation becomes an integration over  𝑡 , which takes the form ∇𝑤𝐿=−∫0𝑇𝑎(𝑡)𝑇∇𝑤𝑓(𝑧(𝑡),𝑤)𝑑𝑡."


* Tópico: Continuous normalizing flow
  * Contexto: Usando uma ODE neural para definir um fluxo de normalização contínuo, propagando a distribuição base através do tempo.
    * Lista de trechos relevantes:
      - "We can make use of a neural ordinary differential equation to define an alternative approach to the construction of tractable normalizing flow models... The resulting framework is known as a continuous normalizing flow..."


* Tópico: Density transformation (for Continuous Flows)
  * Contexto: Calculando a mudança na densidade de probabilidade à medida que a distribuição base é propagada através da ODE neural.
    * Lista de trechos relevantes:
      - "Chen et al. (2018) showed that for neural ODEs, the transformation of the density can be evaluated by integrating a differential equation given by 𝑑ln𝑝(𝑧(𝑡))𝑑𝑡=−Tr(∂𝑓∂𝑧(𝑡))"


* Tópico: Hutchinson's trace estimator
  * Contexto: Um método eficiente para aproximar o traço de uma matriz, usado em fluxos de normalização contínuos.
    * Lista de trechos relevantes:
      - "However, the cost of evaluating the trace can be reduced to 𝑂(𝐷) by using Hutchinson’s trace estimator..."
      - "Tr(𝐴)=𝐸𝜖[𝜖𝑇𝐴𝜖]"


* Tópico: Flow matching
  * Contexto: Uma técnica para melhorar a eficiência do treinamento de fluxos de normalização contínuos.
    * Lista de trechos relevantes:
      - "Significant improvements in training efficiency for continuous normalizing flows can be achieved using a technique called flow matching (Lipman et al., 2022)."

* Tópico: Jacobian and determinant relationships
  * Contexto: Relações matemáticas entre Jacobianos e determinantes em transformações de variáveis.
    * Lista de trechos relevantes:
      - Exercício 18.1

* Tópico: Invertible transformations
  * Contexto: Compor e inverter múltiplas transformações invertíveis.
    * Lista de trechos relevantes:
      - Exercício 18.2

* Tópico: Linear transformations
  * Contexto: Transformações lineares e seus Jacobianos.
    * Lista de trechos relevantes:
      - Exercício 18.3

* Tópico: Autoregressive flow Jacobians
  * Contexto: A estrutura triangular inferior dos Jacobianos em fluxos autoregressivos.
    * Lista de trechos relevantes:
      - Exercício 18.4


* Tópico: Residual network to ODE
  * Contexto: Derivando a equação de propagação direta para uma ODE neural a partir de uma rede residual.
    * Lista de trechos relevantes:
      - Exercício 18.5

* Tópico: Backpropagation for ODEs
  * Contexto: Derivando a equação de retropropagação para ODEs neurais.
    * Lista de trechos relevantes:
      - Exercício 18.6

* Tópico: Gradient evaluation (for ODEs - Exercises)
  * Contexto: Derivando a equação de avaliação de gradiente para ODEs neurais.
    * Lista de trechos relevantes:
      - Exercício 18.7


* Tópico: One-dimensional density transformation
  * Contexto: Derivando a equação para fluxos de normalização contínuos em uma dimensão.
    * Lista de trechos relevantes:
      - Exercício 18.8

* Tópico: Flow line plotting
  * Contexto: Plotando linhas de fluxo usando a inversa da CDF e a ODE neural.
    * Lista de trechos relevantes:
      - Exercício 18.9

* Tópico: Base density and flow inversion
  * Contexto: A relação entre a densidade base e a densidade de saída em fluxos de normalização contínuos.
    * Lista de trechos relevantes:
      - Exercício 18.10

* Tópico: Hutchinson trace estimator - Proving unbiasedness
  * Contexto: Provando que o estimador de traço de Hutchinson é não viesado.
    * Lista de trechos relevantes:
      - Exercício 18.11