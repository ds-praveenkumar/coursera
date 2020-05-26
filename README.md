# coursera
This repo contains the code and documnets related to various courses provided by Coursera 

### Important Formula
 * $sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$
 * $Activation = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
 * cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
 * $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
 * $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$
 * $ \theta = \theta - \alpha \text{ } d\theta$ 