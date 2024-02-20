# Linear Regression using Metropolis Hastings Algorithm

The Metropolis-Hastings algorithm is a **Markov Chain Monte Carlo (MCMC)** method used for sampling from a probability distribution, particularly useful for estimating parameters in Bayesian inference problems. It iteratively proposes new samples based on a proposal distribution, accepting or rejecting them based on an acceptance ratio calculated from the likelihood of the proposed and current samples. 

## The model
A linear model can be expressed as\
$\huge{y = {X}^{T}{\theta} + \epsilon} $  
$\huge{\epsilon \sim N(0, \sigma^{2})}$ 

## The Metropolis Hastings algorithm
The the algorithm for parameter estimation by this method has been explained in chapter 12 of Bayesian filtering and smoothing by Särkkä and Svensson [1].
* Initialize the point $\\mathbf{\theta}_{0}$ from an arbitrary initial distribution.
* For $i = 1, 2, \ldots, N$, do:
  * Sample a candidate point $\theta$ from a proposal distribution:\
    $\huge{\theta^* \mathrel{\sim} \text{q }(\theta^* | \theta^{(i-1)})}$

  * Evaluate the acceptance probability
    
    $\huge{R = {\exp(\phi(\theta^{(i-1)}) - \phi(\theta^*))} \frac{q(\theta^{(i-1)}) | q(\theta^{\*}) }{q(\theta^{\*}) | q(\theta^{(i-1)})}}$\
    where $\phi$ is the energy function given as:
    
    $\huge{\phi_{T}(\theta) = -log (\text{ p}(y_{i:T} | \theta)) - log (\text{ p}(\theta)) }$\
 
    The acceptance probability ($\alpha$):

    $\huge{\alpha_{i} = \min\\{1, R\\}}$

  * Generate a uniform random variable $u \mathrel{\sim} \mathcal{U}(0,1)$ and set\
    $\huge{\theta_{i} = theta^* \text{ if } \alpha_i \ge \text{u}}$ \
    $\huge{\theta_i = theta^{(i-1)} \text{ otherwise}} $

## References

[1] Särkkä S, Svensson L. Bayesian filtering and smoothing. Cambridge university press; 2023 May 31.[Link](https://books.google.co.in/books?hl=en&lr=&id=utXBEAAAQBAJ&oi=fnd&pg=PP1&dq=bayesian+filtering+and+smoothing&ots=GX-dLQ7sTN&sig=aZTp8fQkWR6yzu1NrCQUvIWnYeA&redir_esc=y#v=onepage&q=bayesian%20filtering%20and%20smoothing&f=false) 

## See Also
- [Robust Adaptive Metropolis for parameter estimation in Linear Model](https://github.com/debrup-sarkar/Linear-Regression-using-Robust-Adaptive-Metropolis-MCMC-algorithm-in-MATLAB)







    

