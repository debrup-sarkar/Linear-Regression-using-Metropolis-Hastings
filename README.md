# Linear Regression using Metropolis Hastings Algorithm

The Metropolis-Hastings algorithm is a **Markov Chain Monte Carlo (MCMC)** method used for sampling from a probability distribution, particularly useful for estimating parameters in Bayesian inference problems. It iteratively proposes new samples based on a proposal distribution, accepting or rejecting them based on an acceptance ratio calculated from the likelihood of the proposed and current samples.

## The model
$\huge{y = ax + b + \epsilon} $  
$\huge{\epsilon \sim N(0, \sigma^{2})}$ 

## The algorithm
* Initialize the point $\\mathbf{\theta}_{0}$ from an arbitrary initial distribution.
* For $i = 1, 2, \ldots, N$, do:
  * Sample a candidate point $\theta$ from a proposal distribution:\
    $\huge{mathbf{\theta^*}}$
    \mathbf{\theta^*} \mathrel{\sim} {q}(\mathbf{\theta^*} | \mathbf{\theta}^{i-1})
  * Evaluate the acceptance probability\
    $\huge{R = {\exp(\phi(\theta^{(i-1)}) - \phi(\theta^*))} \frac{q(\theta^{(i-1)}) | q(\theta^{\*}) }{q(\theta^{\*}) | q(\theta^{(i-1)})}}$\
    $\huge{\alpha_{i} = \min\\{1, R\\}}$

  * Generate a uniform random variable $u \mathrel{\sim} \mathcal{U}(0,1)$ and set\
    $\huge{\theta_{i} = theta^* \text{ if } \alpha_i \ge \text{u}}$ \
    $\huge{\theta_i = theta^{(i-1)} \text{ otherwise}} $

## References

[1] Särkkä S, Svensson L. Bayesian filtering and smoothing. Cambridge university press; 2023 May 31. [Link](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=bfb33c01461fe622c2f44f147822e95cea6fa5d0) 






    

