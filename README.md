# Linear Regression using Metropolis Hastings Algorithm

The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo method used for sampling from a probability distribution, particularly useful for estimating parameters in Bayesian inference problems. It iteratively proposes new samples based on a proposal distribution, accepting or rejecting them based on an acceptance ratio calculated from the likelihood of the proposed and current samples.

## The model
$\huge{y = ax + b + \epsilon} $  
$\huge{\epsilon \sim N(0, \sigma^{2})}$ 

## The Algorithm
* Initialize the point $\theta_{0}$ from an arbitrary initial distribution.
* For $i = 1, 2, \ldots, N$, do:
  * Sample a candidate point $\theta$ from a proposal distribution.\
    $\huge{\theta^* \mathrel{\sim} \mathcal{q}(\theta^* | \theta^{i-1})}$
  * Evaluate the acceptance probability\
    $\huge{\alpha_{i} = \min\\{1, R\\}}$\
    $R = {\exp(\phi(\theta^{(i-1)}) - \phi(\theta^*))} {/frac{A}{B}}$

    





    

