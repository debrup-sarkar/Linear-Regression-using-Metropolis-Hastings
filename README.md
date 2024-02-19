# Linear Regression using Metropolis Hastings Algorithm

The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo method used for sampling from a probability distribution, particularly useful for estimating parameters in Bayesian inference problems. It iteratively proposes new samples based on a proposal distribution, accepting or rejecting them based on an acceptance ratio calculated from the likelihood of the proposed and current samples.

## The model
$\huge{y = ax + b + \epsilon} $  
$\huge{\epsilon \sim N(0, \sigma^{2})}$ 

## The Algorithm
* Initialize the point $\theta_{0}$ from an arbitrary initial distribution.
* For i = 1,2,......,N do \
  1.Sample a candidate point **$\theta^{*}$** from a proposal distribution.
  $\huge{\theta^{*} ~ q(\theta^{*} /| \theta^{i-1})}$
