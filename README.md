# Metropolis-Hastings Algorithm for Linear Regression

The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC) method used for sampling from a probability distribution. Here, we'll use it to estimate the parameters of a linear regression model.

## Linear Regression Model

In linear regression, we model the relationship between the independent variable \( x \) and the dependent variable \( y \) using the equation:


\[ y = a + bx + \epsilon \]

where:
- \( y \) is the dependent variable
- \( x \) is the independent variable
- \( a \) is the intercept
- \( b \) is the slope
- \( \epsilon \) is the error term

Our goal is to estimate the parameters \( a \) and \( b \) from the observed data.

## Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm iteratively samples from the posterior distribution of the parameters \( a \) and \( b \) given the data \( (x_i, y_i) \). The algorithm proceeds as follows:

1. Initialize \( a \) and \( b \) with random values.
2. Compute the likelihood of the observed data given the current parameters \( a \) and \( b \).
3. Propose new values of \( a \) and \( b \) using a proposal distribution (e.g., Gaussian).
4. Compute the likelihood of the observed data given the proposed parameters.
5. Compute the acceptance ratio using the likelihoods of the current and proposed parameters.
6. Accept the proposed parameters with probability equal to the acceptance ratio; otherwise, keep the current parameters.
7. Repeat steps 2-6 for a predefined number of iterations.

## Equations

### Likelihood Function

The likelihood function \( L(a, b) \) given the observed data \( (x_i, y_i) \) is given by:

\[ L(a, b) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - (a + bx_i))^2}{2\sigma^2}\right) \]

where \( \sigma^2 \) is the variance of the error term.

### Acceptance Ratio

The acceptance ratio \( \alpha \) for proposing new parameters \( a' \) and \( b' \) given the current parameters \( a \) and \( b \) is given by:

\[ \alpha = \min\left(1, \frac{L(a', b')}{L(a, b)}\right) \]

## Implementation

To implement the Metropolis-Hastings algorithm for linear regression, you can use your preferred programming language (e.g., Python) and libraries such as NumPy and SciPy.

