import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

# sample x
np.random.seed(2023)
x = np.random.rand(100)*30

# set parameters 
a = 0.3
b = 200
sigma = 5

# Obtain response and add noise 
y = a*x + b
noise = np.random.randn(100)*sigma

# Matrix of predictor (1st column) and response (2nd column)
data = np.vstack((x,y)).T + noise.reshape(-1,1)


#plot data

#plt.figure(figsize = (7,5))
#plt.plot(data[:,0],data[:,1],'.')
#plt.scatter(data[:,0],data[:,1],color = 'r')
#plt.xlabel('X')
#plt.ylabel('Y')

def proposal(prec_theta, search_width = 0.5):
    out_theta = np.zeros(3)
    #print(out_theta.shape)
    out_theta[:2] = sc.multivariate_normal(mean = prec_theta[:2].squeeze(),cov = np.eye(2)*search_width**2).rvs()
    extra_adjus = 500;
    out_theta[2] = sc.gamma(a = prec_theta[2]*search_width*extra_adjus , scale = 1/(search_width*extra_adjus)).rvs()
    return out_theta

# log-likelihood

def lhd(x,theta):
    # x ----> data matrix |||||| first column --> input and second column --> output
    # theta ---- parameter matrix
    # theta[0]--a     theta[1] -- b     theta[2]-- sigma
    
    xs = x[:,0]
    ys = x[:,1]
    
    # likelihood = q(X,theta)
    lhd_out = sc.norm.logpdf(ys, loc = theta[0]*xs + theta[1], scale = theta[2])
    
    lhd_out = np.sum(lhd_out)
    return lhd_out

# Prior

def prior(theta):
    prior_out = sc.multivariate_normal.logpdf(theta[:2],mean=np.array([0,0]), cov=np.eye(2)*100)
    prior_out += sc.gamma.logpdf(theta[2], a=1, scale=1)
    return prior_out

# Proposal ratio

def proposal_ratio(theta_old, theta_new, search_width=10):
    # this is the proposal distribution ratio
    # first, we calculate of the pdf of the proposal distribution at the old value of theta with respect to the new 
    # value of theta. And then we do the exact opposite.
    #print(theta_old[:2].shape)
    #print(theta_new[:2].shape)
    prop_ratio_out = sc.multivariate_normal.logpdf(theta_old[:2],mean=theta_new[:2].squeeze(), cov=np.eye(2)*search_width**2)
    prop_ratio_out += sc.gamma.logpdf(theta_old[2], a=theta_new[2]*search_width*500, scale=1/(500*search_width))
    prop_ratio_out -= sc.multivariate_normal.logpdf(theta_new[:2],mean=theta_old[:2].squeeze(), cov=np.eye(2)*search_width**2)
    prop_ratio_out -= sc.gamma.logpdf(theta_new[2], a=theta_old[2]*search_width*500, scale=1/(500*search_width))
    return prop_ratio_out

np.random.seed(100)
width = 0.2
thetas = np.random.rand(3).reshape(-1,1)
theta_chain = thetas.T
theta_rejected = thetas.T
accepted = 0
rejected = 0
N = 50000
burn_in = 20000 

 

for i in range(N):
    
    if i%1000 == 0:
        print('{} itetations done'.format(i))
    
    # 1. Provide a proposal for theta
    
    theta_new = proposal(thetas,search_width = width)
    
    # 2. Calculate the likelihood of this proposal and the likelihood
    
    # For old value of theta
    log_lik_theta_new = lhd(data,theta_new)
    log_lik_theta = lhd(data,thetas)
    
    # 3. Evaluate the prior log-pdf at the current 
    
    theta_new_prior = prior(theta_new)
    theta_prior = prior(thetas.squeeze())
    
    
    # 4. Proposal ratio
    
    prop_ratio = proposal_ratio(thetas.squeeze(),theta_new,search_width = width)
    
    
    # 5. assemble the likelihood,prior and proposal
    likelihood_prior_proposal_ratio = log_lik_theta_new - log_lik_theta + theta_new_prior - theta_prior +  prop_ratio
    
    
    if np.exp(likelihood_prior_proposal_ratio) > sc.uniform().rvs():
        theta_chain = np.vstack((theta_chain,theta_new.reshape(3,1).T))
        thetas = theta_new
        #thetas = np.vstack((thetas.T,theta_new.reshape((3,1))))
        accepted += 1
    else:
        rejected += 1
        theta_chain = np.vstack((theta_chain,thetas.reshape(3,1).T))
        theta_rejected = np.vstack((theta_rejected,theta_new.reshape(3,1).T))


theta_chain = theta_chain[burn_in:-1,:]        
x_hat = np.linspace(-10,32,100)
y_hat = thetas[0]*x_hat + thetas[1]
plt.figure(figsize = (7,5))
plt.plot(x_hat,y_hat,'b-')
plt.scatter(data[:,0],data[:,1],color = 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.figure(figsize = (7,5))
plt.plot(theta_chain[:,0],'r',label = 'a')
plt.plot(theta_chain[:,1],'g',label = 'b')
plt.plot(theta_chain[:,2],'b',label = '$\sigma$')
# plt.plot(theta_rejected[:,0],'r.',label = 'a_rejected')
# plt.plot(theta_rejected[:,1],'g.',label = 'b_rejected')
# plt.plot(theta_rejected[:,2],'b.',label = '$\sigma$_rejected')

plt.ylabel('parameter values')
plt.xlabel('iterations')
plt.legend()
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 4))
n_bins = 40
mean_0 = np.mean(theta_chain[:, 0])
mean_1 = np.mean(theta_chain[:, 1])
mean_2 = np.mean(theta_chain[:, 2])

# Plot histograms
axes[0].hist(theta_chain[:, 0], bins=n_bins)
axes[0].axvline(mean_0, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_0:.2f}')
axes[0].set_title('Histogram for theta_chain[:, 0]')

axes[1].hist(theta_chain[:, 1], bins=n_bins)
axes[1].axvline(mean_1, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_1:.2f}')
axes[1].set_title('Histogram for theta_chain[:, 1]')

axes[2].hist(theta_chain[:, 2], bins=n_bins)
axes[2].axvline(mean_2, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_2:.2f}')
axes[2].set_title('Histogram for theta_chain[:, 2]')
plt.tight_layout()
for ax in axes:
    ax.legend()
plt.show()
    
    

