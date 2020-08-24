### These functions are used for the Bayesian inference
### in the main jupyter notebook. Note that for all following
### implementations, clarity was preferred over computational efficiency
### or numerical stability. The functions presented here are meant
### to reflect the equations in the main text, i.e. if a certain equation
### contains a product or sum running over a set of parameters or samples,
### then we want to write a for-loop in the pyhton functions, although
### a much faster NumPy routine could do the same job a lot faster.
### Furthermore, very often we calculate products of very small probability
### values. This could lead to underflow issues and cause numerical artificats.
### To prevent these artifacts from occuring, one could use log-probabilities
### and thus transform the products into sums. In order to keep things simple
### and instructive, I did not do implement these things. But keep in mind
### they can become relevant IRL.

### Suppress warnings, this makes for a nicer appearance
### of the notebook. Most warnings are due to attempts to
### calculate log(0) in the evaluation of samplings runs.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from tqdm import tqdm

### This is the function we will use
### to generate a Fourier series
def fourier_series(X,P):
    Y  = np.zeros_like(X)
    for i in range(P.size):
        n = i+1
        Y = Y + P[i] * np.cos(n * X)
    return Y

### Compute a uniform prior on the
### interval [min(mu),max(mu)]. Note that
### this prior is not normalized.
def get_prior_uniform(P,mu):
    prior = 1.
    for i in range(P.size):
        if np.min(mu) < P[i] < np.max(mu):
            prior = prior * 1./(np.max(mu)-np.min(mu))
        else:
            prior = prior * 0.
    return prior

### Compute Inverse Gamma prior
def get_prior_IG(P,a):
    prior = 1.
    f     = stats.invgamma(a)
    for i in range(P.size):
        prior = prior * f.pdf(P[i])
    return prior

### Draw a step for the parameter vector
### P0 using a step width sigma. The step
### width is the standard deviation of a
### zero-mean normal distribution
def draw_step(P0,sigma):
    P = np.copy(P0) + np.random.normal(loc=0.,scale=sigma,size=P0.shape)
    return P

### Compute the Likelihood
def get_likelihood(P,X,D,sigma,func):
    L = 1.
    for i in range(D.size):
        f = stats.norm(D[i],sigma)
        L = L * f.pdf(func(X[i], P))
    return L

### Draw a non-deterministic parameter space
### expansion step. The subspace expansion
### is sampled from the last element of the
### parameter vector plus some noise. The noise
### is computed from a zero-mean normal distribution
### with standard deviation M.
def draw_expansion_step_nondet(P0,M):
    P = np.concatenate((P0, [P0[-1] + np.random.normal(0, M)]))
    return P

### Draw a deterministic expansion step. The
### subspace expansion is computed as M*exp(U0)
def draw_expansion_step(P0,U0,M):
    P = np.concatenate((P0, [M*np.exp(U0)]))
    return P

### Draw a contraction step
### This is always deterministic
def draw_contraction_step(P0):
    P = P0[:-1]
    return P

### Compute the proposal probability
### for the non-deterministic proposal
def get_nondet_prob(P0,M):
    f = stats.norm(P0[-2], M)
    P = f.pdf(P0[-1])
    return P

### Compute the proposal probability
### for the deterministic expansion
### proposal.
def get_expansion_prob(U0,sigma):
    f = stats.norm(0,sigma)
    P = f.pdf(U0)
    return P

### Compute the proposal probability
### for the deterministic contraction
### proposal.
def get_contraction_prob(P0,M,sigma):
    f  = stats.norm(0,sigma)
    P  = f.pdf(np.log(P0[-1]/M))
    return P

### Compute the jacobian for the
### deterministic expansion move.
def get_expansion_jacobian(U0,M):
    return M*np.exp(U0)

### Compute the jacobian for the
### deterministic contraction move.
def get_contraction_jacobian(P0):
    return 1./P0[-1]


### This is a class for carrying out
### Metropolis-Hastings sampling. Initialze
### with:
### step_width : Step with for the Markov Chain propagation
### sigma_width: Width for the Inverse Gamma distribution used for
###              sampling sigma.
### p_ref      : Reference parameter vector used for calulating prior.
### x          : x values for target function.
### y          : y values for target function.
### func       : This is the target function.
class MH_sampler:

    def __init__(self,
                 step_width,
                 sigma_width,
                 p_ref,
                 x,
                 y,
                 func):

        self.step_width  = step_width
        self.sigma_width = sigma_width
        self.p_ref       = p_ref
        self.x           = x
        self.y           = y
        self.func        = func

    ### Run the actual sampling.
    ### p0       : Initial state of parameters for sampling
    ### sigma0   : Initial state of sigma for sampling
    ### Nsteps   : Number of steps to sample
    ### se = 0.5 : Parameter used for RJ-MCMC jumping
    ### jump     : Do RJ-MCMC sampling
    ### dete     : Carry out deterministic RJ-MCMC
    def run(self,
            p0,
            sigma0,
            Nsteps,
            se = 0.5,
            jump=False,
            dete=True):

        sigma0      = np.array([sigma0])
        ### Compute the first prior, likelihood and posterior values
        prior0      = get_prior_uniform(p0, self.p_ref) * get_prior_IG(sigma0, self.sigma_width)
        likelihood0 = get_likelihood(p0, self.x, self.y, sigma0, self.func)
        post0       = prior0*likelihood0

        if jump:
            self.accept_theta   = np.zeros((Nsteps, 10))-1.
            self.reject_theta   = np.zeros((Nsteps, 10))-1.
        else:
            self.accept_theta   = np.zeros((Nsteps, p0.size))-1.
            self.reject_theta   = np.zeros((Nsteps, p0.size))-1.

        self.accept_sigma      = np.zeros(Nsteps)-1.
        self.reject_sigma      = np.zeros(Nsteps)-1.

        self.accept_posterior  = np.zeros(Nsteps)-1.
        self.accept_likelihood = np.zeros(Nsteps)-1.
        self.accept_prior      = np.zeros(Nsteps)-1.
        self.accept_dim        = np.zeros(Nsteps)-1.

        self.jump              = np.zeros(Nsteps)-1.

        jumpfactor = 1.

        ### Each model should be a < k < b.
        ### Let's just set a=0, b=11
        model_boundary = np.array([1,10])

        ### The is the main loop
        for i in tqdm(range(Nsteps)):

            ### 1.) Sampling
            ### ============

            sigma1 = draw_step(sigma0, self.step_width)
            p1     = draw_step(p0, self.step_width)
            if jump:
                ### If we jump, draw a random number U0, which
                ### will be used to decide whether we will carry
                ### out an expansion move or a contraction move.
                U0 = np.random.normal(0, self.step_width)
                ### If the length of the parameter vector is in certain bounds (i.e.
                ### if it is more than 1 and less than 10 dimensions), then the
                ### decision whether to do contraction or expansion is random (equal
                ### probability for each move).
                if 1 < p0.size < 10:
                    ### Expansion
                    if np.random.random()>0.5:
                        ### If the move would take use beyond the boundary,
                        ### we don't want to take this move. So the probability
                        ### in this case is zero.
                        if p0.size == model_boundary.max():
                            p1   = p0
                            prob = 0.
                            jac0 = 1.
                        else:
                            ### Is our move deterministic?
                            ### Calculate Jacobian! (here for the expansion move)
                            if dete:
                                p1   = draw_expansion_step(p1, U0, se)
                                prob = 1./get_expansion_prob(U0, self.step_width)
                                jaco = get_expansion_jacobian(U0, se)
                            ### No? Ok, Just calculate the proposal probability
                            else:
                                p1   = draw_expansion_step_nondet(p1, se)
                                prob = 1./get_nondet_prob(p1, se)
                                jaco = 1.
                            ### Scale the proposal probability with the
                            ### probability that such move is attempted.
                            prob /= 0.5
                    ### Contraction
                    else:
                        ### Simiarly to the expansion move, we don't want
                        ### to attempt the move if we are already on the lower
                        ### boundary.
                        if p0.size == model_boundary.min():
                            p1   = p0
                            prob = 0.
                            jac0 = 1.
                        else:
                            ### Exactly same procedure as with the expannsion move.
                            ### Just use the routines for contraction.
                            if dete:
                                p1   = draw_contraction_step(p1)
                                prob = get_contraction_prob(p0, se, self.step_width)
                                jaco = get_contraction_jacobian(p0)
                            else:
                                p1   = p1[:-1]
                                prob = 1.
                                jaco = 1.
                            prob *= 0.5
                ### At the end of the RJ-MCMC proposal, combine the jacobian and
                ### the proposal probability into a combined 'jump probability'.
                ### If we don't attempt RJ moves, this will be 1.
                jumpfactor = jaco*prob

            ### 2.) Acception / Rejection step
            ### ==============================

            ### Compute prior, likelihood and posterior for this move.
            prior1      = get_prior_uniform(p1, self.p_ref) * get_prior_IG(sigma1, self.sigma_width)
            likelihood1 = get_likelihood(p1, self.x, self.y, sigma1, self.func)
            post1       = prior1*likelihood1

            ### If the new move leads to zero posterior, we want to reject!
            if post1 == 0.:
                self.reject_theta[i][:p1.size] = p1
                self.reject_sigma[i]           = sigma1[0]
            ### If not...
            else:
                ### ... accept if the new posterior is larger
                ### then the one from the move before. If not...
                if post1*jumpfactor>post0:
                    prior0      = prior1
                    likelihood0 = likelihood1
                    post0       = post1
                    p0          = p1
                    sigma0      = sigma1
                ### ... accept the new move only if the
                ### ratio of posterior_new/posterior_old is greater
                ### than a uniformly drawn random number on the
                ### interval [0,1]
                else:
                    u   = np.random.random()
                    acc = post1/post0
                    ### Accept
                    if u<acc:
                        prior0      = prior1
                        likelihood0 = likelihood1
                        post0       = post1
                        p0          = p1
                        sigma0      = sigma1
                    ### Reject
                    else:
                        self.reject_theta[i][:p1.size] = p1
                        self.reject_sigma[i]           = sigma1[0]

            self.accept_theta[i][:p0.size] = p0
            self.accept_sigma[i]           = sigma0[0]

            self.accept_dim[i]        = p0.size
            self.accept_posterior[i]  = post0
            self.accept_likelihood[i] = likelihood0
            self.accept_prior[i]      = prior0

            self.jump[i]              = jumpfactor
