---
layout: post
usemathjax: true
title: "MCMC in Practice - Introduction to Markov Chain Monte Carlo"
---

![png](/assets/img/mh-algo.png)

# Introduction to the problem

When speaking about the probability of some event, we are really speaking about integration over some probability distribution. In most applications, we are interested also in computing expectations of functions of random variables (to find the mean, variance, etc.). 
While integrating in this fashion is possible in some cases, high dimensional distributions can lead to difficult or impossible to compute integrals.

Given such a distribution, one may take samples from the distribution, then compute the average value to approximate the expectation. This process of sampling then calculating
some average to approximate the value of an integral of interest is known as *Monte Carlo* integration. So long as our samples are independently drawn, we can be guaranteed a good estimate by laws of large numbers.

However, sampling independently is not feasible in all cases. To remedy this issue, we may construct some dependent sampling process that eventually will provide us with samples from our
desired distribution, given that it is run long enough. These dependent samples will be a *Markov chain*, and the fact that their distribution eventually comes from the distribution of interest (or the target distribution)
will be due to the famous Metropolis-Hastings algorithm.

We now introduce the technical details necessary to explore Markov chain Monte Carlo (MCMC).

## Bayesian inference

Many applications of MCMC are found in Bayesian inference (though frequentists may have use for it sometimes, as well). What exactly does this mean?

In the frequentist approach, probabilities are seen as long run frequencies (objective). Bayesians, on the other hand,
see probability as a quantification of uncertainty about the world (subjective). In Bayesian thought, we notably allow for parameters themselves to be random variables.
Many consider Bayesian inference to be more scientific and intuitive in nature, though this is also a topic of debate.
Here is the set up for a full probability model in Bayesian style:

Let $$ D $$ and $$ \theta $$ denote the observed data and model parameters (and missing data), respectively. It is useful in some
cases to think of $$ \theta $$ as some hypothesis we have, and $$ D $$ the observed evidence about that hypothesis. We can then set up the joint
probability distribution over all random quantities:

$$ P(D,\theta) = P(D|\theta)P(\theta). $$

In the above, we refer to $$ P(\theta) $$ as the *prior* distribution and $$ P(D|\theta) $$ as the *likelihood*. 
The prior is our belief about the distribution of theta before any new evidence is observed, and the likelihood indicates
how likely the data $$ D $$ is to occur, given our prior beliefs about $$ \theta $$. Likelihood may be thought of as
the information contained within our data, and the prior may be thought of as what we know before seeing any data.

Upon observing $$ D $$, we can apply Bayes theorem to find

$$ P(\theta| D) = \frac{P(\theta)P(D|\theta)}{\int P(\theta)P(D|\theta)d\theta}, $$

known as the *posterior* distribution of $$ \theta $$. This posterior is a function of the likelihood and prior, combining
what we observed in the data with our prior knowledge to form an updated belief about $$ \theta $$. This posterior
distribution is the main focus of all Bayesian inference, and therefore of critical importance. Note that the denominator,
called the *normalizing constant*, is just the marginal density $$ P(D) $$, and therefore constant. Due to this, we have

$$ P(\theta| D) \propto P(\theta)P(D|\theta). $$

That is, the posterior distribution differs from the joint distribution by some constant factor.

Much of Bayesian inference has to do with expecations of functions of $$ \theta $$. The posterior expectation of a function
$$ f $$ is 

$$ E[f(\theta)| D] = \frac{\int f(\theta)P(\theta)P(D|\theta)d\theta}{\int P(\theta)P(D|\theta)d\theta}. $$

The required integrals in the above expression have been the source of most practical difficulties in Bayesian inference. 
Analytically, the integrations are impossible in most applications, and even numerical methods/approximations tend to be difficult,
innacurate, or impossible when dealing with a high number of dimensions. In this case, Monte Carlo integration and MCMC
can allow us to find approximatations for these integrals.

## Calculating expectations

For generality, we will set up the problem to be independent of Bayesian or frequentist interests. Let $$ X $$ be a vector of $$ k $$ random variables, with distribution $$ \pi(.) $$. For Bayesians, $$ \pi $$ will be 
a posterior distribution; for frequentists it will be a likelihood. In either case, we wish to evaluate the expectation 

$$ E[f(X)] = \frac{\int f(x)\pi(x)dx}{\int \pi(x)dx} $$

for some function of interest $$ f $$. In this case, we assume that the distribution of $$ X $$ is known up to some normalizing factor. That is,
$$ \int \pi(x)dx $$ is unknown. We will also assume that $$ X $$ is continuous, though the described methods are general. Measure theoretic 
notation could be used to accomodate all the possibilities on $$ X $$.

## Monte Carlo integration

As stated in the introduction, Monte Carlo integration will involve drawing samples $$ \{ X_t, t = 1,\ldots,n \} $$ from $$ \pi $$,
then taking the average to get the approximation

$$ E[f(X)] \approx \frac{1}{n} \sum_{t=1}^n f(X_t). $$

That is, we can approximate the population mean of $$ f(X) $$ using the sample mean above. As long as we draw the $$ X_t $$
independently, an accurate approximation is guaranteed by laws of large numbers, as $$ n $$ goes to infinity. However, this is infeasible given that the
distribution $$ \pi $$ can be very non-standard. This is not necessarily a problem, and we can sample dependently as long as
the samples are drawn throughout the possible values of $$ \pi $$ in the correct proportions.

## Markov chains

As discussed above, we need some way to get a dependent sample from $$ \pi $$ where the samples actually look like they are coming
from $$ \pi $$. In this case, we refer to $$ \pi $$ as the *target distribution*, the distribution we want to sample from.
As it turns out, under quite general conditions, we are able to create a dependent sampling process that will eventually get 
samples of exactly this type.

Suppose we generate a sequence of random variables $$ \{ X_0, X_1, X_2, \ldots \} $$, such that at each time $$ t \geq 0 $$,
the next state or value $$ X_{t+1} $$ is sampled from some distribution $$ P(X_{t+1}|X_t) $$ which only depends on the current state
$$ X_t $$. In other words, $$ X_{t+1} $$ does not depend on the history $$ \{ X_0, X_1, \ldots, X_{t-1} \} $$. Such a sequence
is called a *Markov chain* with *transition kernel* $$ P(.|.) $$. In this case, chain comes from the fact that the values
$$ \{ X_0, X_1, \ldots \} $$ are linked together by the generation process described above, and Markov comes from the "memorylessness"
property that each new state in the chain does not depend upon previous states, only the current state.
We assume that $$ P(.|.) $$ does not depend on $$ t $$, and we will say that the chain is time-homogenous.

The memorylessness property described above guarantees that, although $$ X_t $$ depends directly on the starting state $$ X_0 $$,
the chain will eventually forget its start position, and converge to some unique *stationary distribution* (under some regularity conditions),
that does not depend on the starting state or the current time $$ t $$. We will use $$ \phi(.) $$ to denote this stationary
distribution. So this means that with enough time, our sampled points will begin to look like dependent samples from
the stationary distinction $$ \phi $$.

We refer to the time necessary to achieve convergence to the stationary distribution as the *burn-in*. After, for instance,
$$ m $$ iterations, the points $$ \{ X_t, t = m+1, \ldots, n \} $$ will be dependent samples from $$ \phi $$. Using these
samples, we can then discard the burn-in values and compute the *ergodic average* 

$$ \overline{f} = \frac{1}{n-m} \sum_{t=m+1}^n f(X_t), $$

whose convergence to $$ E[f(X)] $$ is ensured by the ergodic theorem when the distribution of $$ X $$ is the stationary distribution. 
That is, $$ E[f(X)] \approx \overline{f} $$.

## Metropolis-Hastings algorithm

Now that we know how we can use a Markov chain to approximate $$ E[f(X)] $$, we need some method of constructing a Markov 
chain whose stationary distribution is $$ \pi $$. Somewhat remarkably, this construction is very straightforward due to
the algorithm of Metropolis and (later, as a generalization) Hastings. Let's give some intuition for how the algorithm will work.

### Idea behind the algorithm

As a simple example, suppose I am interested in sampling from a standard normal distribution and would like to create a sample
of values from the distribution in order to estimate the mean using MCMC. Let $$ \varphi(.) $$ denote the standard normal density. 

To begin my Markov chain, I will need to pick some
initial point, let's say $$ X_0 = 0 $$. Now, I should define some method of picking my next state $$ X_1 $$. To do this, I'll choose
a number $$ \varepsilon $$ from the uniform distribution $$ U(-0.5, 0.5) $$ and let $$ Y = X_0 + \varepsilon $$, my candidate for the next state. 
I should only be interested in candidates that are not too far below my current location, and so I will only accept the candidate
if it meets some criteria related to the ratio of heights (on the distribution) of the current 
point $$ X_0 $$ and potential next point $$ Y $$. 

To do this, sample a value $$ u $$ from $$ U(0,1) $$ and let $$ \alpha $$ be
the height of my proposed new location $$ \varphi(Y) $$ over my current location $$ \varphi(X_0) $$, where we will set $$ \alpha = 1 $$ if the ratio
exceeds 1. That is,

$$ \alpha = \min \left(1, \frac{\varphi(Y)}{\varphi(X)}\right). $$

We will then put $$ X_1 = Y $$ if $$ u \leq \alpha $$, and otherwise put $$ X_1 = X_0 $$. Continue this process for
many iterations to form the Markov chain, then use the ergodic average to approximate the mean. 

Notice that any time the candidate position is higher on the graph, our chance to
accept becomes 1. When the candidate is below the current point on the graph, the larger the distance between points,
the lower the chance of acceptance becomes. As a result, the number of times some location is visited is proportional to the 
height of the distribution at that location.

### The algorithm

Now, we will give the full Metropolis-Hastings algorithm. For each time $$ t $$, the next state $$ X_{t+1} $$ is chosen by sampling
a *candidate* point $$ Y $$ from some *proposal* distribution $$ q(.|X_t) $$. We accept the candidate point with probability $$ \alpha(X_t, Y) $$, where

$$ \alpha(X,Y) = \min \left(1, \frac{\pi(Y)q(X|Y)}{\pi(X)q(Y|X)}\right). $$

If the candidate is accepted, we put $$ X_{t+1} = Y $$, and otherwise we put $$ X_{t+1} = X_t $$. The acceptance or rejection
is based upon comparing the probability $$ \alpha(X_t,Y) $$ to some $$ u $$ chosen from $$ U(0,1) $$.

The Metropolis-Hastings algorithm:

1. Initialize $$ X_0 $$; set $$ t = 0 $$.
2. Repeat the following steps:
	1. Sample a point $$ Y $$ from $$ q( . \mid X_t) $$;
	2. Sample a $$ U(0,1) $$ random variable $$ u $$;
	3. Calculate $$ \alpha(X_t, Y) $$;
	4. Accept or reject $$ Y $$:
		1. If $$ u \leq \alpha(X_t, Y) $$ set $$ X_{t+1} = Y $$;
		2. Otherwise set $$ X_{t+1} = X_t $$.
	5. Increment $$ t $$.
	
Though simple, this algorithm has the remarkable property that the stationary distribution of the chain (under some regularity conditions)
will be $$ \pi $$, regardless of the proposal distribution $$ q(.|.) $$! Why is this the case?

The probability of moving from state $$ X_t $$ to a new state $$ Y = X_{t+1} $$ 

$$ q(X_{t+1}|X_t)\alpha(X_t,X_{t+1}). $$

The probability of rejecting all possible new candidates is given by

$$ 1 - \int q(Y |X_t)\alpha(X_t,Y)dY .$$

In total, then, the transition kernel is 

$$ P(X_{t+1}|X_t) = q(X_{t+1} |X_t)\alpha(X_t,X_{t+1}) + I(X_{t+1} = X_t)\left[1-\int q(Y|X_t)\alpha(X_t,Y)dY \right] ,$$

where $$ I $$ is the indicator function (1 if condition is satisfied, 0 otherwise).

Now, consider 

$$ \pi(X_t)q(X_{t+1} |X_t)\alpha(X_t,X_{t+1}). $$ 

Suppose $$ \alpha(X_t,X_{t+1}) = 1 $$. Then we have 

$$ \alpha(X_{t+1},X_{t}) = \frac{\pi(X_{t})q(X_{t+1}|X_{t})}{\pi(X_{t+1})q(X_{t}|X_{t+1})} $$

by definition of $$ \alpha $$. Therefore, we have the equality

$$ \pi(X_t)q(X_{t+1} |X_t) = \frac{\pi(X_{t+1})q(X_t|X_{t+1})}{\pi(X_{t+1})q(X_t|X_{t+1})}\cdot \pi(X_t)q(X_{t+1} |X_t) = \pi(X_{t+1})q(X_t|X_{t+1})\alpha(X_{t+1},X_{t}). $$

The opposite case holds in a similar manner. As a result, we have 

$$ \pi(X_t)q(X_{t+1} |X_t)\alpha(X_t,X_{t+1}) = \pi(X_{t+1})q(X_t|X_{t+1})\alpha(X_{t+1},X_{t}) $$

Then, by the form we found for the transition kernel, we have the *detailed balance equation*:

$$ \pi(X_t)P(X_{t+1}|X_{t})  = \pi(X_{t+1})P(X_t|X_{t+1}). $$

Notice that we do not need to worry about including the indicator function part of the transition kernel since it will
only be nonzero in the case that $$ X_{t+1} = X_t $$, and so this part will cancel on each side. Integrating both sides
of the detailed balance equation gives 

$$ \int \pi(X_t)P(X_{t+1}|X_{t}) dX_t  = \int \pi(X_{t+1})P(X_t|X_{t+1}) dX_t, $$

where we can then pull out the function of the future value $$ X_{t+1} $$ on the right hand side to get 

$$ \int \pi(X_t)P(X_{t+1}|X_{t}) dX_t  = \pi(X_{t+1}) \int P(X_t|X_{t+1}) dX_t, $$

which will reduce to 

$$ \int \pi(X_t)P(X_{t+1}|X_{t}) dX_t  = \pi(X_{t+1}), $$

since the integral on the right hand side evaluates to 1.

Notice that the left hand side of the equation gives the marginal distribution for $$ X_{t+1} $$ under the assumption
that $$ X_t $$ is from $$ \pi $$. Essentially, this says that once I am able to get a single sample from $$ \pi $$, all
subsequent samples will also be from $$ \pi $$. In other words, the stationary distribution is $$ \pi $$. With this, we 
have shown that the Metropolis-Hastings algorithm does indeed work (mostly) as intended. We will discuss the fact that the
memorylessness property holds in another article.

# Conclusion

In this article, we saw how to obtain a dependent sample using the Metropolis-Hastings algorithm, and proved some of its
desirable properties. In articles to come, we will explore specific implementations such as Gibbs sampler, see concrete examples
of how to use MCMC techniques, and prove further properties related to the ideas given here.

