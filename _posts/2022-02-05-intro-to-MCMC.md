---
layout: post
usemathjax: true
title: "An Introduction to Markov Chain Monte Carlo in Practice"
---

# Introduction to the problem

When speaking about the probability of some event, we are really speaking about integration over some probability distribution. While finding this integral is possible in some applications, high dimensional distributions can lead to
difficult or impossible to compute integrals. In particular, we are interested in computing expecations.

Given such a distribution, one may take samples from the distribution, then compute the average value to approximate the expectation. This process of sampling then calculating
some average to approximate the value of an integral of interest is known as *Monte Carlo* integration. So long as our samples are independently drawn, we can be guaranteed a good estimate by laws of large numbers.

However, since our distributions can be so non-standard, sampling independently is not feasible. To remedy this issue, we may construct some dependent sampling process that eventually will provide us with samples from our
desired distribution, given that it is run long enough. These dependent samples will be a *Markov chain*, and the fact that their distribution eventually comes from the distribution of interest (or the target distribution)
will be due to the famous Metropolis-Hastings algorithm.

We now introduce the technical details necessary to explore Markov chain Monte Carlo (MCMC).

## Bayesian inference

NOTE: mention measure theoretic Bayes rule

## Calculating expectations

For generality, we will set up the problem to be independent of Bayesian or frequentist interests. Let $$ X $$ be a vector of $$ k $$ random variables, with distribution $$ \pi(.) $$. For Bayesians, $$ \pi $$ will be 
a posterior distribution; for frequentists it will be a likelihood. In either case, we wish to evaluate the expectation 

$$ E[f(X)] = \frac{\int f(x)\pi(x)dx}{\int \pi(x)dx} $$

## Monte Carlo integration

## Markov chains

## Metropolis-Hastings algorithm

# Implementing MCMC