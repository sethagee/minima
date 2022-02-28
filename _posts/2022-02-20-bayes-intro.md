---
layout: post
usemathjax: true
title: "Bayesian Basics and RStan"
---

# The Bayesian View on Probability

We begin our discussion as most do, with the statement of Bayes' formula:

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)}, $$

where $$ A $$ and $$ B $$ are events, with $$ P(B) \neq 0 $$. Replacing $$ A $$ and $$ B $$ with less abstract events
gives us the object of all Bayesian inference.

Let $$ D $$ and $$ \theta $$ denote the observed data and model parameters (and missing data), respectively. We
may also find it useful to call $$ \theta $$ a hypothesis. Then 

$$ P(\theta| D) = \frac{P(\theta)P(D|\theta)}{\int P(\theta)P(D|\theta)d\theta}. $$

Given the above formula, we have the following interpretations:

- The *prior* distribution $$ P(\theta) $$ tells us the probability that $$ \theta $$ is true before any data is observed.
- The *likelihood* $$ P(D | \theta) $$ is the evidence about the hypothesis $$ \theta $$ provided by the data $$ D $$.
- The *posterior* $$ P(\theta | D) $$ tells us the probability that our hypothesis $$ \theta $$ is true after seeing the data.

In what has become an ongoing debate for many decades, we have two options on how to choose the prior when it is unkown 
(the case in most experiments). Do we pick one using some reasonable guess (Bayesian) or do we completely leave it out,
using only the likelihood (frequentist)? In the recent past, frequentist ideaology dominated, due in no small part to the
high computational effort required to evaluate integrals required in Bayesian methods.

Through the advent of powerful computing methods and easy to use Bayesian analysis software (such as Stan), Bayesian
methods have begun to be in favor yet again. This article gives a short overview of how to set up RStan and do Bayesian
inference with it. 

# Bayesian Inference by Hand

Insert example here

# Installing RStan

https://github.com/stan-dev/rstan/wiki/Configuring-C---Toolchain-for-Windows
https://github.com/Stan-dev/rstan/wiki/RStan-Getting-Started

Do one of the Stan examples

# Bayesian Linear Regression

Use something from Kaggle here. See how results compare to standard lm(). Hopefully use MCMC.

# Closing Thoughts