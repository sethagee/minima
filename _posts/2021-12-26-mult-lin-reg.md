---
layout: post
title: "Multiple Linear Regression in Python"
usemathjax: true
---

In this article we build upon our work in the previous one, simple linear regression. We will give a generalization of the previous algorithms to $$n$$-dimensions, speak about standardization/normalization, and again compare our results to that of a popular library for computing linear regression. 


```python
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

%matplotlib inline
```

Since we have an idea of how linear regression works (at least in one dimension), let's motivate today's discussion with an example of what multiple linear regression might look like. When we say multiple linear regression, we simply mean allowing the number of features (columns) to be as large as we want. 

For simplicity, though, let's just add another column of data to our data from last time:


```python
# Create arrays X and y
X = np.array([
    [1, 235],
    [5.2, 436],
    [4, 301],
    [7.4, 1034],
    [9, 1679]
])

y = np.array([
    1.1,
    6.3,
    4.4,
    8,
    11.3
])

# Create 3D scatterplot of data
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter(X[:,0], X[:,1], y, color = "r")
ax.set_xlabel("X1")
ax.set_ylabel("X2") 
# show plot
plt.show()
```


    
![png](/assets/img/mult/output_4_0.png)
    


As you can see, we're not in Kansas anymore. Adding this additional column will take our space from 2-D to 3-D. Adding any more columns would have made visualizations of the above kind near impossible! As you may expect, this will also effect what our "line" of best fit will be like.

Let's use a built in library to find our parameters $$\theta$$:


```python
reg = LinearRegression().fit(X, y)
reg.coef_
```




    array([1.06774533e+00, 8.77861339e-04])




```python
reg.intercept_
```




    -0.10738894256710196



We now have two coefficients and one intercept. It should be plane to see what happens next...


```python
# Setting up equation for the plane
x_p = np.linspace(0,10,10)
y_p = np.linspace(0,1600,10)
x1, x2 = np.meshgrid(x_p,y_p)
z = reg.intercept_ + reg.coef_[0] * x1 + reg.coef_[1] * x2

# Plot plane with scatter plot
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(projection ="3d") 

ax.plot_surface(x1, x2, z)
ax.scatter(X[:,0], X[:,1], y, color = "r")
ax.set_xlabel("X1")
ax.set_ylabel("X2") 

# Better angle to see how points lie wrt plane
ax.view_init(10,90)
plt.show()

```


    
![png](/assets/img/mult/output_9_0.png)
    


It's a bird, it's a plane! Okay, enough plane jokes. As is shown above, our "line" of best fit is now a plane of best fit. Adding a second coordinate $$\theta_2$$ to each $$X$$ value added a dimension to our space, and also changed the entire set up of the problem. Indeed, our method to predict some value $$y$$ given two $$X$$-values $$x_1$$ and $$x_2$$ will be given by the hypothesis

$$ h_\theta(x_1,x_2) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 $$.

Using this equation, we can predict the $$ y $$-value given values 3 and 300 for $$x_1$$ and $$x_2$$ respectively:


```python
# Predicted value for x1 = 3 and x2 = 300
predicted = reg.intercept_ + reg.coef_[0] * 3 + reg.coef_[1] * 300

predicted
```




    3.3592054379019194



We can't stay in 3-D forever: let's generalize.

## Generalized Cost Function and Gradient Descent

As explained in the previous project on simple linear regression, when trying to fit data according to a linear hypothesis, we have a cost function that gives us a measure of the error between our "line" and the actual data. In this project, we generalize the hypothesis to include $$n$$ data features. That is, 

$$ h_\theta(x^{(i)}) = \theta_0 + \theta_1x^{(i)}_1 +\ldots +\theta_n x^{(i)}_n $$

where $$n$$ is the number of features or columns in the data set and the input $$x^{(i)}$$ is the $$i$$-th row of data. With our generalized hypothesis also comes a generalized cost function

$$ J(\theta) = \frac{\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2}{2m} $$

with the vectorized form

$$ J(\theta) = \frac{1}{m} (X\theta - y)^T (X\theta - y). $$

Notice that the vectorized form has not changed at all! That means that we should be able to use our function from before and it will generalize to $$n$$-dimensions:


```python
def linearCost(X, y, theta):
    """
    INPUTS:
    -------
    X : array_like
        Inputs. Number of rows is m and number of columns is n+1 (incl. bias term).
        Shape (m, n+1)
    y : array_like
        Outputs at each data point x. 
        Shape (m, )
    theta : array_like
        Parameters for linear regression. 
        Shape (n+1, )
    OUTPUT:
    -------
    J : float
        The value of the cost function
    """
    # Get size of data and initialize the cost function J
    m = y.size
    J = 0
    
    # Use the vectorized implementation described above to get J
    J = (X @ theta - y).T @ (X @ theta - y) / (2*m)
    
    return J
```

For the gradient descent algorithm we need only make another slight change. When updating the value of $$\theta$$, we need to make $$n$$ updates. That is, 

Update according to the rules 

$$ \theta_j := \theta_j - \alpha \frac{\partial }{\partial \theta_j}J(\theta) \\
    j = 0, 1, \ldots, n $$ 
	
until convergence. In this case, we have 

$$\frac{\partial }{\partial \theta_j}J(\theta) = \frac{1}{m} \sum_i^m (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}_j, $$

but we will luckily not need to know these specifics. We can again use our vectorized form $$X^T(X\theta - y)$$, and so the algorithm is given by:


```python
def gradientDescent(X, y, theta, alpha, num_iters):
    """
    INPUTS:
    -------
    X : array_like
        Inputs. Number of rows is m and number of columns is n+1 (incl. bias term).
        Shape (m, n+1)
    y : array_like
        Outputs at each data point x. 
        Shape (m, )
    theta : array_like
        Parameters for linear regression. 
        Shape (n+1, )
    alpha : float
        Learning rate
    num_iters : int
        Number of iterations to run gradient descent
    OUTPUT:
    -------
    theta : array_like
        The optimal parameters as found by gradient descent
    cost_history : array_like
        Array of with iterations in column 0 and cost function value at iteration i in column 1
        Shape (num_iters, 2)
    """
    # Find size of data set and initialize history list
    m = y.size
    cost_history = np.zeros([num_iters, 2])
    
    for i in range(num_iters):
        theta += (-1 * alpha) * X.T @ (X @ theta - y)
        cost_history[i, 0] = i
        cost_history[i, 1] = linearCost(X, y, theta)
    
    return theta, cost_history
```

Let's go ahead and test out the functions and compare them with the results from the beginning.


```python
# Add bias
X = np.concatenate([np.ones((5,1)), X], axis=1)
```


```python
# Set theta, alpha, num iters, and initialize history
theta = np.array([0.,0.,0.])
alpha = 0.01
num_iters = 100
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)
```

NOTE: Overflow occurs at this point.

```python
theta
```




    array([nan, nan, nan])




```python
plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/mult/output_22_1.png)
    


Um... what just happened? Let's tune the inputs a bit (a lot):


```python
# Set theta, alpha, num iters, and initialize history
theta = np.array([0.,0.,0.])
alpha = 0.000000455
num_iters = 100
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)
```


```python
theta
```




    array([0.00016822, 0.00070681, 0.007468  ])




```python
plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/mult/output_26_1.png)
    



```python
# Predicted value when x1 = 3, x2 = 300
predicted = theta[0] + theta[1] * 3 + theta[2] * 300

predicted
```




    2.2426892129614004



This graph at a glance appears to be better, but the predicted value appears to be quite off.

What went wrong? Let's explore a bit.

Since we have such a small data set and only two features, it may be worthwile to do our gradient descent calculations "by hand" to see what is happening in each step.


```python
# Set theta and alpha
theta = np.array([0.,0.,0.])
alpha = 0.01

alpha * X.T @ (X @ theta - y)
```




    array([-3.11000e-01, -2.12360e+00, -3.15744e+02])



The above shows exactly what happens in step 1 of gradient descent. The output above tells us where to go from the intitial point (0, 0, 0). Notice that we take a small step of less than a unit in coordinate 1, but a huge step of over 300 units in coordinate 3! Let's continue this experiment a bit further.


```python
theta = theta - alpha * X.T @ (X @ theta - y)

alpha * X.T @ (X @ theta - y)
```




    array([1.16354358e+04, 8.35754252e+04, 1.33376601e+07])



Already, we can see that the step size has blown up far beyond what should have happened. We chose our $$\alpha$$ to be too large. Let's explore the second chosen $$\alpha$$ and see what happens. Remember, any value of $$\alpha$$ very much larger than this will cause divergence:


```python
# Same set up but with smaller alpha
theta = np.array([0.,0.,0.])
alpha = 0.000000455

alpha * X.T @ (X @ theta - y)
```




    array([-1.4150500e-05, -9.6623800e-05, -1.4366352e-02])



We begin by taking much smaller steps, which should keep us from diverging. Since this step size is so small, let's jump a few iterations and see what happens:


```python
# 1000 iterations
for i in range(1000):
    theta = theta - alpha * X.T @ (X @ theta - y)
    
print(f'Step size: {alpha * X.T @ (X @ theta - y)}')
print(f'Theta: {theta}')
print(f'Predicted value: {theta[0] + theta[1] * 3 + theta[2] * 300:.2f}')
```

    Step size: [-1.60577453e-06 -6.55774643e-06  4.24925502e-08]
    Theta: [0.0016182  0.00662674 0.00743188]
    Predicted value: 2.25
    

After 1000 iterations, it appears that our steps have become more of a crawl. However, these values of $$\theta$$ will not give us a good predicted value, as we saw before. That probably means we still have not converged to the right value. Let's do another 10,000 iterations:


```python
# 10,000 additional iterations
for i in range(10000):
    theta = theta - alpha * X.T @ (X @ theta - y)
    
print(f'Step size: {alpha * X.T @ (X @ theta - y)}')
print(f'Theta: {theta}')
print(f'Predicted value: {theta[0] + theta[1] * 3 + theta[2] * 300:.2f}')
```

    Step size: [-1.49193617e-06 -6.13069642e-06  3.97172882e-08]
    Theta: [0.01710036 0.07004508 0.00702098]
    Predicted value: 2.33
    

Again, the steps are smaller, but $$\theta$$ and the predicted value continues to improve. What if we go another order of magnitude up and do 100,000 more iterations:


```python
# 100,000 additional iterations
for i in range(100000):
    theta = theta - alpha * X.T @ (X @ theta - y)
    
print(f'Step size: {alpha * X.T @ (X @ theta - y)}')
print(f'Theta: {theta}')
print(f'Predicted value: {theta[0] + theta[1] * 3 + theta[2] * 300:.2f}')
```

    Step size: [-6.94121011e-07 -3.13176657e-06  2.02296002e-08]
    Theta: [0.1219074  0.51638151 0.00413275]
    Predicted value: 2.91
    

It seems that we have only the appearance of convergence. In fact, the plot of the value of $$J$$ against the number of iterations was misleading. If we were to plot 10,000 iterations and zoom in, we see a linear decrease in the cost function value. The predicted value continues to improve, and so we must have chosen our value $$\alpha$$ too small. 

However, this value was the smallest such that the values for $$\theta$$ did not diverge. How, then, can we fix this problem? As it turns out, the problem stems from our values in the first and second column of $$X$$ being very different in their sizes. We can take care of this using standardization.

## Standardization

From elementary statistics, we know that any random variable $$X \sim N(\mu,\sigma^2)$$ under the transformation

$$ Z = \frac{X - \mu}{\sigma} $$

follows the distribution $$Z \sim N(0,1)$$. In this case, we now know that approximately 95% of values will lie between -2 and 2. This transformation is known as the standardization of the variable $$X$$.

Using the same logic, given some column of data, we can take the mean and standard deviation of that column and apply the same transformation to get each of the values of the column into some standard range. That is, given a column $$x^{(i)}$$, 

$$ x^{(i)}_{s} = \frac{x^{(i)} - \overline{x}_i}{s_i}, i=1,\ldots,n $$

where $$\overline{x}_i = \frac{1}{m} \sum_{j=1}^m x_j^{(i)}$$ and $$s_i^2 = \frac{1}{m-1}\sum_{j=1}^m (x_j^{(i)} - \overline{x}_i)^2$$. Let's write a function to standardize the data:


```python
# Set X and y to be the same as before

X = np.array([
    [1, 235],
    [5.2, 436],
    [4, 301],
    [7.4, 1034],
    [9, 1679]
])

y = np.array([
    1.1,
    6.3,
    4.4,
    8,
    11.3
])
```


```python
def standardizeData(X):
    """
    INPUTS:
    -------
    X : array_like
        Inputs. Number of rows is m and number of columns is n (no bias term).
        Shape (m, n)
    OUTPUT:
    -------
    X_s : array_like
        The standardized matrix X
        Shape (m, n)
    X_mean: array_like
        Means for each feature as a vector
        Shape (n, )
    X_std: array_like
        Standard deviations for each feature as a vector
        Shape (n, )
    """
    
    # Copy X into X_s
    X_s = X.copy()
    
    # Initialize the mean and standard deviation vectors
    X_mean = np.zeros(X.shape[1])
    X_std = np.zeros(X.shape[1])
    
    # Find the mean and standard deviation for each column 
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    
    # Standardize each column
    for i in range(X.shape[1]):
        X_s[:, i] = (X_s[:, i] - X_mean[i]) / X_std[i]
        
    return X_s, X_mean, X_std
```

Notice that the above function returns not only the standardized matrix $$X_s$$, but also the vector of means and standard deviations we used for each column. When we eventually make a prediction, the data will have been trained with standardized inputs and (in this case) unstandardized outputs. As a result, given any value $$x = (x_1, \ldots, x_n)$$, we will need to standardize in order to make an accurate prediction. That is, to make a prediction $$y$$, we will use the equation

$$ y = \theta_0+\theta_1\frac{(x_1 - \overline{x}_1)}{s_1}+\ldots+\theta_n \frac{(x_n - \overline{x}_n)}{s_n}. $$

Now that we have the above function, let's standardize the data and see what it looks like:


```python
# Initialize X_s, X_mean, and X_std
X_s = np.zeros(X.shape)
X_mean = np.zeros(X.shape[1])
X_std = np.zeros(X.shape[1])

# Compute X_s, X_mean, and X_std
X_s, X_mean, X_std = standardizeData(X)
```


```python
# Add bias 
X_s = np.concatenate([np.ones((5,1)), X_s], axis=1)
X_s
```




    array([[ 1.        , -1.56112403, -0.91400505],
           [ 1.        , -0.04336456, -0.54803889],
           [ 1.        , -0.47701012, -0.79383706],
           [ 1.        ,  0.75165231,  0.54075598],
           [ 1.        ,  1.32984639,  1.71512502]])



Just as expected, the values are now between -2 and 2 and are now similar in magnitude. What might the first step of gradient descent look like now with an $$\alpha$$ of 0.01, a value that quickly lead to divergence before:


```python
# Initialize theta and set alpha
theta = np.array([0.,0.,0.])
alpha = 0.01

X_s.T @ (X_s @ theta - y)
```




    array([-31.1       , -16.95120508, -15.75602696])



Recall that before, the third coordinate was 100 times as large as the other two. Now, each step is around the same size. Let's run the full gradient descent algorithm with 100 iterations to see what happens:


```python
# Set theta, alpha, num iters, and initialize history
theta = np.array([0.,0.,0.])
alpha = 0.01
num_iters = 100
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X_s, y, theta, alpha, num_iters)
```


```python
theta
```




    array([6.18317431, 2.1931107 , 1.24358633])




```python
plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/mult/output_53_1.png)
    


It looks like we may have reached convergence, although looks can be deceiving! We need to check if the predicted value lines up with the one from before of around 3.36:


```python
# Find predicted value
predicted = theta[0] + theta[1] * (3 - X_mean[0]) / X_std[0] + theta[2] * (300 - X_mean[1]) / X_std[1]

predicted
```




    3.3550418914237294



Just as we hoped, the value of the prediction is now very close to the one found using sklearn.

## Note on standardization vs. normalization

As a closing remark for multiple linear regression, I feel that other methods of standardization are worth mentioning. In some cases, normalization is preferred over standardization. This is the same as standardization, but you subtract the min instead of the mean and divide by the difference of the max and min values rather than the standard deviation. This puts values in a scale between 0 and 1.

In general, normalization may be used when the underlying distribution is not Normal, although standardization can still be used in this case. Of particular note, standardization will preserve outliers where normalization will not.

Since we only have 5 values, either choice should have worked sufficiently well for us.
