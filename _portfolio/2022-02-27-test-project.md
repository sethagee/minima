---
layout: post
title: "Test"
usemathjax: true
---

# Simple Linear Regression

In this project, we give an explanation and implementation of the regularized linear regression with gradient descent algorithm. The results are compared against those of the package from sklearn.


```python
# import libraries
import numpy as np

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

%matplotlib inline
```

## Introduction and example data

Suppose we have a list of data $$ X = \{x_1,\ldots,x_m\}$$ which have some corresponding data $$ y = \{y_1, \ldots, y_m\} $$. As a basic example, the set $$ X $$ may be the heights of $$m$$ people and the set $$y$$ their corresponding shoe sizes. Given the data, we may wonder, "Can I make a prediction about a person's shoe size given their height?" In fact, we may argue that the relationship should be roughly linear: if we plot shoe size against height, the graph should be roughly a straight line.

Mathematically, we may say that the hypothesis of a linear relationship between $$X$$ and $$y$$ should yield an equation of the form 

$$ h_{\theta}(x) = \theta_0 + \theta_1 x $$

which, when fed an input $$ x $$, should give us an approximation $$h_\theta(x)$$ for $$ y $$ based upon the parameters $$ \theta_0, \theta_1 $$. 

So, if these parameters $$ \theta_0,\theta_1 $$ are to give me an approximation for $$ y $$ based upon $$ x $$, how can I choose them?

We first give a very simple example where our hypothesis is a linear function of only two parameters.



```python
# Create some simple data X and y 
X = np.array([
    1,
    5.2,
    4,
    7.4,
    9
])

y = np.array([
    1.1,
    6.3,
    4.4,
    8,
    11.3
])

# Plot y against X 
plt.scatter(X, y)
plt.xlabel("X value")
plt.ylabel("y value")
```




    Text(0, 0.5, 'y value')




    
![png](/assets/img/output_5_1.png)
    


## What is a good line? The cost function

There are many lines that one may draw that could be a reasonable fit for the data:


```python
plt.scatter(X, y)
plt.xlabel("X value")
plt.ylabel("y value")

x = np.linspace(0, 10, 1000)
plt.plot(x, 1.2*x, 'r--')
plt.plot(x, 0.8*x + 1, 'g--')
plt.plot(x, 1.6*x - 1, 'k--')
```




    
![png](/assets/img/output_7_1.png)
    


Although we do not yet have a metric by which to measure fit of the above lines, visually, the red line appears to follow the trend of data much better than the other two. In particular, each point of data appears "close" to the line. To be specific, let's look at a comparison of the vertical distances between the points and lines by subtracting the y values from the corresponding points on the line.


```python
plt.figure(figsize=(14,10))
plt.scatter(X, y)
plt.xlabel("X value")
plt.ylabel("y value")
plt.plot(x, 1.2*x, 'r--')

# Plot values from X on the red line
line_/assets/img/output = 1.2 * X
plt.scatter(X, line_/assets/img/output, c='r')

# Find vertical distance between data and line and plot them
v_dist = line_/assets/img/output - y

for i in range(5):
    plt.vlines(X[i], y[i], line_/assets/img/output[i])
    d = round(v_dist[i], 2)
    label = f'Distance is {d}'
    plt.annotate(label, (X[i], y[i] - 0.5), textcoords='offset points', xytext=(0,10))
```


    
![png](/assets/img/output_9_0.png)
    


Compare these distances to the black dashed line from above:


```python
plt.figure(figsize=(14,10))
plt.scatter(X, y)
plt.xlabel("X value")
plt.ylabel("y value")
plt.plot(x, 0.8*x + 1, 'g--')

# Plot values from X on the red line
line_/assets/img/output = 0.8 * X + 1
plt.scatter(X, line_/assets/img/output, c='g')

# Find vertical distance between data and line and plot them
v_dist = line_/assets/img/output - y

for i in range(5):
    plt.vlines(X[i], y[i], line_/assets/img/output[i])
    d = round(v_dist[i], 2)
    label = f'Distance is {d}'
    plt.annotate(label, (X[i], y[i] - 0.5), textcoords='offset points', xytext=(0,10))
```


    
![png](/assets/img/output_11_0.png)
    


How can we then go about fitting a line to data? We need some method of minimizing these distances so that we can determine the line which gives us the collective minimum error from line to data. As it turns out, this error (or cost) function is given by 

$$ J(\theta_0, \theta_1) = \frac{\sum_{i=1}^m(\theta_0 + \theta_1 x_i - y_i)^2}{2m}$$

where $$ \theta_0$$ is our intercept of the line and $$ \theta_1$$ is the slope of our line, $$ m$$ is the number of values in our data set, and $$ x_i$$ and $$ y_i$$ are the respective inputs and /assets/img/outputs of the data set.

Notice that the above function does not give you the average error, but instead the average squared error. Taking the square of distances or errors has such advantages as ensuring distances are positive and also further penalizing large distances.

Let's do a some examples of the square error using our lines from before.


```python
# Calculate the cost for each of the lines from before
g_cost = 0
b_cost = 0
r_cost = 0

for i in range(5):
    # Green line
    g_cost += (1 + 0.8*X[i] - y[i])**2 / (2*5)
    # Black line
    b_cost += (-1 + 1.6*X[i] - y[i])**2 / (2*5)
    # Red line
    r_cost += (1.2*X[i] - y[i])**2 / (2*5)
    
print(f"Green cost = {g_cost:.2f}, Blue cost = {b_cost:.2f}, Red cost = {r_cost:.2f}")
```

    Green cost = 1.26, Blue cost = 1.48, Red cost = 0.12
    

Just as expected! The red line gives us the lowest cost function, and therefore the line of best fit (at least amongst the lines we chose at the beginning). Let's now implement the cost as a function so that we are able to call it whenever necessary. The implementation below will use some linear algebra.

Here is our set up: let $$ m$$ be the number of training examples. Since we are doing simple linear regression, we assume only one feature on our data. Then our set of data $$ X$$ is a vector of size $$ m \times 1$$ and $$ y$$ is the same size. We will also have a set of parameters $$\theta$$ which will be of size $$ 2\times 1$, one parameter for the intercept and one for the slope. 

Since we have a parameter for both the intercept and slope, but only one column of data in $$ X$$ , it is both convenient and logical to simply add another column of ones to our data $$ X$$ . 


```python
X = np.stack([np.ones(5), X], axis=1)
X
```




    array([[1. , 1. ],
           [1. , 5.2],
           [1. , 4. ],
           [1. , 7.4],
           [1. , 9. ]])



This addition of ones is called the bias. Once we have added this column on, we simply take $$ X\theta $$ to get a $$ m\times 1$$ vector whose $$ i$$ -th element is $$ \theta_0 + \theta_1x_i$$ . Then, subtracting $y$$ from $X\theta$, the resulting vector has $i$-th entry $\theta_0 + \theta_1x_i - y_i$. 

Finally, multiply the above vector by its transpose to get the desired sum. That is, take $(X\theta - y)^T (X\theta - y)$$ to get the sum $\sum_{i=1}^m (\theta_0+\theta_1x_i - y_i)^2$. The cost function can then be found by dividing this by $2m$.


```python
def simpleLinCost(X, y, theta):
    """
    INPUTS:
    -------
    X : array_like
        Inputs. Number of rows is m and number of columns is 2. Bias term must be added.
        Shape (m, 2)
    y : array_like
        /assets/img/outputs at each data point x. 
        Shape (m, )
    theta : array_like
        Parameters for linear regression. 
        Shape (2, )
    /assets/img/output:
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

Let's make sure the function works as intended. We know what the values of $J$$ should be for the green, blue, and red lines from before. Let's test to make sure everything adds up:


```python
# Set theta values for each line in a 3x3 matrix
theta_gbr = np.array([[1,0.8],
                     [-1, 1.6],
                     [0, 1.2]])

# Calculate the cost for each row of theta values
g_cost = simpleLinCost(X, y, theta_gbr[0,:])
b_cost = simpleLinCost(X, y, theta_gbr[1,:])
r_cost = simpleLinCost(X, y, theta_gbr[2,:])

print(f"Green cost = {g_cost:.2f}, Blue cost = {b_cost:.2f}, Red cost = {r_cost:.2f}")
```

    Green cost = 1.26, Blue cost = 1.48, Red cost = 0.12
    

Beautiful! It looks like everything is working just as we wanted.

## Line of best fit: The gradient descent algorithm

Now comes the next logical step in our procedure for finding the line of best fit: How do I know when I have found THE best line? In other words, how can I minimize the cost function $$ J $? The function $J$$ has two arguments, so this now becomes a 3-D optimization problem.

The gradient $$ \nabla J(\theta_0, \theta_1) $$ gives the direction of steepest ascent and is defined as 

$$$ \nabla J(\theta_0, \theta_1) = \begin{bmatrix} \frac{\partial }{\partial \theta_0}J(\theta_0, \theta_1) \\ \frac{\partial }{\partial \theta_1} J(\theta_0, \theta_1)  \end{bmatrix} $$

In this case, it is simple to calculate $$\frac{\partial }{\partial \theta_0}J(\theta_0, \theta_1) = \frac{1}{m} \sum_i (\theta_0+\theta_1x_i - y_i) $$$ and  $$\frac{\partial }{\partial \theta_1}J(\theta_0, \theta_1) = \frac{1}{m} \sum_i (\theta_0+\theta_1x_i - y_i)x_i $$

We are not interested in finding where the function is increasing most at any given point, however. We are interested in where the function is decreasing most rapidly at a given point, or the negative of the gradient, since we want to minimize the cost function $J$$ with respect to $\theta_0$$ and $\theta_1$. Using this idea, we could make our way down the surface by going in the direction of steepest descent at each point. This idea is encapsulated by the gradient descent algorithm:

Update according to the rules $$
    \theta_0 := \theta_0 - \alpha \frac{\partial }{\partial \theta_0}J(\theta_0, \theta_1) \\
    \theta_1 := \theta_1 - \alpha \frac{\partial }{\partial \theta_1}J(\theta_0, \theta_1)
$$$ until convergence.

So, given some initial parameters $\theta_0$$ and $\theta_1$, we find which direction goes down most steeply relative to that point. Then, we walk in that direction and continue the process until we have found a minimum. The constant value $\alpha$, known as the learning rate, essentially tells us how fast or slow to walk downhill.

Let's implement the above algorithm and see if we can find the optimal line to fit our data. Again, we can vectorize a portion of the algorithm by noticing that $\nabla J(\theta) = X^T(X\theta - y)$.


```python
def gradientDescent(X, y, theta, alpha, num_iters):
    """
    INPUTS:
    -------
    X : array_like
        Inputs. Number of rows is m and number of columns is 2. Bias term must be added.
        Shape (m, 2)
    y : array_like
        /assets/img/outputs at each data point x. 
        Shape (m, )
    theta : array_like
        Parameters for linear regression. 
        Shape (2, )
    alpha : float
        Learning rate
    num_iters : int
        Number of iterations to run gradient descent
    /assets/img/output:
    -------
    theta : array_like
        The optimal parameters as found by gradient descent
    cost_history : list
        List of cost function values for each iteration
    """
    # Find size of data set and initialize history list
    m = y.size
    cost_history = np.zeros([num_iters, 2])
    
    for i in range(num_iters):
        theta += (-1 * alpha) * X.T @ (X @ theta - y)
        cost_history[i, 0] = i
        cost_history[i, 1] = simpleLinCost(X, y, theta)
    
    return theta, cost_history
```

## Algorithm testing

Now that we have a method of optimizing $\theta$, let's test it out and see what kind of line it gives us.


```python
# Set theta, alpha, num iters, and initialize history
theta = np.array([1.0,1.0])
alpha = 0.01
num_iters = 100
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

plt.figure(figsize=(14,10))
plt.scatter(X[:, 1], y)
plt.xlabel("X value")
plt.ylabel("y value")
plt.plot(x, 1.2*x, 'r--')
plt.plot(x, theta[0] + theta[1]*x, '--')
```




    [<matplotlib.lines.Line2D at 0x255b36662b0>]




    
![png](/assets/img/output_23_1.png)
    


In the figure above, we plot the new line given to us from the $\theta$$ from gradient descent in blue and also the red line from our original example. As it turns out, we made a pretty good guess!

In terms of convergence, it can be helpful to view the history of the cost function to see if running more iterations may help. The plot is given below:


```python
plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/output_25_1.png)
    


Let's tweak some of those values and see what happens. First, we will increase the number of iterations from 100 to 200 and see what happens:


```python
theta = np.array([1.0,1.0])
alpha = 0.01
num_iters = 200
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/output_27_1.png)
    


Intersting. It looks like we may still need to increase the number of iterations until the line levels off. First, let's explore what happens when we change $\alpha$$ without changing number of iterations. Let's try a small increase of 0.011:


```python
theta = np.array([1.0,1.0])
alpha = 0.011
num_iters = 100
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/output_29_1.png)
    


Woah! That doesn't look right. According to the plot, instead of minimizing the cost function, we have began spinning out of control and increasing it somehow. This is a clear indication that our learning rate is too high, even though it was only an increase of 10%. Now let's reduce it just a tiny bit...


```python
theta = np.array([1.0,1.0])
alpha = 0.01094
num_iters = 100
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/output_31_1.png)
    


We've struck parabola! I just thought this was neat, however useless for our purposes. Anyways, we can decrease it bit more and increase the iterations to get a very nice plot that appears to level off around 0.100:


```python
theta = np.array([1.0,1.0])
alpha = 0.0105
num_iters = 300
cost_history = np.zeros([num_iters, 2])

theta, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

plt.figure(figsize=(14,10))
plt.scatter(cost_history[:,0],cost_history[:,1]) 
plt.xlabel("Number of iterations")
plt.ylabel("Value of cost function")
```




    Text(0, 0.5, 'Value of cost function')




    
![png](/assets/img/output_33_1.png)
    


This gives us a final line of best fit with line of best fit roughly equal to $y = -0.25 + 1.22 x$, as shown below:


```python
plt.figure(figsize=(14,10))
plt.scatter(X[:, 1], y)
plt.xlabel("X value")
plt.ylabel("y value")
plt.plot(x, theta[0] + theta[1]*x, '--')
```




    [<matplotlib.lines.Line2D at 0x255b3740a00>]




    
![png](/assets/img/output_35_1.png)
    


Now, as a final portion of the project, let's compare our results to those obtained by a popular python package for computing linear regression.


```python
reg = LinearRegression().fit(X, y)

plt.figure(figsize=(14,10))
plt.scatter(X[:, 1], y)
plt.xlabel("X value")
plt.ylabel("y value")
plt.plot(x, theta[0] + theta[1]*x, '--')
plt.plot(x, reg.intercept_ + reg.coef_[1]*x)
```




    [<matplotlib.lines.Line2D at 0x255b3b92ee0>]




    
![png](/assets/img//assets/img/output_37_1.png)
    


As you can see, the two lines almost coincide. Not too bad! With a bit of extra tuning on $$ \alpha $$ and the number of iterations, we may even be able to get them closer! 

Obviously this example has only 5 data points, but I felt that it gave a good sense of what was going on in each step of the process. In the continuation, we will explore not only more data points, but allow for an increased number of parameters as well.


```python

```
