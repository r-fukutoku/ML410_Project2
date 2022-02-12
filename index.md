Create a new Github page with a presentation on the concepts of Locally Weighted Regression and Random Forest. 

Apply the regression methods to real data sets, such as "Cars" or "Boston Housing Data" where you consider only one input variable 
(the weight of the car for the "Cars" data set and the number of rooms for the "Boston Hausing" data). 
The output varable is the mileage for the "Cars" data set and "cmedv" or the median price for the housing data.

For each method and data set report the crossvalidated mean square error and determine which method is achieveng the better results.
In this paper you should also include theoretical considerations, examples of Python coding and plots. 
The final results should be clearly stated.

# Locally Weighted Regression & Random Forest

## Locally Weighted Regression (Abbreviated "loess" ; "lowess")
The main idea of linear regression is the assumption that:


If we pre-multiply this equation with a matrix of weights (the "weights" are on the main diagonal and the rest of the elements are 0) we get:


The distancfe between two independent observations is the Euclidean distance bw the two represented  𝑝− dimensional vectors. The equation is:


We shall have  𝑛  differrent weight vectors because we have  𝑛  different observations.

Important aspect: linear regression can be seen as a linear combination of the observed outputs (values of the dependent variable).

We have:


We solve for  𝛽  (by assuming that  𝑋𝑇𝑋  is invertible):


We take the expected value of this equation and obtain:


Therefore the predictions we make are:


The big Idea: the predictions we make are a linear combination of the actual observed values of the dependent valuable!

For locally weighted regression,  𝑦̂   is pbtained as a different linear combination of the values of y.


## Random Forest
Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.

![image](https://user-images.githubusercontent.com/98488324/153693726-36f3fe10-9648-4606-92cb-293b6c78a9dd.png)

The diagram above shows the structure of a Random Forest. You can notice that the trees run in parallel with no interaction amongst them. A Random Forest operates by constructing several decision trees during training time and outputting the mean of the classes as the prediction of all the trees.

By default, the decision trees we use here will make their predictions based on the mean value of the target within each leaf of the tree, and the splitting criteria will be based on minimizing the MSE.



## Applications with Real Data
Cars Data (output varable is the mileage): 

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
cars = pd.read_csv("drive/MyDrive/DATA410_AdvML/cars.csv")
```
<img width="234" alt="image" src="https://user-images.githubusercontent.com/98488324/153694695-0e275da1-6379-44db-af1b-92a43d0c0544.png">


#### Locally Weighted Regression:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120

import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)

def lowess_reg(x, y, xnew, kern, tau):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # IMPORTANT: we expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
    
x = cars['WGT'].values
y = cars['MPG'].values

lowess_reg(x,y,2100,tricubic,0.01)

xnew = np.arange(1500,5500,10) 
yhat = lowess_reg(x,y,xnew,tricubic,80)

plt.scatter(x,y)
plt.plot(xnew,yhat,color='red',lw=2)

```

![image](https://user-images.githubusercontent.com/98488324/153694807-2bc7665e-df35-4703-a756-44192ccd0ac5.png)


```python
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)

scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))

yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scaled,tricubic,0.1)

mse(yhat_test,ytest)
```
15.961885966790936


```python
plt.plot(np.sort(xtest_scale.ravel()),yhat_test)
```
![image](https://user-images.githubusercontent.com/98488324/153695299-b5a1f418-3757-4854-ab90-0a4184959d79.png)


#### Random Forest:
```python
rf = RandomForestRegressor(n_estimators=100,max_depth=3)
rf.fit(xtrain_scaled,ytrain)

mse(ytest,rf.predict(xtest_scale))
```
15.931305250431844

```python
yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scale.ravel(),tricubic,0.1)

dat_test = np.column_stack([xtest_scale,ytest,yhat_test])

sorted_dat_test = dat_test[np.argsort(dat_test[:,0])]

kf = KFold(n_splits=10,shuffle=True,random_state=310)
mse_lwr = []
mse_rf = []

mse_lwr = []
mse_rf = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_lwr = lowess_reg(xtrain.ravel(),ytrain,xtest.ravel(),tricubic,0.5)
  rf.fit(xtrain,ytrain)
  yhat_rf = rf.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_rf.append(mse(ytest,yhat_rf))
```
```python
np.mean(mse_lwr)
```
17.584499477691253

```python
np.mean(mse_rf)
```
18.3197148440588

 
#### Final results: 
The crossvalidated mean square error 
determine which method is achieveng the better results.





### Reference for Regressions
Bakshi, C. (Jun 8, 2020). _Medium_.
https://levelup.gitconnected.com/random-forest-regression-209c0f354c84

Sicotte, X. (May 24, 2018).
https://xavierbourretsicotte.github.io/loess.html


##### Jekyll Themes
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/r-fukutoku/Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

##### Support or Contact
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
