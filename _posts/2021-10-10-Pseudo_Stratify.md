# Pseudo Stratification for Skewed Continuous Response Variables


## High Level Idea
Most compitent datascientists understand that if you have a categorical response variable that is extremely unbalanced, you should stratify your train/test split. This ensures that your test and train set are both representative of the overall dataset. 

But what happens when you have a continuous response variable that is extremely skewed? Well in short you have the same problem, and without this trick, its quite likely you'll generates a test set not representative of the overall dataset. To be more robust we could probably use Kolmogorov-Smirnov tests, but for this we will simply compare means of the distributions, as this is often an important metric in practice.

1. TOC
{:toc} 

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
```

## Generate an uneven distribution and plot it
This function generates a gamma distribution, gamma_shape being how skewed it is, and nsamples being how large our dataset is. Gamma is a perfect distribution for this, because very skewed datasets are often estimated quite well by gamma distributions. One example where I encountered this was insurance claims for vehicles, the vast majority are small issues like windshield damage, but can also be catostrophically expensive accidents.

```python
def generate_distribution_and_plot(gamma_shape,gamma_scale,n_samples):    
    gamma_target=np.random.gamma(gamma_shape,gamma_scale,n_samples)
    X=np.random.rand(len(gamma_target))
    plt.figure(figsize=(8,5))
    plt.hist(gamma_target,bins=round(len(gamma_target)**.5))
    plt.xlabel('Sample Values',fontsize=12)
    plt.ylabel('Samples Per Bin',fontsize=12)
    plt.title('Gamma Distribution',fontsize=14)
    plt.show()
    return pd.DataFrame(data={'y':gamma_target,'X':X})

## EXAMPLE
df_example=generate_distribution_and_plot(.5,1,500)
```


![png](/images/Pseudo_Stratify_files/output_3_0.png)


## Evaluation of Train/Test Split Performance
"grab_means_aftersplit" randomly splits the dataset into train and test splits, calculates their means, and returns their distance to the overall mean, as a percentage relative to the overall mean.

"Get_avg_diff" repeats this process 50 times, to give us an estimation of how far on average our train, and test sets are from the distribution mean. Note that as test is 20% and train is 80% of the dataset, test will be much further off on average.

Ignore the stratify argument for now.

```python
def grab_means_aftersplit(df,stratify=None):
    if stratify==None:
        X_train, X_test, y_train, y_test = train_test_split(df['X'],df['y'],test_size=.2,stratify=None)
    else:
        X_train, X_test, y_train, y_test = train_test_split(df['X'],df['y'],test_size=.2,stratify=df[stratify])
    actual_mean=df['y'].mean()
    train_diff=abs(round(100-y_train.mean()/actual_mean*100,1))
    test_diff=abs(round(100-y_test.mean()/actual_mean*100,1))
    return train_diff,test_diff

def get_avg_diff(df,pseudo_stratify=None,verbose=False):
    train_diffs,test_diffs=[],[]
    for trial in range(50):
        train_diff,test_diff=grab_means_aftersplit(df,stratify=pseudo_stratify)
        train_diffs.append(train_diff)
        test_diffs.append(test_diff)
    train_mean_dev=round(np.mean(train_diffs),1)
    test_mean_dev=round(np.mean(test_diffs),1)
    if verbose==True:
        print('Absolute % Deviation From Mean (50 trials)')
        print('Train',train_mean_dev)
        print('Test',test_mean_dev)
    return train_mean_dev,test_mean_dev

##Example 
train_mean,test_mean=get_avg_diff(df_example,None,verbose=True)
```

    Absolute % Deviation From Mean (50 trials)
    Train 2.5
    Test 10.0
    

## Solution, Pseudo-Stratified Splitting!
The solution is quite simple in theory, we divide the dataset, in ascending order, into n_splits, (in this example 10), assign each sample a split group and then stratify our sampling based on this new column. 

We can see on average, the stratified split is much closer to the overall distribution.

```python
def add_splits(df_input,n_splits,split_name,verbose=False):
    sorted_indices=np.argsort(df_input.y) #This gives us list of indices that correspond to ascending values
    splits=np.array_split(sorted_indices, n_splits) #Now we split this into 10 sublists 
    #Print the means for every split and put the split number into the dataframe
    for i in range(n_splits):                    
        if verbose==True: #If we want to print intermediate info
            print('Split:',i,'Mean',round(df_input.y[splits[i]].mean(),2)) 
        df_input.loc[splits[i],split_name]=i
    return df_input

##Example
df_example=add_splits(df_example,n_splits=10,split_name='splits',verbose=True)
train_mean,test_mean=get_avg_diff(df_example,'splits',verbose=True)
```

    Split: 0 Mean 0.0
    Split: 1 Mean 0.02
    Split: 2 Mean 0.05
    Split: 3 Mean 0.11
    Split: 4 Mean 0.18
    Split: 5 Mean 0.28
    Split: 6 Mean 0.44
    Split: 7 Mean 0.72
    Split: 8 Mean 1.13
    Split: 9 Mean 2.26
    Absolute % Deviation From Mean (50 trials)
    Train 1.2
    Test 4.9
    

## Test Various Splits
Now it seems this did indeed help, the train and test samples are closer to the mean on average. But how many splits is correct, too few and we dont maximize performance, too many and we overfit with no added performance. 

To test this, we feed in an array of number of splits we would like to test, get the average performance for each of them, and plot the results. 

A good rule of thumb I've found for recommended number of splits, is to cap the samples at 10,000, (so if its 20,000 treat it as 10,000). And then take the square root of the test set. This rule is obviously problematic since its based only on sample size and not on skew which also very much effects the output, but in my experience works alright. 

```python
def test_various_splits(df,split_vals,recommended_splits=None):
    test_mean_devs,train_mean_devs=[],[]
    test_std_devs,train_std_devs=[],[]
    for n_splits in split_vals: #go through all splits that arnt 0
        if n_splits==0: 
            print('No Splits:')
            train_dev,test_dev=get_avg_diff(df,pseudo_stratify=None,verbose=True)
        elif n_splits == recommended_splits: 
            print()
            print('Recommended Split',str(recommended_splits)+':')
            train_dev,test_dev=get_avg_diff(df,pseudo_stratify='splits',verbose=True)
        else:
            df=add_splits(df,n_splits,'splits')
            train_dev,test_dev=get_avg_diff(df,pseudo_stratify='splits')
        train_mean_devs.append(train_dev)
        test_mean_devs.append(test_dev)
    plt.figure(figsize=(8,5))
    plt.plot(split_vals,train_mean_devs,label='train_mean_diff')
    plt.plot(split_vals,test_mean_devs,label='test_mean_diff')
    plt.legend()
    plt.ylabel('Absolute Difference from Mean',size=11)
    plt.xlabel('Number of Splits',size=11)
    plt.title('How many splits is too many?',size=14)
    if recommended_splits!=None:
        plt.axvline(recommended_splits,ls='--',color='grey')
        plt.text(recommended_splits+1,np.max(test_mean_devs)*.75,'Recommended Split',rotation=0)
```

## Four Test Cases
Here we test the four combinations of low sample size or medium sample size, with low skew or high skew. Lower sample sizes, and higher skew make stratified sampling more important. Low sample sizes mean more volatility in sampling, and higher skew means more individual values that have disporportionate influence on the distributions mean and variance.  
Due to the 50 trials at each split level, this method would be slow at high sample sizes, but it gives us insight into the relationship between split levels and performance, and perhaps the primitively calculated recommended splits generally does the job. 

## Extreme case for stratified sampling importance
Low samples, high skew

Stratified sampling is very important

```python
#Create a gamma distribution, sample it
n_samples=1500
gamma_shape=.1
gamma_scale=1
print('Samples:',str(n_samples)+',','Gamma Shape:',gamma_shape)
df_samples=generate_distribution_and_plot(gamma_shape,gamma_scale,n_samples)
print()
recommended_splits=round(((n_samples*.2)**.5))
#print('Recommended Split:',recommended_splits)
test_various_splits(df_samples,np.arange(0,50,1),recommended_splits=recommended_splits)
```

    Samples: 1500, Gamma Shape: 0.1
    


![png](/images/Pseudo_Stratify_files/output_12_1.png)


    
    No Splits:
    Absolute % Deviation From Mean (50 trials)
    Train 4.1
    Test 16.2
    
    Recommended Split 17:
    Absolute % Deviation From Mean (50 trials)
    Train 1.7
    Test 6.8
    


![png](/images/Pseudo_Stratify_files/output_12_3.png)


## Medium Importance of Stratified Sampling 1
Low sample size, low skew

```python
#Create a gamma distribution, sample it
n_samples=1500
gamma_shape=.5
gamma_scale=1
print('Samples:',str(n_samples)+',','Gamma Shape:',gamma_shape)
df_samples=generate_distribution_and_plot(gamma_shape,gamma_scale,n_samples)
print()
recommended_splits=round(((n_samples*.2)**.5))
test_various_splits(df_samples,np.arange(0,50,1),recommended_splits=recommended_splits)
```

    Samples: 1500, Gamma Shape: 0.5
    


![png](/images/Pseudo_Stratify_files/output_14_1.png)


    
    No Splits:
    Absolute % Deviation From Mean (50 trials)
    Train 1.5
    Test 6.1
    
    Recommended Split 17:
    Absolute % Deviation From Mean (50 trials)
    Train 0.6
    Test 2.2
    


![png](/images/Pseudo_Stratify_files/output_14_3.png)


## Medium Importance of Stratified Sampling 2
Medium Sample Size, high skew

```python
recommended_splits
np.arange(0,recommended_splits*2,3)
```




    array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
           51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87])



```python
#Create a gamma distribution, sample it
n_samples=15000
gamma_shape=.1
gamma_scale=1
print('Samples:',str(n_samples)+',','Gamma Shape:',gamma_shape)
df_samples=generate_distribution_and_plot(gamma_shape,gamma_scale,n_samples)
print()
recommended_splits=round(((np.min([n_samples,10000]))*.2)**.5)
test_various_splits(df_samples,np.arange(0,recommended_splits*3,3),recommended_splits=recommended_splits)
```

    Samples: 15000, Gamma Shape: 0.1
    


![png](/images/Pseudo_Stratify_files/output_17_1.png)


    
    No Splits:
    Absolute % Deviation From Mean (50 trials)
    Train 1.0
    Test 4.2
    
    Recommended Split 45:
    Absolute % Deviation From Mean (50 trials)
    Train 0.4
    Test 1.6
    


![png](/images/Pseudo_Stratify_files/output_17_3.png)


## Low Importance of Stratified Sampling
Medium sample size, low skew

```python
#Create a gamma distribution, sample it
n_samples=15000
gamma_shape=.5
gamma_scale=1
print('Samples:',str(n_samples)+',','Gamma Shape:',gamma_shape)
df_samples=generate_distribution_and_plot(gamma_shape,gamma_scale,n_samples)
print()
recommended_splits=round(((np.min([n_samples,10000]))*.2)**.5)
test_various_splits(df_samples,np.arange(0,recommended_splits*3,3),recommended_splits=recommended_splits)
```

    Samples: 15000, Gamma Shape: 0.5
    


![png](/images/Pseudo_Stratify_files/output_19_1.png)


    
    No Splits:
    Absolute % Deviation From Mean (50 trials)
    Train 0.4
    Test 1.8
    

## Results
| Sample Size | Gamma Shape | N Splits | Train Error No Strat | Test Error No Strat | Train Error With Strat | Test Error With Strat
| --- | --- | --- | --- | --- | --- | --- |
| 1500 | .1 | 17 | 4.1 | 16.2 | 1.7 | 6.8 |
| 1500 | .5 | 17 | 1.5 | 6.1 | 0.6 | 2.2 |
| 15000 | .1 | 45| 1.0 | 4.2 | 0.4 | 1.6 |
| 15000 | .5 | .843 | .843 | Gaussian | .5 | 1.9 |
