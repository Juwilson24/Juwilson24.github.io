```python

```


```python

```


```python

```

## Machine Learning Practice

Objective: After this assignment, you can build a pipeline that
1. Preprocesses realistic data (multiple variable types) in a pipeline that handles each variable type
1. Estimates a model using CV
1. Hypertunes a model on a CV folds within training sample
1. Finally, evaluate its performance in the test sample

Let's start by loading the data


```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# load data and split off X and y
housing = pd.read_csv('input_data2/housing_train.csv')
y = np.log(housing.v_SalePrice)
housing = housing.drop('v_SalePrice',axis=1)
```


```python
# housing
```

To ensure you can be graded accurately, we need to make the "randomness" predictable. (I.e. you should get the exact same answers every single time we run this.)

Per the recommendations in the [sk-learn documentation](https://scikit-learn.org/stable/common_pitfalls.html#general-recommendations), what that means is we need to put `random_state=rng` inside every function in this file that accepts "random_state" as an argument.



```python
# create test set for use later - notice the (random_state=rng)
rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng)
```


```python
# housing
```

## Part 1: Preprocessing the data

1. Set up a single pipeline called `preproc_pipe` to preprocess the data.
    1. For **all** numerical variables, impute missing values with SimpleImputer and scale them with StandardScaler
    1. `v_Lot_Config`: Use OneHotEncoder on it 
    1. Drop any other variables (handle this **inside** the pipeline)
1. Use this pipeline to preprocess X_train. 
    1. Describe the resulting data **with two digits.**
    1. How many columns are in this object?

_HINTS:_
- _You do NOT need to type the names of all variables. There is a lil trick to catch all the variables._
- _The first few rows of my print out look like this:_

| | count | mean | std | min  | 25%  | 50% |  75% |  max
| --- | --- | --- | ---  | ---  | --- |  --- |  --- |  ---
|  v_MS_SubClass | 1455 | 0 | 1 | -0.89 | -0.89 | -0.2 | 0.26 | 3.03
|  v_Lot_Frontage | 1455 | 0 | 1 | -2.2 | -0.43 | 0 | 0.39 | 11.07
|  v_Lot_Area  | 1455 | 0 | 1 | -1.17 | -0.39 | -0.11 | 0.19 | 20.68
| v_Overall_Qual | 1455 | 0 | 1 | -3.7 | -0.81 | -0.09 | 0.64 | 2.8


```python
number_columns = X_train.select_dtypes(np.number).columns
```


```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Pipeline for numerical variables
numer_pipe = make_pipeline(
    SimpleImputer(strategy='mean'),  # Impute missing values with mean
    StandardScaler()                 # Scale features
)

cat_pipe   = make_pipeline(OneHotEncoder())

# Combine into a single preprocessing pipeline
preproc_pipe = ColumnTransformer(
    [
        # Tuple for numerical vars: (name, pipeline, columns to apply to)
        ("num_impute_scale", numer_pipe, number_columns),
        # Add other tuples for categorical vars if needed, e.g.:
        ("cat_encode", cat_pipe, ['v_Lot_Config']),
    ],
    remainder='drop'  # Drop any columns not explicitly transformed
)
preproc_pipe
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>ColumnTransformer(transformers=[(&#x27;num_impute_scale&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer()),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1...
       &#x27;v_Bedroom_AbvGr&#x27;, &#x27;v_Kitchen_AbvGr&#x27;, &#x27;v_TotRms_AbvGrd&#x27;, &#x27;v_Fireplaces&#x27;,
       &#x27;v_Garage_Yr_Blt&#x27;, &#x27;v_Garage_Cars&#x27;, &#x27;v_Garage_Area&#x27;, &#x27;v_Wood_Deck_SF&#x27;,
       &#x27;v_Open_Porch_SF&#x27;, &#x27;v_Enclosed_Porch&#x27;, &#x27;v_3Ssn_Porch&#x27;, &#x27;v_Screen_Porch&#x27;,
       &#x27;v_Pool_Area&#x27;, &#x27;v_Misc_Val&#x27;, &#x27;v_Mo_Sold&#x27;, &#x27;v_Yr_Sold&#x27;],
      dtype=&#x27;object&#x27;)),
                                (&#x27;cat_encode&#x27;,
                                 Pipeline(steps=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder())]),
                                 [&#x27;v_Lot_Config&#x27;])])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;num_impute_scale&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer()),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1...
       &#x27;v_Bedroom_AbvGr&#x27;, &#x27;v_Kitchen_AbvGr&#x27;, &#x27;v_TotRms_AbvGrd&#x27;, &#x27;v_Fireplaces&#x27;,
       &#x27;v_Garage_Yr_Blt&#x27;, &#x27;v_Garage_Cars&#x27;, &#x27;v_Garage_Area&#x27;, &#x27;v_Wood_Deck_SF&#x27;,
       &#x27;v_Open_Porch_SF&#x27;, &#x27;v_Enclosed_Porch&#x27;, &#x27;v_3Ssn_Porch&#x27;, &#x27;v_Screen_Porch&#x27;,
       &#x27;v_Pool_Area&#x27;, &#x27;v_Misc_Val&#x27;, &#x27;v_Mo_Sold&#x27;, &#x27;v_Yr_Sold&#x27;],
      dtype=&#x27;object&#x27;)),
                                (&#x27;cat_encode&#x27;,
                                 Pipeline(steps=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder())]),
                                 [&#x27;v_Lot_Config&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">num_impute_scale</label><div class="sk-toggleable__content"><pre>Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1st_Flr_SF&#x27;, &#x27;v_2nd_Flr_SF&#x27;, &#x27;v_Low_Qual_Fin_SF&#x27;, &#x27;v_Gr_Liv_Area&#x27;,
       &#x27;v_Bsmt_Full_Bath&#x27;, &#x27;v_Bsmt_Half_Bath&#x27;, &#x27;v_Full_Bath&#x27;, &#x27;v_Half_Bath&#x27;,
       &#x27;v_Bedroom_AbvGr&#x27;, &#x27;v_Kitchen_AbvGr&#x27;, &#x27;v_TotRms_AbvGrd&#x27;, &#x27;v_Fireplaces&#x27;,
       &#x27;v_Garage_Yr_Blt&#x27;, &#x27;v_Garage_Cars&#x27;, &#x27;v_Garage_Area&#x27;, &#x27;v_Wood_Deck_SF&#x27;,
       &#x27;v_Open_Porch_SF&#x27;, &#x27;v_Enclosed_Porch&#x27;, &#x27;v_3Ssn_Porch&#x27;, &#x27;v_Screen_Porch&#x27;,
       &#x27;v_Pool_Area&#x27;, &#x27;v_Misc_Val&#x27;, &#x27;v_Mo_Sold&#x27;, &#x27;v_Yr_Sold&#x27;],
      dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">cat_encode</label><div class="sk-toggleable__content"><pre>[&#x27;v_Lot_Config&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
from df_after_transform import df_after_transform

preproc_df = df_after_transform(preproc_pipe,X_train)
print(f'There are {preproc_df.shape[1]} columns in the preprocessed data.')
preproc_df.describe().T.round(2)
```

    There are 41 columns in the preprocessed data.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>v_MS_SubClass</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.89</td>
      <td>-0.89</td>
      <td>-0.20</td>
      <td>0.26</td>
      <td>3.03</td>
    </tr>
    <tr>
      <th>v_Lot_Frontage</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-2.20</td>
      <td>-0.43</td>
      <td>0.00</td>
      <td>0.39</td>
      <td>11.07</td>
    </tr>
    <tr>
      <th>v_Lot_Area</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-1.17</td>
      <td>-0.39</td>
      <td>-0.11</td>
      <td>0.19</td>
      <td>20.68</td>
    </tr>
    <tr>
      <th>v_Overall_Qual</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-3.70</td>
      <td>-0.81</td>
      <td>-0.09</td>
      <td>0.64</td>
      <td>2.80</td>
    </tr>
    <tr>
      <th>v_Overall_Cond</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-4.30</td>
      <td>-0.53</td>
      <td>-0.53</td>
      <td>0.41</td>
      <td>3.24</td>
    </tr>
    <tr>
      <th>v_Year_Built</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-3.08</td>
      <td>-0.62</td>
      <td>0.05</td>
      <td>0.98</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>v_Year_Remod/Add</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-1.63</td>
      <td>-0.91</td>
      <td>0.43</td>
      <td>0.96</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>v_Mas_Vnr_Area</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.57</td>
      <td>-0.57</td>
      <td>-0.57</td>
      <td>0.33</td>
      <td>7.87</td>
    </tr>
    <tr>
      <th>v_BsmtFin_SF_1</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.96</td>
      <td>-0.96</td>
      <td>-0.16</td>
      <td>0.65</td>
      <td>11.20</td>
    </tr>
    <tr>
      <th>v_BsmtFin_SF_2</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.29</td>
      <td>-0.29</td>
      <td>-0.29</td>
      <td>-0.29</td>
      <td>8.29</td>
    </tr>
    <tr>
      <th>v_Bsmt_Unf_SF</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-1.28</td>
      <td>-0.77</td>
      <td>-0.23</td>
      <td>0.55</td>
      <td>3.58</td>
    </tr>
    <tr>
      <th>v_Total_Bsmt_SF</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-2.39</td>
      <td>-0.59</td>
      <td>-0.14</td>
      <td>0.55</td>
      <td>11.35</td>
    </tr>
    <tr>
      <th>v_1st_Flr_SF</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-2.07</td>
      <td>-0.68</td>
      <td>-0.19</td>
      <td>0.55</td>
      <td>9.76</td>
    </tr>
    <tr>
      <th>v_2nd_Flr_SF</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.78</td>
      <td>-0.78</td>
      <td>-0.78</td>
      <td>0.85</td>
      <td>3.98</td>
    </tr>
    <tr>
      <th>v_Low_Qual_Fin_SF</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>14.09</td>
    </tr>
    <tr>
      <th>v_Gr_Liv_Area</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-2.23</td>
      <td>-0.72</td>
      <td>-0.14</td>
      <td>0.43</td>
      <td>7.82</td>
    </tr>
    <tr>
      <th>v_Bsmt_Full_Bath</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.82</td>
      <td>-0.82</td>
      <td>-0.82</td>
      <td>1.11</td>
      <td>3.04</td>
    </tr>
    <tr>
      <th>v_Bsmt_Half_Bath</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.24</td>
      <td>-0.24</td>
      <td>-0.24</td>
      <td>-0.24</td>
      <td>7.94</td>
    </tr>
    <tr>
      <th>v_Full_Bath</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-2.84</td>
      <td>-1.03</td>
      <td>0.78</td>
      <td>0.78</td>
      <td>2.59</td>
    </tr>
    <tr>
      <th>v_Half_Bath</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.76</td>
      <td>-0.76</td>
      <td>-0.76</td>
      <td>1.25</td>
      <td>3.26</td>
    </tr>
    <tr>
      <th>v_Bedroom_AbvGr</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-3.51</td>
      <td>-1.07</td>
      <td>0.15</td>
      <td>0.15</td>
      <td>6.24</td>
    </tr>
    <tr>
      <th>v_Kitchen_AbvGr</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-5.17</td>
      <td>-0.19</td>
      <td>-0.19</td>
      <td>-0.19</td>
      <td>4.78</td>
    </tr>
    <tr>
      <th>v_TotRms_AbvGrd</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-2.83</td>
      <td>-0.93</td>
      <td>-0.30</td>
      <td>0.33</td>
      <td>5.39</td>
    </tr>
    <tr>
      <th>v_Fireplaces</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.94</td>
      <td>-0.94</td>
      <td>0.63</td>
      <td>0.63</td>
      <td>5.32</td>
    </tr>
    <tr>
      <th>v_Garage_Yr_Blt</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-3.41</td>
      <td>-0.67</td>
      <td>0.00</td>
      <td>0.97</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>v_Garage_Cars</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-2.34</td>
      <td>-1.03</td>
      <td>0.28</td>
      <td>0.28</td>
      <td>2.91</td>
    </tr>
    <tr>
      <th>v_Garage_Area</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-2.20</td>
      <td>-0.69</td>
      <td>0.01</td>
      <td>0.46</td>
      <td>4.65</td>
    </tr>
    <tr>
      <th>v_Wood_Deck_SF</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.74</td>
      <td>-0.74</td>
      <td>-0.74</td>
      <td>0.59</td>
      <td>10.54</td>
    </tr>
    <tr>
      <th>v_Open_Porch_SF</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.71</td>
      <td>-0.71</td>
      <td>-0.31</td>
      <td>0.32</td>
      <td>7.67</td>
    </tr>
    <tr>
      <th>v_Enclosed_Porch</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.36</td>
      <td>-0.36</td>
      <td>-0.36</td>
      <td>-0.36</td>
      <td>9.33</td>
    </tr>
    <tr>
      <th>v_3Ssn_Porch</th>
      <td>1455.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>19.74</td>
    </tr>
    <tr>
      <th>v_Screen_Porch</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.29</td>
      <td>-0.29</td>
      <td>-0.29</td>
      <td>-0.29</td>
      <td>9.69</td>
    </tr>
    <tr>
      <th>v_Pool_Area</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.08</td>
      <td>-0.08</td>
      <td>-0.08</td>
      <td>-0.08</td>
      <td>17.05</td>
    </tr>
    <tr>
      <th>v_Misc_Val</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>v_Mo_Sold</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-2.03</td>
      <td>-0.55</td>
      <td>-0.18</td>
      <td>0.56</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>v_Yr_Sold</th>
      <td>1455.0</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>-1.24</td>
      <td>-1.24</td>
      <td>0.00</td>
      <td>1.25</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>v_Lot_Config_Corner</th>
      <td>1455.0</td>
      <td>0.18</td>
      <td>0.38</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>v_Lot_Config_CulDSac</th>
      <td>1455.0</td>
      <td>0.06</td>
      <td>0.24</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>v_Lot_Config_FR2</th>
      <td>1455.0</td>
      <td>0.02</td>
      <td>0.15</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>v_Lot_Config_FR3</th>
      <td>1455.0</td>
      <td>0.01</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>v_Lot_Config_Inside</th>
      <td>1455.0</td>
      <td>0.73</td>
      <td>0.44</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



## Part 2: Estimating one model

_Note: A Lasso model is basically OLS, but it pushes some coefficients to zero. Read more in the `sklearn` User Guide._

1. Report the mean test score (**show 5 digits**) when you use cross validation on a Lasso Model (after using the preprocessor from Part 1) with
    - alpha = 0.3, 
    - CV uses 10 `KFold`s
    - R$^2$ scoring 
1. Now, still using CV with 10 `KFold`s and R$^2$ scoring, let's find the optimal alpha for the lasso model. You should optimize the alpha out to the exact fifth digit that yields the highest R2. 
    1. According to the CV function, what alpha leads to the highest _average_ R2 across the validation/test folds? (**Show 5 digits.**)
    1. What is the mean test score in the CV output for that alpha?  (**Show 5 digits.**)
    1. After fitting your optimal model on **all** of X_train, how many of the variables did it select? (Meaning: How many coefficients aren't zero?)
    3. After fitting your optimal model on **all** of X_train, report the 5 highest  _non-zero_ coefficients (Show the names of the variables and the value of the coefficients.)
    4. After fitting your optimal model on **all** of X_train, report the 5 lowest _non-zero_ coefficients (Show the names of the variables and the value of the coefficients.)
    5. After fitting your optimal model on **all** of X_train, now use your predicted coefficients on the test ("holdout") set! What's the R2?


```python
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

lasso_pipe = Pipeline([
    ('preprocessor', preproc_pipe),  
    ('model', Lasso(alpha=0.3))     #Alpha = 0.3
])

#10-fold cross-validation with R2 scoring
scores = cross_validate(lasso_pipe,
                       X_train, 
                       y_train,
                       cv=10,              
                       scoring='r2',        
                       return_train_score=False)  

mean_test_score = np.mean(scores['test_score'])
print(f"Mean test R² score: {mean_test_score:.5f}")
```

    Mean test R² score: 0.08666



```python
lasso_pipe = Pipeline([
    ('preprocessor', preproc_pipe),  
    ('model', Lasso(max_iter=10000)) 
])

alphas = np.logspace(-4, 2, 1000)  

# Grid search
grid_search = GridSearchCV(
    estimator=lasso_pipe,
    param_grid={'model__alpha': alphas},
    cv=10,
    scoring='r2',
    n_jobs=-1  
)

# Fit the grid search
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['model__alpha']
best_score = grid_search.best_score_

print(f"Alpha: {best_alpha:.5f}")
print(f"Mean test R² score at optimal alpha: {best_score:.5f}")
```

    Alpha: 0.00769
    Mean test R² score at optimal alpha: 0.83108



```python
#Make and fit pipeline
optimal_lasso = Pipeline([
    ('preprocessor', preproc_pipe), 
    ('model', Lasso(alpha=best_alpha, max_iter=10000))  # Useb optimal alpha
])

optimal_lasso.fit(X_train, y_train)
# Get coefficients from Lasso
lasso_coefs = optimal_lasso.named_steps['model'].coef_

# Count non-zero coefficients
num_selected_vars = np.sum(lasso_coefs != 0)

print(f"Number of Non-Zero Variables: {num_selected_vars}")
```

    Number of Non-Zero Variables: 21



```python
# Optimal model from the pipeline
optimal_model = optimal_lasso.named_steps['model']
feature_names = optimal_lasso.named_steps['preprocessor'].get_feature_names_out()

# Df with the coefficients
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': optimal_model.coef_
})

# Filter for top 5 non-zero coefficients 
non_zero_coefs = coef_df[coef_df['coefficient'] != 0].copy()
non_zero_coefs['abs_coef'] = non_zero_coefs['coefficient'].abs()
top_5 = non_zero_coefs.sort_values('abs_coef', ascending=False).head(5)

print("5 highest non-zero coefficients:")
top_5.apply(lambda row: print(f"{row['feature']}: {row['coefficient']:.5f}"), axis=1)
```

    5 highest non-zero coefficients:
    num_impute_scale__v_Overall_Qual: 0.13445
    num_impute_scale__v_Gr_Liv_Area: 0.09828
    num_impute_scale__v_Year_Built: 0.06629
    num_impute_scale__v_Garage_Cars: 0.04761
    num_impute_scale__v_Overall_Cond: 0.03576





    3     None
    15    None
    5     None
    25    None
    4     None
    dtype: object




```python
# Filter for bottom 5 non-zero coefficients 
bottom_5 = (coef_df[coef_df['coefficient'] != 0]
            .assign(abs_coef=lambda df: df['coefficient'].abs())
            .nsmallest(5, 'abs_coef'))

print("5 smallest non-zero coefficients:")
bottom_5.apply(lambda row: print(f"{row['feature']}: {row['coefficient']:.5f}"), axis=1)
```

    5 smallest non-zero coefficients:
    num_impute_scale__v_Pool_Area: -0.00246
    num_impute_scale__v_Bedroom_AbvGr: 0.00395
    num_impute_scale__v_Kitchen_AbvGr: -0.00403
    num_impute_scale__v_BsmtFin_SF_1: 0.00427
    num_impute_scale__v_1st_Flr_SF: 0.00507





    32    None
    20    None
    21    None
    8     None
    12    None
    dtype: object




```python
best_model = grid_search.best_estimator_

# Refit the data
best_model.fit(X_train, y_train)

test_r2 = best_model.score(X_test, y_test)
print(f"R² score: {test_r2:.5f}")

```

    R² score: 0.86548


## Part 3: Optimizing and estimating your own model

You can walk! Let's try to run! The next skill level is trying more models and picking your favorite. 

Read this whole section before starting!  

1. Output 1: Build a pipeline with these 3 steps and **display the pipeline** 
    1. step 1: preprocessing: possible preprocessing things you can try include imputation, scaling numerics, outlier handling, encoding categoricals, and feature creation (polynomial transformations / interactions) 
    1. step 2: feature "selection": [Either selectKbest, RFEcv, or PCA](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)
    1. step 3: model estimation: [e.g. a linear model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) or an [ensemble model like HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
1. Pick two hyperparameters to optimize: Both of the hyperparameters you optimize should be **numeric** in nature (i.e. not median vs mean in the imputation step). Pick parameters you think will matter the most to improve predictions. 
    - Put those parameters into a grid search and run it. 
1. Output 2: Describe what each of the two parameters you chose is/does, and why you thought it was important/useful to optimize it.
1. Output 3: Plot the average (on the y-axis) and STD (on the x-axis) of the CV test scores from your grid search for 25+ models you've considered. Highlight in red the dot corresponding to the model **you** prefer from this set of options, and **in the figure somewhere,** list the parameters that red dot's model uses. 
    - Your plot should show at least 25 _**total**_ combinations.
    - You'll try far more than 25 combinations to find your preferred model. You don't need to report them all.
1. Output 4: Tell us the set of possible values for each parameter that were reported in the last figure.
    - For example: "Param 1 could be 0.1, 0.2, 0.3, 0.4, and 0.5. Param 2 could be 0.1, 0.2, 0.3, 0.4, and 0.5." Note: Use the name of the parameter in your write up, don't call it "Param 1".
    - Adjust your gridsearch as needed so that your preferred model doesn't use a hyperparameter whose value is the lowest or highest possible value for that parameter. Meaning: If the optimal is at the high or low end for a parameter, you've _probably_ not optimized it!
1. Output 5: Fit your pipeline on all of X_train using the optimal parameters you just found. Now use your predicted coefficients on the test ("holdout") set! **What's the R2 in the holdout sample with your optimized pipeline?**


```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import set_config

set_config(display='diagram')

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

# Preprocessing 
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),# Numeric imputation
    ('scaler', StandardScaler()) # Numeric scaling
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Categorical imputation
    ('encoder', OneHotEncoder(handle_unknown='ignore')) # Categorical encoding
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('constant_filter', VarianceThreshold(threshold=0.01)),
    ('feature_selection', SelectKBest(score_func=f_regression)),
    ('estimator', Ridge())
])

display(pipe)

# Create GridSearchCV 
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid={},  
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
#Uses cross validation
grid_search.fit(X_train, y_train)

# Results
print(f"Average CV R2: {grid_search.best_score_:.5f}")
```


<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;,...
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;))])),
                (&#x27;constant_filter&#x27;, VarianceThreshold(threshold=0.01)),
                (&#x27;feature_selection&#x27;,
                 SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)),
                (&#x27;estimator&#x27;, Ridge())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;,...
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;))])),
                (&#x27;constant_filter&#x27;, VarianceThreshold(threshold=0.01)),
                (&#x27;feature_selection&#x27;,
                 SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)),
                (&#x27;estimator&#x27;, Ridge())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1st_Flr_SF&#x27;...
       &#x27;v_Foundation&#x27;, &#x27;v_Bsmt_Qual&#x27;, &#x27;v_Bsmt_Cond&#x27;, &#x27;v_Bsmt_Exposure&#x27;,
       &#x27;v_BsmtFin_Type_1&#x27;, &#x27;v_BsmtFin_Type_2&#x27;, &#x27;v_Heating&#x27;, &#x27;v_Heating_QC&#x27;,
       &#x27;v_Central_Air&#x27;, &#x27;v_Electrical&#x27;, &#x27;v_Kitchen_Qual&#x27;, &#x27;v_Functional&#x27;,
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">num</label><div class="sk-toggleable__content"><pre>Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1st_Flr_SF&#x27;, &#x27;v_2nd_Flr_SF&#x27;, &#x27;v_Low_Qual_Fin_SF&#x27;, &#x27;v_Gr_Liv_Area&#x27;,
       &#x27;v_Bsmt_Full_Bath&#x27;, &#x27;v_Bsmt_Half_Bath&#x27;, &#x27;v_Full_Bath&#x27;, &#x27;v_Half_Bath&#x27;,
       &#x27;v_Bedroom_AbvGr&#x27;, &#x27;v_Kitchen_AbvGr&#x27;, &#x27;v_TotRms_AbvGrd&#x27;, &#x27;v_Fireplaces&#x27;,
       &#x27;v_Garage_Yr_Blt&#x27;, &#x27;v_Garage_Cars&#x27;, &#x27;v_Garage_Area&#x27;, &#x27;v_Wood_Deck_SF&#x27;,
       &#x27;v_Open_Porch_SF&#x27;, &#x27;v_Enclosed_Porch&#x27;, &#x27;v_3Ssn_Porch&#x27;, &#x27;v_Screen_Porch&#x27;,
       &#x27;v_Pool_Area&#x27;, &#x27;v_Misc_Val&#x27;, &#x27;v_Mo_Sold&#x27;, &#x27;v_Yr_Sold&#x27;],
      dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">cat</label><div class="sk-toggleable__content"><pre>Index([&#x27;parcel&#x27;, &#x27;v_MS_Zoning&#x27;, &#x27;v_Street&#x27;, &#x27;v_Alley&#x27;, &#x27;v_Lot_Shape&#x27;,
       &#x27;v_Land_Contour&#x27;, &#x27;v_Utilities&#x27;, &#x27;v_Lot_Config&#x27;, &#x27;v_Land_Slope&#x27;,
       &#x27;v_Neighborhood&#x27;, &#x27;v_Condition_1&#x27;, &#x27;v_Condition_2&#x27;, &#x27;v_Bldg_Type&#x27;,
       &#x27;v_House_Style&#x27;, &#x27;v_Roof_Style&#x27;, &#x27;v_Roof_Matl&#x27;, &#x27;v_Exterior_1st&#x27;,
       &#x27;v_Exterior_2nd&#x27;, &#x27;v_Mas_Vnr_Type&#x27;, &#x27;v_Exter_Qual&#x27;, &#x27;v_Exter_Cond&#x27;,
       &#x27;v_Foundation&#x27;, &#x27;v_Bsmt_Qual&#x27;, &#x27;v_Bsmt_Cond&#x27;, &#x27;v_Bsmt_Exposure&#x27;,
       &#x27;v_BsmtFin_Type_1&#x27;, &#x27;v_BsmtFin_Type_2&#x27;, &#x27;v_Heating&#x27;, &#x27;v_Heating_QC&#x27;,
       &#x27;v_Central_Air&#x27;, &#x27;v_Electrical&#x27;, &#x27;v_Kitchen_Qual&#x27;, &#x27;v_Functional&#x27;,
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">VarianceThreshold</label><div class="sk-toggleable__content"><pre>VarianceThreshold(threshold=0.01)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">SelectKBest</label><div class="sk-toggleable__content"><pre>SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">Ridge</label><div class="sk-toggleable__content"><pre>Ridge()</pre></div></div></div></div></div></div></div>


    Fitting 5 folds for each of 1 candidates, totalling 5 fits
    Average CV R2: 0.80706



```python
import warnings 

# parameter grid with two important numeric hyperparameters
param_grid = {
    'feature_selection__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  
    'estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] 
}

# GridSearchCV 
with warnings.catch_warnings():# Gets rid of warnings
    warnings.simplefilter("ignore")
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R2 score: {grid_search.best_score_:.5f}")
```

    Fitting 5 folds for each of 80 candidates, totalling 400 fits
    Best parameters: {'estimator__alpha': 10.0, 'feature_selection__k': 100}
    Best R2 score: 0.86210


## Parameters
1. feature_selection__k 
    - Controls which features are used by SelectKBest 
    -  Ensures that only the k features with the highest scores according to the f_regression metric are included
    - Increases training speed by reducing the amount of features
2. estimator__alpha 
    - Controls how those features are weighted
    - Prevents overfitting with large coefficients
    - Finds optimal alpha for the model


```python
import matplotlib.pyplot as plt


# Convert grid search to a df
cv_results = pd.DataFrame(grid_search.cv_results_)

# Filter for 25+ combinations
plot_results = cv_results.sort_values('mean_test_score', ascending=False).head(25)

plt.figure(figsize=(12, 8))

plt.scatter(plot_results['std_test_score'], 
            plot_results['mean_test_score'],
            alpha=0.6, s=100, label='All Models')

best_idx = plot_results['mean_test_score'].idxmax()
plt.scatter(plot_results.loc[best_idx, 'std_test_score'],
            plot_results.loc[best_idx, 'mean_test_score'],
            color='red', s=200, label='Best Model')

plt.xlabel('Standard Deviation of CV Scores', fontsize=12)
plt.ylabel('Average CV Score', fontsize=12)
plt.title('Model Performance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](output_24_0.png)
    



```python
print("Previous Paramenters:")
print(f"feature_selection__k could be: {', '.join(map(str, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))}")
print(f"estimator__alpha could be: {', '.join(map(str, [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))}")

# Adjusted parameter grids
k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Adjusts feature_selection__k
adjusted_k = k_values + [5, 15] * (grid_search.best_params_['feature_selection__k'] == min(k_values)) + [110, 120] * (grid_search.best_params_['feature_selection__k'] == max(k_values))

# Adjusts estimator__alpha
adjusted_alpha = alpha_values + [0.00001, 0.00005] * (grid_search.best_params_['estimator__alpha'] == min(alpha_values)) + [5000.0, 10000.0] * (grid_search.best_params_['estimator__alpha'] == max(alpha_values))

print("Adjusted Parameters")
print(f"feature_selection__k could be: {', '.join(map(str, adjusted_k))}")
print(f"estimator__alpha could be: {', '.join(map(str, adjusted_alpha))}")

# Update grid search
grid_search.set_params(param_grid={
    'feature_selection__k': adjusted_k,
    'estimator__alpha': adjusted_alpha
})
grid_search.fit(X_train, y_train)
```

    Previous Paramenters:
    feature_selection__k could be: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    estimator__alpha could be: 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
    Adjusted Parameters
    feature_selection__k could be: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
    estimator__alpha could be: 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
    Fitting 5 folds for each of 96 candidates, totalling 480 fits





<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                                         Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                                         (&#x27;scaler&#x27;,
                                                                                          StandardScaler())]),
                                                                         Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_Bs...
      dtype=&#x27;object&#x27;))])),
                                       (&#x27;constant_filter&#x27;,
                                        VarianceThreshold(threshold=0.01)),
                                       (&#x27;feature_selection&#x27;,
                                        SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)),
                                       (&#x27;estimator&#x27;, Ridge())]),
             n_jobs=-1,
             param_grid={&#x27;estimator__alpha&#x27;: [0.0001, 0.001, 0.01, 0.1, 1.0,
                                              10.0, 100.0, 1000.0],
                         &#x27;feature_selection__k&#x27;: [10, 20, 30, 40, 50, 60, 70,
                                                  80, 90, 100, 110, 120]},
             scoring=&#x27;r2&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                                         Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                                         (&#x27;scaler&#x27;,
                                                                                          StandardScaler())]),
                                                                         Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_Bs...
      dtype=&#x27;object&#x27;))])),
                                       (&#x27;constant_filter&#x27;,
                                        VarianceThreshold(threshold=0.01)),
                                       (&#x27;feature_selection&#x27;,
                                        SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)),
                                       (&#x27;estimator&#x27;, Ridge())]),
             n_jobs=-1,
             param_grid={&#x27;estimator__alpha&#x27;: [0.0001, 0.001, 0.01, 0.1, 1.0,
                                              10.0, 100.0, 1000.0],
                         &#x27;feature_selection__k&#x27;: [10, 20, 30, 40, 50, 60, 70,
                                                  80, 90, 100, 110, 120]},
             scoring=&#x27;r2&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;,...
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;))])),
                (&#x27;constant_filter&#x27;, VarianceThreshold(threshold=0.01)),
                (&#x27;feature_selection&#x27;,
                 SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)),
                (&#x27;estimator&#x27;, Ridge())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1st_Flr_SF&#x27;...
       &#x27;v_Foundation&#x27;, &#x27;v_Bsmt_Qual&#x27;, &#x27;v_Bsmt_Cond&#x27;, &#x27;v_Bsmt_Exposure&#x27;,
       &#x27;v_BsmtFin_Type_1&#x27;, &#x27;v_BsmtFin_Type_2&#x27;, &#x27;v_Heating&#x27;, &#x27;v_Heating_QC&#x27;,
       &#x27;v_Central_Air&#x27;, &#x27;v_Electrical&#x27;, &#x27;v_Kitchen_Qual&#x27;, &#x27;v_Functional&#x27;,
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">num</label><div class="sk-toggleable__content"><pre>Index([&#x27;v_MS_SubClass&#x27;, &#x27;v_Lot_Frontage&#x27;, &#x27;v_Lot_Area&#x27;, &#x27;v_Overall_Qual&#x27;,
       &#x27;v_Overall_Cond&#x27;, &#x27;v_Year_Built&#x27;, &#x27;v_Year_Remod/Add&#x27;, &#x27;v_Mas_Vnr_Area&#x27;,
       &#x27;v_BsmtFin_SF_1&#x27;, &#x27;v_BsmtFin_SF_2&#x27;, &#x27;v_Bsmt_Unf_SF&#x27;, &#x27;v_Total_Bsmt_SF&#x27;,
       &#x27;v_1st_Flr_SF&#x27;, &#x27;v_2nd_Flr_SF&#x27;, &#x27;v_Low_Qual_Fin_SF&#x27;, &#x27;v_Gr_Liv_Area&#x27;,
       &#x27;v_Bsmt_Full_Bath&#x27;, &#x27;v_Bsmt_Half_Bath&#x27;, &#x27;v_Full_Bath&#x27;, &#x27;v_Half_Bath&#x27;,
       &#x27;v_Bedroom_AbvGr&#x27;, &#x27;v_Kitchen_AbvGr&#x27;, &#x27;v_TotRms_AbvGrd&#x27;, &#x27;v_Fireplaces&#x27;,
       &#x27;v_Garage_Yr_Blt&#x27;, &#x27;v_Garage_Cars&#x27;, &#x27;v_Garage_Area&#x27;, &#x27;v_Wood_Deck_SF&#x27;,
       &#x27;v_Open_Porch_SF&#x27;, &#x27;v_Enclosed_Porch&#x27;, &#x27;v_3Ssn_Porch&#x27;, &#x27;v_Screen_Porch&#x27;,
       &#x27;v_Pool_Area&#x27;, &#x27;v_Misc_Val&#x27;, &#x27;v_Mo_Sold&#x27;, &#x27;v_Yr_Sold&#x27;],
      dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label sk-toggleable__label-arrow">cat</label><div class="sk-toggleable__content"><pre>Index([&#x27;parcel&#x27;, &#x27;v_MS_Zoning&#x27;, &#x27;v_Street&#x27;, &#x27;v_Alley&#x27;, &#x27;v_Lot_Shape&#x27;,
       &#x27;v_Land_Contour&#x27;, &#x27;v_Utilities&#x27;, &#x27;v_Lot_Config&#x27;, &#x27;v_Land_Slope&#x27;,
       &#x27;v_Neighborhood&#x27;, &#x27;v_Condition_1&#x27;, &#x27;v_Condition_2&#x27;, &#x27;v_Bldg_Type&#x27;,
       &#x27;v_House_Style&#x27;, &#x27;v_Roof_Style&#x27;, &#x27;v_Roof_Matl&#x27;, &#x27;v_Exterior_1st&#x27;,
       &#x27;v_Exterior_2nd&#x27;, &#x27;v_Mas_Vnr_Type&#x27;, &#x27;v_Exter_Qual&#x27;, &#x27;v_Exter_Cond&#x27;,
       &#x27;v_Foundation&#x27;, &#x27;v_Bsmt_Qual&#x27;, &#x27;v_Bsmt_Cond&#x27;, &#x27;v_Bsmt_Exposure&#x27;,
       &#x27;v_BsmtFin_Type_1&#x27;, &#x27;v_BsmtFin_Type_2&#x27;, &#x27;v_Heating&#x27;, &#x27;v_Heating_QC&#x27;,
       &#x27;v_Central_Air&#x27;, &#x27;v_Electrical&#x27;, &#x27;v_Kitchen_Qual&#x27;, &#x27;v_Functional&#x27;,
       &#x27;v_Fireplace_Qu&#x27;, &#x27;v_Garage_Type&#x27;, &#x27;v_Garage_Finish&#x27;, &#x27;v_Garage_Qual&#x27;,
       &#x27;v_Garage_Cond&#x27;, &#x27;v_Paved_Drive&#x27;, &#x27;v_Pool_QC&#x27;, &#x27;v_Fence&#x27;,
       &#x27;v_Misc_Feature&#x27;, &#x27;v_Sale_Type&#x27;, &#x27;v_Sale_Condition&#x27;],
      dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" ><label for="sk-estimator-id-25" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label sk-toggleable__label-arrow">VarianceThreshold</label><div class="sk-toggleable__content"><pre>VarianceThreshold(threshold=0.01)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label sk-toggleable__label-arrow">SelectKBest</label><div class="sk-toggleable__content"><pre>SelectKBest(score_func=&lt;function f_regression at 0x151e3b240&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label sk-toggleable__label-arrow">Ridge</label><div class="sk-toggleable__content"><pre>Ridge()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
best_model = grid_search.best_estimator_

# Refit the data
best_model.fit(X_train, y_train)

test_r2 = best_model.score(X_test, y_test)
print(f"Optimized pipeline R2 score: {test_r2:.5f}")
```

    Optimized pipeline R2 score: 0.88212

