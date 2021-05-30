# Using linear regression to predict NBA teams win percentages based on the basketball 'four factors'




```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
```


```python
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguedashteamstats
```


```python
#create datasets for the testing --> four factors of each team

#create dataframe of april four factors of each team in current season
teamsperformance = leaguedashteamstats.LeagueDashTeamStats(season='2020-21',
                                        measure_type_detailed_defense='Four Factors').get_data_frames()[0]
teamsperformance = teamsperformance.sort_values('TEAM_ID')
teamsperformance.drop(teamsperformance.columns.difference(['W_PCT','EFG_PCT',
                                                           'FTA_RATE','TM_TOV_PCT','OREB_PCT',
                                                           'OPP_EFG_PCT',
                                                          'OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT'])
                                                                 , 1, inplace=True)
```


```python
#create training data

teamsperformance2020 = leaguedashteamstats.LeagueDashTeamStats(season='2019-20',
                                        measure_type_detailed_defense='Four Factors').get_data_frames()[0]
teamsperformance2020 = teamsperformance2020.sort_values('TEAM_ID')
teamsperformance2020.drop(teamsperformance2020.columns.difference(['W_PCT','EFG_PCT',
                                                           'FTA_RATE','TM_TOV_PCT','OREB_PCT',
                                                           'OPP_EFG_PCT',
                                                          'OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT'])
                                                                 , 1, inplace=True)

teamsperformance2019 = leaguedashteamstats.LeagueDashTeamStats(season='2018-19',
                                        measure_type_detailed_defense='Four Factors').get_data_frames()[0]
teamsperformance2019 = teamsperformance2019.sort_values('TEAM_ID')
teamsperformance2019.drop(teamsperformance2019.columns.difference(['W_PCT','EFG_PCT',
                                                           'FTA_RATE','TM_TOV_PCT','OREB_PCT',
                                                           'OPP_EFG_PCT',
                                                          'OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT'])
                                                                 , 1, inplace=True)

teamsperformance2018 = leaguedashteamstats.LeagueDashTeamStats(season='2017-18',
                                        measure_type_detailed_defense='Four Factors').get_data_frames()[0]
teamsperformance2018 = teamsperformance2018.sort_values('TEAM_ID')
teamsperformance2018.drop(teamsperformance2018.columns.difference(['W_PCT','EFG_PCT',
                                                           'FTA_RATE','TM_TOV_PCT','OREB_PCT',
                                                           'OPP_EFG_PCT',
                                                          'OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT'])
                                                                 , 1, inplace=True)

teamsperformance2017 = leaguedashteamstats.LeagueDashTeamStats(season='2016-17',
                                        measure_type_detailed_defense='Four Factors').get_data_frames()[0]
teamsperformance2017 = teamsperformance2017.sort_values('TEAM_ID')
teamsperformance2017.drop(teamsperformance2017.columns.difference(['W_PCT','EFG_PCT',
                                                           'FTA_RATE','TM_TOV_PCT','OREB_PCT',
                                                           'OPP_EFG_PCT',
                                                          'OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT'])
                                                                 , 1, inplace=True)


```


```python
#merge the two dataframes

teamsperformance_test = pd.concat([teamsperformance2019,
                                   teamsperformance2020,teamsperformance2018,teamsperformance2017], ignore_index=True)

```


```python
predict = "W_PCT"

X_train = np.array(teamsperformance_test.drop([predict],1))

Y_train = np.array(teamsperformance_test[predict])

X_test = np.array(teamsperformance.drop([predict],1))

Y_test = np.array(teamsperformance[predict])


linear = linear_model.LinearRegression()

linear.fit(X_train, Y_train)


acc = linear.score(X_test,Y_test)
print("Accuracy: ",round(acc,3))

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)
```

    Accuracy:  0.882
    Co: 
     [ 5.1450666   0.51360365 -3.96250379  1.80964059 -4.603686   -0.54025971
      3.3053348  -1.13373453]
    Intercept: 
     0.13537231640192643
    


```python
predictions = linear.predict(X_test)

for i in range(len(predictions)):
    print(round(predictions[i],3), X_test[i], Y_test[i])
```

    0.588 [0.539 0.278 0.133 0.284 0.53  0.237 0.124 0.258] 0.569
    0.556 [0.543 0.234 0.141 0.289 0.539 0.273 0.142 0.263] 0.5
    0.253 [0.508 0.261 0.157 0.28  0.556 0.235 0.145 0.271] 0.306
    0.524 [0.537 0.293 0.144 0.302 0.55  0.236 0.132 0.245] 0.431
    0.478 [0.547 0.197 0.151 0.267 0.538 0.253 0.127 0.233] 0.431
    0.586 [0.55  0.242 0.123 0.253 0.534 0.259 0.129 0.266] 0.583
    0.645 [0.557 0.219 0.136 0.292 0.545 0.258 0.143 0.249] 0.653
    0.532 [0.551 0.239 0.146 0.222 0.522 0.286 0.146 0.273] 0.542
    0.274 [0.521 0.252 0.145 0.24  0.555 0.256 0.144 0.277] 0.236
    0.689 [0.564 0.222 0.135 0.27  0.531 0.239 0.132 0.246] 0.653
    0.584 [0.536 0.271 0.152 0.269 0.526 0.237 0.152 0.252] 0.583
    0.522 [0.546 0.252 0.144 0.24  0.542 0.234 0.154 0.267] 0.556
    0.688 [0.566 0.233 0.134 0.269 0.536 0.202 0.127 0.245] 0.639
    0.35 [0.52  0.254 0.139 0.271 0.556 0.264 0.149 0.283] 0.319
    0.679 [0.575 0.258 0.134 0.252 0.531 0.241 0.125 0.274] 0.667
    0.56 [0.524 0.242 0.133 0.264 0.509 0.257 0.131 0.262] 0.569
    0.223 [0.49  0.24  0.129 0.252 0.547 0.215 0.128 0.256] 0.292
    0.522 [0.542 0.227 0.131 0.247 0.531 0.258 0.144 0.299] 0.472
    0.675 [0.541 0.293 0.143 0.277 0.521 0.262 0.154 0.263] 0.681
    0.661 [0.564 0.212 0.126 0.248 0.534 0.25  0.137 0.259] 0.708
    0.541 [0.54  0.238 0.112 0.273 0.546 0.258 0.125 0.269] 0.583
    0.441 [0.549 0.248 0.133 0.253 0.557 0.253 0.136 0.287] 0.431
    0.428 [0.517 0.243 0.113 0.243 0.541 0.23  0.131 0.267] 0.458
    0.208 [0.509 0.242 0.158 0.254 0.547 0.216 0.128 0.258] 0.306
    0.473 [0.529 0.241 0.133 0.256 0.543 0.293 0.162 0.284] 0.375
    0.767 [0.563 0.244 0.142 0.284 0.507 0.207 0.115 0.243] 0.722
    0.554 [0.528 0.232 0.131 0.285 0.535 0.249 0.146 0.264] 0.528
    0.454 [0.531 0.288 0.137 0.25  0.539 0.277 0.139 0.264] 0.472
    0.356 [0.519 0.273 0.151 0.267 0.541 0.257 0.141 0.275] 0.278
    0.428 [0.532 0.238 0.149 0.276 0.55  0.219 0.148 0.278] 0.458
    


```python
corr = teamsperformance.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col0,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col1,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col2,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col3,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col4,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col5,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col6,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col7,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col8{
            background-color:  #b40426;
            color:  #f1f1f1;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col1{
            background-color:  #e0654f;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col2{
            background-color:  #5572df;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col3,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col5,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col7,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col2,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col6,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col8,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col0,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col1,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col4{
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col4,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col4,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col5{
            background-color:  #c5d6f2;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col6{
            background-color:  #5977e3;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col8{
            background-color:  #445acc;
            color:  #f1f1f1;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col0{
            background-color:  #da5a49;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col3,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col1{
            background-color:  #6180e9;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col4,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col2{
            background-color:  #aac7fd;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col5{
            background-color:  #6c8ff1;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col6,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col0,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col8{
            background-color:  #7b9ff9;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col7{
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col8{
            background-color:  #6485ec;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col0{
            background-color:  #a5c3fe;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col1{
            background-color:  #5d7ce6;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col3{
            background-color:  #c7d7f0;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col6{
            background-color:  #7da0f9;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col7{
            background-color:  #a6c4fe;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col8{
            background-color:  #bed2f6;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col1{
            background-color:  #7093f3;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col2{
            background-color:  #bbd1f8;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col4,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col5{
            background-color:  #c1d4f4;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col5{
            background-color:  #d1dae9;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col6{
            background-color:  #536edd;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col7{
            background-color:  #c9d7f0;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col0{
            background-color:  #dedcdb;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col1{
            background-color:  #a9c6fd;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col2{
            background-color:  #abc8fd;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col3{
            background-color:  #b5cdfa;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col5{
            background-color:  #c4d5f3;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col7{
            background-color:  #5673e0;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col2{
            background-color:  #82a6fb;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col3,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col1{
            background-color:  #a3c2fe;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col4,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col2{
            background-color:  #a2c1ff;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col6{
            background-color:  #7295f4;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col7{
            background-color:  #bad0f8;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col8{
            background-color:  #e0dbd8;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col0{
            background-color:  #afcafc;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col2{
            background-color:  #88abfd;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col3{
            background-color:  #6f92f3;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col4{
            background-color:  #6a8bef;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col7,#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col8{
            background-color:  #f2cbb7;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col0{
            background-color:  #92b4fe;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col1{
            background-color:  #6788ee;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col3{
            background-color:  #d6dce4;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col4{
            background-color:  #80a3fa;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col5{
            background-color:  #e8d6cc;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col6{
            background-color:  #f1cdba;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col8{
            background-color:  #f5c0a7;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col0{
            background-color:  #7396f5;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col3{
            background-color:  #688aef;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col5{
            background-color:  #f0cdbb;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col6{
            background-color:  #e4d9d2;
            color:  #000000;
        }#T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col7{
            background-color:  #efcfbf;
            color:  #000000;
        }</style><table id="T_31a15534_bfb4_11eb_8f6f_d45d647c435b" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >W_PCT</th>        <th class="col_heading level0 col1" >EFG_PCT</th>        <th class="col_heading level0 col2" >FTA_RATE</th>        <th class="col_heading level0 col3" >TM_TOV_PCT</th>        <th class="col_heading level0 col4" >OREB_PCT</th>        <th class="col_heading level0 col5" >OPP_EFG_PCT</th>        <th class="col_heading level0 col6" >OPP_FTA_RATE</th>        <th class="col_heading level0 col7" >OPP_TOV_PCT</th>        <th class="col_heading level0 col8" >OPP_OREB_PCT</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row0" class="row_heading level0 row0" >W_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col0" class="data row0 col0" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col1" class="data row0 col1" >0.81</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col2" class="data row0 col2" >-0.15</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col3" class="data row0 col3" >-0.34</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col4" class="data row0 col4" >0.17</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col5" class="data row0 col5" >-0.68</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col6" class="data row0 col6" >-0.09</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col7" class="data row0 col7" >-0.24</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow0_col8" class="data row0 col8" >-0.38</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row1" class="row_heading level0 row1" >EFG_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col0" class="data row1 col0" >0.81</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col1" class="data row1 col1" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col2" class="data row1 col2" >-0.26</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col3" class="data row1 col3" >-0.17</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col4" class="data row1 col4" >0.05</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col5" class="data row1 col5" >-0.41</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col6" class="data row1 col6" >0.03</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col7" class="data row1 col7" >-0.21</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow1_col8" class="data row1 col8" >-0.24</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row2" class="row_heading level0 row2" >FTA_RATE</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col0" class="data row2 col0" >-0.15</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col1" class="data row2 col1" >-0.26</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col2" class="data row2 col2" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col3" class="data row2 col3" >0.23</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col4" class="data row2 col4" >0.17</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col5" class="data row2 col5" >0.02</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col6" class="data row2 col6" >0.04</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col7" class="data row2 col7" >0.16</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow2_col8" class="data row2 col8" >0.13</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row3" class="row_heading level0 row3" >TM_TOV_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col0" class="data row3 col0" >-0.34</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col1" class="data row3 col1" >-0.17</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col2" class="data row3 col2" >0.23</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col3" class="data row3 col3" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col4" class="data row3 col4" >0.15</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col5" class="data row3 col5" >0.08</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col6" class="data row3 col6" >-0.11</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col7" class="data row3 col7" >0.29</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow3_col8" class="data row3 col8" >-0.14</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row4" class="row_heading level0 row4" >OREB_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col0" class="data row4 col0" >0.17</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col1" class="data row4 col1" >0.05</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col2" class="data row4 col2" >0.17</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col3" class="data row4 col3" >0.15</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col4" class="data row4 col4" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col5" class="data row4 col5" >0.01</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col6" class="data row4 col6" >-0.21</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col7" class="data row4 col7" >-0.12</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow4_col8" class="data row4 col8" >-0.43</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row5" class="row_heading level0 row5" >OPP_EFG_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col0" class="data row5 col0" >-0.68</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col1" class="data row5 col1" >-0.41</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col2" class="data row5 col2" >0.02</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col3" class="data row5 col3" >0.08</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col4" class="data row5 col4" >0.01</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col5" class="data row5 col5" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col6" class="data row5 col6" >0.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col7" class="data row5 col7" >0.23</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow5_col8" class="data row5 col8" >0.31</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row6" class="row_heading level0 row6" >OPP_FTA_RATE</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col0" class="data row6 col0" >-0.09</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col1" class="data row6 col1" >0.03</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col2" class="data row6 col2" >0.04</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col3" class="data row6 col3" >-0.11</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col4" class="data row6 col4" >-0.21</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col5" class="data row6 col5" >0.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col6" class="data row6 col6" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col7" class="data row6 col7" >0.51</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow6_col8" class="data row6 col8" >0.43</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row7" class="row_heading level0 row7" >OPP_TOV_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col0" class="data row7 col0" >-0.24</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col1" class="data row7 col1" >-0.21</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col2" class="data row7 col2" >0.16</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col3" class="data row7 col3" >0.29</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col4" class="data row7 col4" >-0.12</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col5" class="data row7 col5" >0.23</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col6" class="data row7 col6" >0.51</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col7" class="data row7 col7" >1.00</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow7_col8" class="data row7 col8" >0.48</td>
            </tr>
            <tr>
                        <th id="T_31a15534_bfb4_11eb_8f6f_d45d647c435blevel0_row8" class="row_heading level0 row8" >OPP_OREB_PCT</th>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col0" class="data row8 col0" >-0.38</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col1" class="data row8 col1" >-0.24</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col2" class="data row8 col2" >0.13</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col3" class="data row8 col3" >-0.14</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col4" class="data row8 col4" >-0.43</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col5" class="data row8 col5" >0.31</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col6" class="data row8 col6" >0.43</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col7" class="data row8 col7" >0.48</td>
                        <td id="T_31a15534_bfb4_11eb_8f6f_d45d647c435brow8_col8" class="data row8 col8" >1.00</td>
            </tr>
    </tbody></table>




```python
teams = leaguedashteamstats.LeagueDashTeamStats(season='2020-21',
                                        measure_type_detailed_defense='Four Factors').get_data_frames()[0]
teams = teams.sort_values('TEAM_ID')
teams.drop(teams.columns.difference([
            "TEAM_NAME",'W_PCT']), 1, inplace=True)
teams = teams.reset_index(drop=True)

predictions_df = pd.DataFrame(predictions, columns = ['Predicted WIN PCT'])
predicted_wins = teams.join(predictions_df)

difference =  (predicted_wins['Predicted WIN PCT'] - predicted_wins['W_PCT'])*100

difference_df = pd.DataFrame(difference, columns = ['difference %'])

predicted_wins = predicted_wins.join(difference_df)

games_in_season = 72

predicted_wins["Predicted Ws"] = round((predicted_wins['Predicted WIN PCT'] * games_in_season),0)
predicted_wins["Predicted Ls"] = (games_in_season - round((predicted_wins['Predicted WIN PCT'] * games_in_season),0))

print(predicted_wins.sort_values('Predicted Ws',ascending=False))


predicted_wins.to_csv("LinearRegression.csv", sep='\t')
```

                     TEAM_NAME  W_PCT  Predicted WIN PCT  difference %  \
    25               Utah Jazz  0.722           0.767340      4.533993   
    12         Milwaukee Bucks  0.639           0.688272      4.927187   
    9              LA Clippers  0.653           0.688601      3.560099   
    18      Philadelphia 76ers  0.681           0.674753     -0.624746   
    14           Brooklyn Nets  0.667           0.679113      1.211300   
    19            Phoenix Suns  0.708           0.661350     -4.665039   
    6           Denver Nuggets  0.653           0.645135     -0.786475   
    0            Atlanta Hawks  0.569           0.587633      1.863282   
    5         Dallas Mavericks  0.583           0.586421      0.342134   
    10      Los Angeles Lakers  0.583           0.583937      0.093674   
    26       Memphis Grizzlies  0.528           0.553559      2.555938   
    1           Boston Celtics  0.500           0.555908      5.590753   
    15         New York Knicks  0.569           0.560249     -0.875110   
    20  Portland Trail Blazers  0.583           0.541370     -4.162990   
    3     New Orleans Pelicans  0.431           0.523680      9.268050   
    17          Indiana Pacers  0.472           0.521517      4.951698   
    7    Golden State Warriors  0.542           0.531701     -1.029908   
    11              Miami Heat  0.556           0.522416     -3.358416   
    24         Toronto Raptors  0.375           0.472532      9.753205   
    4            Chicago Bulls  0.431           0.477888      4.688822   
    27      Washington Wizards  0.472           0.453965     -1.803542   
    21        Sacramento Kings  0.431           0.441419      1.041856   
    29       Charlotte Hornets  0.458           0.427500     -3.049968   
    22       San Antonio Spurs  0.458           0.427595     -3.040496   
    28         Detroit Pistons  0.278           0.355546      7.754598   
    13  Minnesota Timberwolves  0.319           0.350257      3.125688   
    8          Houston Rockets  0.236           0.273702      3.770235   
    2      Cleveland Cavaliers  0.306           0.253124     -5.287599   
    16           Orlando Magic  0.292           0.223061     -6.893901   
    23   Oklahoma City Thunder  0.306           0.207743     -9.825659   
    
        Predicted Ws  Predicted Ls  
    25          55.0          17.0  
    12          50.0          22.0  
    9           50.0          22.0  
    18          49.0          23.0  
    14          49.0          23.0  
    19          48.0          24.0  
    6           46.0          26.0  
    0           42.0          30.0  
    5           42.0          30.0  
    10          42.0          30.0  
    26          40.0          32.0  
    1           40.0          32.0  
    15          40.0          32.0  
    20          39.0          33.0  
    3           38.0          34.0  
    17          38.0          34.0  
    7           38.0          34.0  
    11          38.0          34.0  
    24          34.0          38.0  
    4           34.0          38.0  
    27          33.0          39.0  
    21          32.0          40.0  
    29          31.0          41.0  
    22          31.0          41.0  
    28          26.0          46.0  
    13          25.0          47.0  
    8           20.0          52.0  
    2           18.0          54.0  
    16          16.0          56.0  
    23          15.0          57.0  
    


```python
import matplotlib.style as style
plt.style.use('fivethirtyeight')
plt.scatter(predicted_wins['W_PCT'],predicted_wins['Predicted WIN PCT'],c= np.sqrt(predicted_wins["difference %"] ** 2),
            cmap="coolwarm",s=200)
plt.plot([0.2, 0.8], [0.2, 0.8])

plt.xlabel("Actual Win Percentage")
plt.ylabel("Predicted Win Percentage")
fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig('FourFactors.png')
```


    
![png](FourFactorsCopy_files/FourFactorsCopy_11_0.png)
    


# Problems with the model


```python
plt.scatter(teamsperformance['TM_TOV_PCT'],teamsperformance['OPP_TOV_PCT']);
plt.show()
```


    
![png](FourFactorsCopy_files/FourFactorsCopy_13_0.png)
    



```python
plt.scatter(teamsperformance['EFG_PCT'],teamsperformance['OPP_FTA_RATE']);
plt.show()
```


    
![png](FourFactorsCopy_files/FourFactorsCopy_14_0.png)
    



```python
plt.scatter(teamsperformance['W_PCT'],teamsperformance['EFG_PCT']);
plt.show()
```


    
![png](FourFactorsCopy_files/FourFactorsCopy_15_0.png)
    


You could likely model a teams success off of their EFG% but this will most likely favour good offensive teams over good defensive teams. The four factors model allows for less effective offensive teams to be predicted more accurately since it considers their ability to stop opposition from shooting as good of a percentage or generating more possessions through offesnive rebounds and forcing turnovers.


```python

```


```python

```
