# Python for Bigquerey and SQL


```python
import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
```


```python
JSON_PATH = "path to JSON FILE from Google"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=JSON_PATH
JSON_PATH = "Path to JSON File from GA4"
client = bigquery.Client()
```

# High Level Idea
Pulling very specific GA4 data takes a bit of SQL knowledge, and intricate queries.

Writing these yourself is a headache, but with some python knowledge, its actually fairly easy.


# Getting GA4 Running

Guide to set up the GA4 API
https://developers.google.com/analytics/devguides/reporting/core/v4/quickstart/service-py

Explore a free GA4 dataset
https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset


```python
# Lets grab a sample of the data, and just a few columns for simplicity sake
```


```python
q = """
SELECT event_timestamp, event_name, event_params                                                                
FROM  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
LIMIT 15
;
"""
sample_df = client.query(q).to_dataframe()
#grab one sample
sample_df.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_timestamp</th>
      <th>event_name</th>
      <th>event_params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1605594151663973</td>
      <td>page_view</td>
      <td>[{'key': 'page_referrer', 'value': {'string_va...</td>
    </tr>
  </tbody>
</table>
</div>



This data is filled with nested data here which is an issue if we want to do data-science or analysis

What if we want specific value inside "event_params" as its own column, which is currently nested inside event_params?



```python
sample_df.iloc[0]['event_params'][:5]
```




    array([{'key': 'ga_session_number', 'value': {'string_value': None, 'int_value': 2, 'float_value': None, 'double_value': None}},
           {'key': 'ga_session_id', 'value': {'string_value': None, 'int_value': 4389558959, 'float_value': None, 'double_value': None}},
           {'key': 'clean_event', 'value': {'string_value': 'gtm.js', 'int_value': None, 'float_value': None, 'double_value': None}},
           {'key': 'page_location', 'value': {'string_value': 'https://shop.googlemerchandisestore.com/Google+Redesign/Apparel/Google+Dino+Game+Tee', 'int_value': None, 'float_value': None, 'double_value': None}},
           {'key': 'engagement_time_msec', 'value': {'string_value': None, 'int_value': 40560, 'float_value': None, 'double_value': None}}],
          dtype=object)



# Cross Join
This trick is given by google as its a common issue

Theoretically this is how it works, we join the main table every row of the nested table.
This means for this one observation there would now be 5, one with each key
Then we simply tell that we only want records with the key equal to ga_session_number


```python
#It looks like this
querey = '''
SELECT
  event_timestamp,event_name, value.int_value as ga_session_id

FROM
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS T
    CROSS JOIN
      T.event_params

WHERE
  event_params.key = 'ga_session_id'
  AND _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  LIMIT 500
  ;
'''
#Now lets look at the results
df_sample = client.query(querey).to_dataframe()
df_sample.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_timestamp</th>
      <th>event_name</th>
      <th>ga_session_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1609079909474813</td>
      <td>first_visit</td>
      <td>6267689519</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1609079913730045</td>
      <td>user_engagement</td>
      <td>6267689519</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1609079909474813</td>
      <td>session_start</td>
      <td>6267689519</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1609079909474813</td>
      <td>page_view</td>
      <td>6267689519</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1609045476565920</td>
      <td>first_visit</td>
      <td>1433811680</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1609045476565920</td>
      <td>page_view</td>
      <td>1433811680</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1609046700919403</td>
      <td>scroll</td>
      <td>1433811680</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1609045476565920</td>
      <td>session_start</td>
      <td>1433811680</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1609031351701297</td>
      <td>first_visit</td>
      <td>6998346161</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1609031351701297</td>
      <td>page_view</td>
      <td>6998346161</td>
    </tr>
  </tbody>
</table>
</div>



Look how nice this looks! We have pulled out a nested subclause as a column
Naturally the next question becomes, what about pulling out multiple of these nested columns?

Theoretically we just want to keep making new tables and join them to our previous tables 
resulting in multiple exploded columns all in one row

Lets turn this into two functions, one for generating a single querey for exploding one column out, and another for joining that to another table!


```python
def single_querey(start_date,end_date,val_type,field,sub_field):
    start_date = '\''+start_date+ '\''
    end_date = '\''+end_date+ '\''
    querey= """
    SELECT
      event_timestamp,event_name, value.{2}_value as {4}
    FROM

        `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS T
        CROSS JOIN
          T.{3}
    WHERE
      {3}.key = '{4}'
      AND _TABLE_SUFFIX BETWEEN {0} AND {1};
    """.format(start_date,end_date,val_type,field,sub_field)
    #print(querey)
    return(querey)

def join_queries(q1,q2,joined_col='T'):
    q1=q1.replace(';','')
    q2=q2.replace(';','')
    q_new = """
    SELECT T.*,{2}.{2}
     FROM 
     ( {0} ) AS T 
     LEFT JOIN 
     ( {1} ) AS {2} 
     ON T.event_timestamp = {2}.event_timestamp
     AND
     T.event_name = {2}.event_name;
    """.format(q1,q2,joined_col)
    return(q_new)
```


```python
#Now lets use these functions to grab 3 exploded out columns, page_location, page_title and ga_session_id
```


```python
start_date = '20201101'
end_date =  '20201202'
q1 = single_querey(start_date,end_date,'int','event_params','ga_session_id')
q2 = single_querey(start_date,end_date,'string','event_params','page_title')
q3 = single_querey(start_date,end_date,'string','event_params','page_location')
q_new = join_queries(q1,q2,joined_col = 'page_title')
q_new = join_queries(q_new,q3,joined_col = 'page_location')
```


```python
%%time
df = client.query(q_new).to_dataframe()
df.head()
```

    CPU times: total: 2.42 s
    Wall time: 18.7 s
    




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
      <th>event_timestamp</th>
      <th>event_name</th>
      <th>ga_session_id</th>
      <th>page_title</th>
      <th>page_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1605498662409146</td>
      <td>view_item</td>
      <td>2200271228</td>
      <td>Google Mural Socks</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1605497966809821</td>
      <td>user_engagement</td>
      <td>2200271228</td>
      <td>#IamRemarkable | Shop by Brand | Google Mercha...</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1605498605857861</td>
      <td>user_engagement</td>
      <td>2200271228</td>
      <td>Google Men's Puff Jacket Black</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1605489713558309</td>
      <td>session_start</td>
      <td>970858984</td>
      <td>Water Bottles | Drinkware | Google Merchandise...</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1605535584700554</td>
      <td>user_engagement</td>
      <td>7103403152</td>
      <td>Google Sherpa Zip Hoodie Charcoal</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
  </tbody>
</table>
</div>



Here we are able to run a very complex querey that would be a serious headache to write, automatically using python, and a pull a 
full month of data in less than half a minute.


```python
#Heres what that querey looks like
print(q_new)
```

    
        SELECT T.*,page_location.page_location
         FROM 
         ( 
        SELECT T.*,page_title.page_title
         FROM 
         ( 
        SELECT
          event_timestamp,event_name, value.int_value as ga_session_id
        FROM
    
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS T
            CROSS JOIN
              T.event_params
        WHERE
          event_params.key = 'ga_session_id'
          AND _TABLE_SUFFIX BETWEEN '20201101' AND '20201202'
         ) AS T 
         LEFT JOIN 
         ( 
        SELECT
          event_timestamp,event_name, value.string_value as page_title
        FROM
    
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS T
            CROSS JOIN
              T.event_params
        WHERE
          event_params.key = 'page_title'
          AND _TABLE_SUFFIX BETWEEN '20201101' AND '20201202'
         ) AS page_title 
         ON T.event_timestamp = page_title.event_timestamp
         AND
         T.event_name = page_title.event_name
         ) AS T 
         LEFT JOIN 
         ( 
        SELECT
          event_timestamp,event_name, value.string_value as page_location
        FROM
    
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS T
            CROSS JOIN
              T.event_params
        WHERE
          event_params.key = 'page_location'
          AND _TABLE_SUFFIX BETWEEN '20201101' AND '20201202'
         ) AS page_location 
         ON T.event_timestamp = page_location.event_timestamp
         AND
         T.event_name = page_location.event_name;
        
    
