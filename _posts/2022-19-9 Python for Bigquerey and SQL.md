## Python for Bigquerey and SQL


```python
import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
```


```python
JSON_PATH = "path to JSON FILE from Google"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=JSON_PATH
#JSON_PATH = "C:\\Users\\Asche\\Documents\\Code\\API\\maiatestcase-1ae98fa45803.json"

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
print(sample_df.head(1))
```

        event_timestamp     event_name  \
    0  1611926617966805  session_start   
    
                                            event_params  
    0  [{'key': 'ga_session_id', 'value': {'string_va...  
    

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
print(df_sample.head(10))
```

        event_timestamp       event_name  ga_session_id
    0  1611421556048084        page_view     7201624258
    1  1611421469875837  user_engagement     7201624258
    2  1611421393802658    session_start     7201624258
    3  1611421393802658        page_view     7201624258
    4  1611421398830116        page_view     7201624258
    5  1611421393802658      first_visit     7201624258
    6  1611421466964373           scroll     7201624258
    7  1611393680523542        view_item     5444923623
    8  1611393723523759           scroll     5444923623
    9  1611393675117845        page_view     5444923623
    

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
print(df.head())
```

    CPU times: total: 3.02 s
    Wall time: 24.9 s
    




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
      <td>1604997315221864</td>
      <td>user_engagement</td>
      <td>3310034080</td>
      <td>Google Youth F/C Pullover Hoodie</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1604999410142324</td>
      <td>user_engagement</td>
      <td>4022806627</td>
      <td>Android Garden Tee Orange</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1604998926310461</td>
      <td>page_view</td>
      <td>4022806627</td>
      <td>Google Leather Strap Hat Black</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1604999035202555</td>
      <td>page_view</td>
      <td>4022806627</td>
      <td>Google Infant Hero Onesie Grey</td>
      <td>https://shop.googlemerchandisestore.com/Google...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1604999263084837</td>
      <td>view_item</td>
      <td>4022806627</td>
      <td>Google Crew Combed Cotton Sock</td>
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
        
    
