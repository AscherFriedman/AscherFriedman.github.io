#!/usr/bin/env python
# coding: utf-8

# # Python for Bigquerey and SQL

# In[1]:


import pandas as pd
import numpy as np
import os
from google.cloud import bigquery


# In[3]:


JSON_PATH = "path to JSON FILE from Google"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=JSON_PATH
JSON_PATH = "C:\\Users\\Asche\\Documents\\Code\\API\\maiatestcase-1ae98fa45803.json"
client = bigquery.Client()


# # High Level Idea
# Pulling very specific GA4 data takes a bit of SQL knowledge, and intricate queries.
# 
# Writing these yourself is a headache, but with some python knowledge, its actually fairly easy.
# 

# # Getting GA4 Running
# 
# Guide to set up the GA4 API
# https://developers.google.com/analytics/devguides/reporting/core/v4/quickstart/service-py
# 
# Explore a free GA4 dataset
# https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset

# In[28]:


# Lets grab a sample of the data, and just a few columns for simplicity sake


# In[88]:


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


# This data is filled with nested data here which is an issue if we want to do data-science or analysis
# 
# What if we want specific value inside "event_params" as its own column, which is currently nested inside event_params?
# 

# In[83]:


sample_df.iloc[0]['event_params'][:5]


# # Cross Join
# This trick is given by google as its a common issue
# 
# Theoretically this is how it works, we join the main table every row of the nested table.
# This means for this one observation there would now be 5, one with each key
# Then we simply tell that we only want records with the key equal to ga_session_number

# In[87]:


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


# Look how nice this looks! We have pulled out a nested subclause as a column
# Naturally the next question becomes, what about pulling out multiple of these nested columns?
# 
# Theoretically we just want to keep making new tables and join them to our previous tables 
# resulting in multiple exploded columns all in one row
# 
# Lets turn this into two functions, one for generating a single querey for exploding one column out, and another for joining that to another table!

# In[69]:


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


# In[ ]:


#Now lets use these functions to grab 3 exploded out columns, page_location, page_title and ga_session_id


# In[80]:


start_date = '20201101'
end_date =  '20201202'
q1 = single_querey(start_date,end_date,'int','event_params','ga_session_id')
q2 = single_querey(start_date,end_date,'string','event_params','page_title')
q3 = single_querey(start_date,end_date,'string','event_params','page_location')
q_new = join_queries(q1,q2,joined_col = 'page_title')
q_new = join_queries(q_new,q3,joined_col = 'page_location')


# In[86]:


get_ipython().run_cell_magic('time', '', 'df = client.query(q_new).to_dataframe()\ndf.head()')


# Here we are able to run a very complex querey that would be a serious headache to write, automatically using python, and a pull a 
# full month of data in less than half a minute.

# In[82]:


#Heres what that querey looks like
print(q_new)

