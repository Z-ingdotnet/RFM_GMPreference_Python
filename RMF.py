import pyodbc 
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from efficient_apriori import apriori
import mlxtend
from datetime import timedelta
import matplotlib.pyplot as plt
import squarify
from sqlalchemy import create_engine
import time

#import pyreadr
#result = pyreadr.read_r('/path/to/file.RData') 


cnxn = pyodbc.connect('driver={custom define driver};server=***************:8000;database=mytestDW;trusted_connection=yes')  

cursor = cnxn.cursor()
cursor.execute('select * from Model.analytics.GOps_GMProductPreference_DEV_20200601')

df = pd.read_sql_query("select * from Model.analytics.GOps_GMProductPreference_DEV_20200601 where TagGroup='Prodcut Type'",cnxn)



df = pd.read_sql_query('''
                       select distinct r.CustomerID
                       from GMRating r with (nolock) 
                       left join Customerinfo p with (nolock) on r.Customer_Key = p.Customer_Key 
                       inner join (select r.CustomerID, max(r.GDate) LastGDate
                                   , case when dateadd(month, -6, '2019-01-01') <= dateadd(month, 16 * (-1), max(r.GDate)) then dateadd(month, 16 * (-1), max(r.GDate)) else dateadd(month, -6, '2019-01-01') end as StartGDate 
                                   from GMRating r with (nolock) 
                                   left join Customerinfo p with (nolock) on r.Customer_Key = p.Customer_Key 
                                           where p.RealCustomerFlag = 1 and p.Status = 'A' 
                                                                      and r.GDate between '2019-01-01' 
                                                                      and '2020-07-31' group by r.CustomerID ) d on r.CustomerID = d.CustomerID 
                                                                                                               and r.GDate between d.StartGDate 
                                                                                                               and d.LastGDate group by r.CustomerID, d.StartGDate, d.LastGDate
                      ''',cnxn)


df = pd.read_sql_query('''
            select a.CustomerID, a.TWinAmount, a.accountingdate,a.GameCount,CustomerValueScore/100 WS from (
                       select Customer_Key,CustomerID, TWinAmount, accountingdate ,GameCount from Customer_Transaction r with (nolock) where accountingdate between '2019-01-01' and '2020-07-31' and TWinAmount>0 )a
                       inner join Customerinfo p with (nolock) on a.Customer_Key = p.Customer_Key and p.RealCustomerFlag = 1 and p.Status = 'A' 
                       inner join (select Customer_Key,CustomerValueScore from PlayerInfo) rpt on a.Customer_Key = rpt.Customer_Key
                      ''',cnxn)



# Convert AccountingDate from object to datetime format
df['accountingdate'] = pd.to_datetime(df['accountingdate'])




# Create snapshot date
snapshot_date = df['accountingdate'].max() + timedelta(days=1)
print(snapshot_date)

# Grouping by CustomerID
data_process = df.groupby(['CustomerID']).agg({
        'accountingdate': lambda x: (snapshot_date - x.max()).days,
        'GameCount': 'sum',
        'TWinAmount': 'sum',
        'WS': 'max'})
# Rename the columns 
data_process.rename(columns={'accountingdate': 'Recency',
                         'GameCount': 'Frequency',
                         'TWinAmount': 'MonetaryValue'}, inplace=True)

print(data_process.head(10))
print('{:,} rows; {:,} columns'.format(data_process.shape[0], data_process.shape[1]))

# Plot RFM distributions
plt.figure(figsize=(12,10))
# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(data_process['Recency'])
# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(data_process['Frequency'])
# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(data_process['MonetaryValue'])
# Show the plot
plt.show()


# Create labels for Recency and Frequency
r_labels = range(4, 0, -1); f_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(data_process['Recency'], q=4, labels=r_labels)
# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(data_process['Frequency'], q=4, labels=f_labels)
# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(data_process['MonetaryValue'], q=4, labels=f_labels)

# Create new columns R and F and M 
data_process = data_process.assign(R = r_groups.values, F = f_groups.values,M = m_groups.values)
# Create new column M
data_process.head()

# Concat RFM quartile values to create RFM Segments
def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])
data_process['RFM_Segment'] = data_process.apply(join_rfm, axis=1)
rfm = data_process
rfm.head()

#rfm.drop(['R','F','M'], axis=1, inplace=True)  


rfm_count_unique = rfm.groupby('RFM_Segment')['RFM_Segment'].nunique()
print(rfm_count_unique.sum())

rfm['Value_Score'] = rfm[['R','F','M','WS']].sum(axis=1)
print(rfm['Value_Score'].head())

# Define rfm_level function
def rfm_level(df):
    if df['Value_Score'] >= 10:
        return 'Can\'t Loose Them'
    elif ((df['Value_Score'] >= 9) and (df['Value_Score'] < 10)):
        return 'Champions'
    elif ((df['Value_Score'] >= 7) and (df['Value_Score'] < 8)):
        return 'Loyal'
    elif ((df['Value_Score'] >= 6) and (df['Value_Score'] < 7)):
        return 'Potential'
    elif ((df['Value_Score'] >= 5) and (df['Value_Score'] < 6)):
        return 'Average'
    #elif ((df['Value_Score'] >= 4) and (df['Value_Score'] < 5)):
    #   return 'Needs Attention'
    else:
        return 'Require Activation'
# Create a new variable RFM_Level
rfm['Value_Level'] = rfm.apply(rfm_level, axis=1)
# Print the header with top 5 rows to the console
rfm.head()


# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = rfm.groupby('Value_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)
# Print the aggregated dataset
print(rfm_level_agg)

rfm_level_agg.columns = rfm_level_agg.columns.droplevel()
rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfm_level_agg['Count'], 
              label=['Can\'t Loose Them',
                     'Champions',
                     'Loyal',
                     #'Needs Attention',
                     'Potential', 
                     'Average', 
                     'Require Activation'], alpha=.6 )
plt.title("Value Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()




#con = create_engine('mssql+pyodbc://*******:ps@***************H01:8000/custommodel?driver=customdefineddriver')
#df.to_sql("transfer.Iverson_Clusters", con, if_exists='replace')



from sqlalchemy import create_engine
con = create_engine('mssql+pyodbc://://*******:ps@@***************H01:8000/custommodel?driver=customdefineddriver')
df.to_sql("transfer.Iverson_Clusters", con, if_exists='replace')





from sqlalchemy import create_engine
# parameters
DB = {'servername': '***************H01:8000',
      'database': 'custommodel',
      'driver': 'driver=SQL Server Native Client 11.0'}

DB = {'servername': '***************H01:8000',
      'database': 'custommodel',
      'driver': 'driver=SQL Server'}

# create the connection
engine = create_engine('mssql+pyodbc://' + DB['servername'] + '/' + DB['database'] + "?" + DB['driver'])
conn = engine.connect()

rfm.to_sql("transfer.Iverson_Clusters", con=engine, if_exists='replace',method='multi')






rfm_test=rfm.head()

DB = {'servername': '***************H01:8000',
      'database': 'custommodel',
      'driver': 'driver=ODBC Driver 17 for SQL Server'}

# create the connection
engine = create_engine('mssql+pyodbc://' + DB['servername'] + '/' + DB['database'] + "?" + DB['driver'])
conn = engine.connect()

start_time = time.time()
#rfm_test.to_sql(name="Iverson_Clusters2", con=conn, schema='analytics',index=False,if_exists="append",method="multi")
rfm.to_sql(name="Iverson_Clusters2", con=conn, schema='analytics',if_exists="replace",method="multi",chunksize=50)
print("--- %s seconds ---" % (time.time() - start_time))


#t=engine.execute("SELECT * FROM custommodel.analytics.Iverson_Clusters2").fetchall()
#engine.execute("SELECT * FROM custommodel.analytics.Asiatop200_new_2").fetchall()



rfm.to_csv(r'X:\Users\IZ\cluster.csv')












