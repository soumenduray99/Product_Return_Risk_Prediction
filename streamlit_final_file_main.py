import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sb
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from itertools import product
from xgboost import XGBClassifier
import pickle
import streamlit as st
from imblearn.combine import SMOTETomek
import plotly.express as px
import fitz  
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import tempfile

# Initialize global components
pc = Pinecone(api_key="pcsk_2RoTjB_TokUmpAk6whfneoCgZoQqrt42m9cL3WZ8Hpe2MFpGejKvHtJ5K5AK7iMtPpTXvm")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
llm = pipeline('text-generation', model='gpt2')

# Global variables for Pinecone
index_name = "document-qa-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(index_name)

# Generating Live Data
def data_gen(n):

  # Select the Data
  categories = ["T-shirts","Shirts","Trousers","Jeans","Dresses","Skirts","Shorts","Jackets","Sweaters","Activewear","Sleepwear",
                       "Undergarments","Formal Wear","Ethnic Wear","Suits"]
  brand=['Nike','Puma','Adidas','Pantaloons','Zudio','Raymond',"Levie's",'H&M' ]
  Age=[i for i in range(18,71) ] 
  Gender=['Male','Female' ]
  state = [ "Andhra Pradesh", "Arunachal Pradesh","Assam","Bihar", "Chhattisgarh",  "Goa", "Gujarat", "Haryana", "Himachal Pradesh","Jharkhand","Karnataka",
           "Kerala","Madhya Pradesh","Maharashtra","Manipur", "Meghalaya","Mizoram","Nagaland","Odisha","Punjab", "Rajasthan","Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"]
  Qty=[i for i in range(1,8) ]
  price=[i for i in range(400,1001 ) ]
  discount=[10,15,20,25,30 ]
  product_rating=[ float(i) for i in np.arange(1,5.5,0.5)  ]
  high_return_risk=['No','Yes']
  
  # Generate The data based on rows 
  age_d=[ int(i) for i in np.random.choice(Age,size=n)  ] 
  gnder_d=[str(i) for i in np.random.choice(Gender,size=n)  ] 
  state_d=[ str(i) for i in np.random.choice(state,size=n)  ] 
  cat_d=[ str(i) for i in np.random.choice(categories,size=n)  ] 
  bd_d=[ str(i) for i in np.random.choice(brand,size=n)  ] 
  qty_d=[ int(i) for i in np.random.choice(Qty,size=n)  ] 
  prc_d=[ float(i) for i in np.random.choice(price,size=n)  ] 
  dsc_d=[ int(i) for i in np.random.choice(discount,size=n)  ]
  pr_d=[ float(i) for i in np.random.choice(product_rating,size=n)  ]
  hrr_d=[str(i) for i in np.random.choice(high_return_risk,size=n)  ]
   
  # Create the dataframe for this
  df= pd.DataFrame(index= [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") for i in range(n) ] )
  df['Category']=cat_d   
  df['Brand']=bd_d
  df['Age']=age_d
  df['Gender']=gnder_d
  df['State']=state_d
  df['Quantity']=qty_d
  df['Price']=prc_d
  df['Discount']=dsc_d
  df['Product Rating']=pr_d
  df['High_Return_Risk']=hrr_d

  return df

# Model building , training and testing
data_train=data_gen(500000)

def model_train(df):
  # Outliers Treatment
  cont=df.dtypes[df.dtypes!='object'].index
  for i in cont:
    q1=np.percentile(df[i],25)
    q3=np.percentile(df[i],75)
    iqr=q3-q1
    up=q3+1.5*iqr
    lw=q1-1.5*iqr
    df[i]=np.where(df[i]>up,up,df[i])
    df[i]=np.where(df[i]<lw,lw,df[i])

  # Label Encoding
  le_dict = {}
  clm_enc=['State','Category','Gender', 'Brand','High_Return_Risk' ]
  for i in clm_enc:
    le=LabelEncoder()
    df[i]=le.fit_transform(df[i])
    le_dict[i]=le

  # Select x and y
  x=df.drop(['High_Return_Risk'],axis=1)
  y=df['High_Return_Risk']

  # Train and Test Devision
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)

  # Scaling
  ss=StandardScaler()
  xtrain_ss=pd.DataFrame(ss.fit_transform(xtrain),columns=xtrain.columns)
  xtest_ss=pd.DataFrame(ss.transform(xtest),columns=xtest.columns)

  # Smote Tomek
  stk = SMOTETomek(random_state=0)
  xtrain_ss, ytrain = stk.fit_resample(xtrain_ss,ytrain)

  # Model Apply
  xgb=XGBClassifier(eta=0.1,gamma=4.75,reg_alpha=1.1)
  xgb.fit(xtrain_ss,ytrain)
  y_pd_xgb=xgb.predict(xtest_ss)
  acc_scr=accuracy_score(ytest,y_pd_xgb)
  
  return xgb,ss,le_dict

dta1=data_train.copy()
def model_load():
  model=model_train(dta1)
  pickle.dump(model,open('Return_Risk_Prediction.pkl','wb'))

  with open('Return_Risk_Prediction.pkl', 'rb') as file:
    mdl = pickle.load(file)
    return mdl

# RAG Chatbot Functions
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []
    
    for page in doc:
        text = page.get_text()
        chunks = [p for p in text.split('\n') if len(p.strip()) > 0]
        text_chunks.extend(chunks)
        
    return text_chunks

def index_documents(pdf_paths):
    all_chunks = []
    for path in pdf_paths:
        chunks = process_pdf(path)
        all_chunks.extend(chunks)
        
    embeddings = embedding_model.encode(all_chunks)
    
    vectors = []
    for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        vectors.append({
            "id": f"vec_{idx}",
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        })
        
    index.upsert(vectors=vectors)

def query_documents(question, top_k=3):
    query_embedding = embedding_model.encode(question).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    context = "\n".join([match.metadata['text'] for match in results.matches])
    
    prompt = f"""Answer the question based on the context below. If you don't know the answer, say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    response = llm(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        truncation=True
    )
    
    return response[0]['generated_text'].replace(prompt, "").strip()

# Using Streamlit for python building
st.set_page_config(layout="wide", page_title="Product Analytics Dashboard")
st.image("e_commerce.png")
st.title("Product Return Risk Analysis")
sample_pdfs = [
        "sample_python_doc.pdf",
        "sample_ml_doc.pdf",
        "sample_sql_doc.pdf"
    ]

#tab1,tab2,tab3=st.tab([''] )

with st.sidebar:
    st.title("Enter the value for Rows ")
    data_row = st.number_input("Enter number of rows:", min_value=5000, max_value=900000, step=5000)
    num_row=st.slider("Enter number of rows to display",min_value=20,max_value=1000,step=1 )

data=data_gen(data_row)

tab1,tab2,tab3,tab4=st.tabs(['Product_Return_Table','Dashboard','Model_Prediction','Q&A'])

with tab1:
  st.subheader("Product Return Risk Table")
  st.dataframe(data.head(num_row), width=1000, height=500) 
  st.subheader('To export the file, click on: ',)
  st.download_button(label='Export File',data= data.to_csv().encode('utf-8'),
                   file_name=f'Product_Return_Risk_dataset.csv',mime='text/csv')

with tab2:
  
  cl_k1,cl_k2,cl_k3,cl_k4,cl_k5=st.columns(5)
  category_optionx=st.selectbox("Category",['All','Ethnic Wear','Sweaters','Undergarments','Suits','Sleepwear','T-shirts','Jackets',
                                          'Activewear','Skirts','Trousers', 'Jeans','Shirts','Formal Wear','Dresses','Shorts'] )
  if category_optionx=='All':
    datax=data
  else:
    datax=data[data['Category']==category_optionx]
    
  with cl_k1:
    sm_sls=round(sum((datax['Price']*datax['Quantity'])*(1-datax['Discount']/100))/(10**7),2)
    st.metric('Total Sales',f'₹{sm_sls} cr')
  with cl_k2:
    sm_qty=sum(datax['Quantity'])
    st.metric('Total Quantity Sold',f'{sm_qty}')
  with cl_k3:
    sm_sls=round(sum((datax['Price']*datax['Quantity'])*(1-datax['Discount']/100) ),2)
    aov=round((sm_sls/datax.shape[0]),2)
    st.metric('Average Order Value',f'₹{aov}' )  
  with cl_k4:
    av_rt=round(sum(datax['Product Rating'])/len(datax['Product Rating']),2)
    st.metric('Average Rating',av_rt)
  with cl_k5:
    av_ag=round(sum(datax['Age'])/len(datax['Age']))
    st.metric('Average Age',av_ag )
    
  cl11,cl12,cl13=st.columns(3)
  with cl11:
    st.subheader('Categorywise Total Customer ')
    fig11,ax=plt.subplots(figsize=(5,7))
    fr=datax['Category'].value_counts().reset_index()
    fr.sort_values('count',ascending=True,inplace=True)
    ax.barh(fr['Category'],fr['count'],color='yellow',edgecolor='black')
    ax.bar_label(ax.containers[0],label_type='center')
    ax.set_xlabel('Total Customer')
    ax.set_ylabel('Category')
    st.pyplot(fig11)
  
  with cl12:
    st.subheader('Gender')
    fig12,ax=plt.subplots(figsize=(6,6))
    gnd_cn= datax['Gender'].value_counts()
    ax.pie(gnd_cn,autopct='%0.2f',colors=['green','yellow'] ,labels=gnd_cn.index)
    ax.legend()
    st.pyplot(fig12)
  
  with cl13:
    st.subheader('Brand Rating')
    fig13,ax=plt.subplots(figsize=(6,6))
    rt_br=datax.groupby(['Brand'])['Product Rating'].mean().reset_index()
    rt_br['Product Rating']=rt_br['Product Rating'].round(2)
    rt_br.rename(columns={'Product Rating':'Rating'},inplace=True)
    ax.bar(rt_br['Brand'],rt_br['Rating'],color='yellow',edgecolor='black')
    ax.bar_label(ax.containers[0],label_type='center')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Rating')
    ax.set_xticklabels(rt_br['Brand'],rotation=90)
    st.pyplot(fig13)
    
  cl21,cl22=st.columns(2)
  with cl21:
    with st.container():
      st.subheader("Statewise Total Qty sold")
      fig21,ax=plt.subplots(figsize=(15,7))
      br_am=datax.groupby(['State'])['Quantity'].sum().reset_index()
      ax.bar(br_am['State'],br_am['Quantity'],color='yellow',edgecolor='black')
      ax.bar_label(ax.containers[0])
      ax.set_xlabel('State')
      ax.set_ylabel('Quantity')
      plt.xticks(rotation=45, ha='right')
      st.pyplot(fig21,bbox_inches='tight')
  
  with cl22 :
    with st.container():
      st.subheader('Brandwise Sales')
      fig22,ax=plt.subplots(figsize=(13,6))
      dft1=datax.copy()
      dft1['Amount']=(datax['Price']*datax['Quantity'])*(1-datax['Discount']/100)/(10^7)
      br_am=dft1.groupby(['Brand'])['Amount'].sum().reset_index()
      br_am['Amount']=round( br_am['Amount'])
      ax.barh(br_am['Brand'],br_am['Amount'],color='yellow',edgecolor='black')
      ax.bar_label(ax.containers[0],label_type='center')
      ax.set_xlabel('Brand')
      ax.set_ylabel('Amount')
      st.pyplot(fig22)

  cl31,cl32,cl33=st.columns(3)
  with cl31:
    st.subheader('Age vs Gender ')
    dta1=datax[datax['High_Return_Risk']=='Yes']
    dta1['Age_Grp']=np.where(dta1['Age'].between(18,25),'18-25',
                         np.where(dta1['Age'].between(26,35),'26-35',
                         np.where(dta1['Age'].between(36,45),'36-45',
                         np.where(dta1['Age'].between(46,55),'46-60','>60'))))
    fig,ax=plt.subplots(figsize=(6,6))
    ck=sb.countplot(data=dta1,x='Age_Grp',hue='Gender',ax=ax,color='yellow',edgecolor='black',legend=['Male','Female'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    for j in ck.containers:
      ck.bar_label(j)
    ax.legend()
    st.pyplot(fig)
    
  with cl32:
    st.subheader('Amount Range')
    fig32,ax=plt.subplots(figsize=(5,5))
    dft2=datax.copy()
    dft2['Amount']=(datax['Price']*datax['Quantity'])*(1-datax['Discount']/100)/(10^7)
    ax.hist(dft2['Amount'],bins=np.arange(1,max(dft2['Amount']),50),color='yellow',edgecolor='black')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Frequency')
    st.pyplot(fig32)
    
  with cl33:
    st.subheader("Brandwise Return Risk")
    fig33,ax=plt.subplots(figsize=(5,4.6))
    dft3=datax[datax['High_Return_Risk']=='Yes']
    er1=dft3['Brand'].value_counts().reset_index()
    ax.bar(er1['Brand'],er1['count'],color='yellow',edgecolor='black')
    ax.bar_label(ax.containers[0],label_type='center')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Total_Return')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig33)
    
  cl41=st.container() 
  with cl41:
    st.subheader('Predicted Value')
    dt_pd_id=data.reset_index()
    dt_pdx=dt_pd_id.copy()
    dt_prdx=dt_pdx[['Category','Brand','Age','Gender','State','Quantity','Price','Discount','Product Rating']]
    dt_px=dt_prdx.copy()
    clm_enc3=['State','Category','Gender', 'Brand']
    modelx, scalerx, le_dictx = model_train(data)
    for i in clm_enc3:
      dt_px[i] = le_dictx[i].transform(dt_px[i])
    dt_scaledx = scalerx.transform(dt_px)
    predx = modelx.predict(dt_scaledx)
    dt_prd_fx=dt_pdx[['index','Category','Brand','Age','Gender','State','Quantity','Price','Discount','Product Rating']]
    dt_prd_fx['Predicted_value']=predx
    num_row_fx=st.slider("Enter number of rows to display",min_value=10,max_value=100,step=1 )
    dt_pred_fxy=dt_prd_fx.head(num_row_fx)
    fig41 = px.line( x= list(range(dt_pred_fxy.shape[0]))  , 
                    y=dt_pred_fxy['Predicted_value'], markers=True,
                    title="Return Risk Prediction Over Time",
                    color_discrete_sequence=["yellow"])               
    fig41.update_layout( xaxis_title="Date", yaxis_title="Return_Risk",hovermode="x unified")
    st.plotly_chart(fig41, use_container_width=True)

    
with tab3:
  st.subheader('Prediction using options selected')
  category_option=st.selectbox("Category",['Ethnic Wear','Sweaters','Undergarments','Suits','Sleepwear','T-shirts','Jackets',
                                          'Activewear','Skirts','Trousers', 'Jeans','Shirts','Formal Wear','Dresses','Shorts'] )
  brand_option=st.selectbox('Brand',['Puma', 'Pantaloons', "Levie's", 'Zudio', 'Nike', 'Raymond', 'H&M', 'Adidas'])
  Age=st.number_input("Age",18,70)
  Gender=st.selectbox('Gender',['Male','Female'])
  State=st.selectbox('State',['Uttar Pradesh','Arunachal Pradesh','Manipur','Uttarakhand','West Bengal','Meghalaya',
                              'Punjab','Mizoram','Karnataka','Kerala','Maharashtra','Himachal Pradesh',
                              'Nagaland','Rajasthan','Andhra Pradesh','Telangana','Tamil Nadu','Haryana','Chhattisgarh',
                              'Jharkhand','Gujarat','Tripura','Madhya Pradesh','Goa','Assam','Odisha','Bihar','Sikkim'])
  Quantity=st.number_input('Quantity',1,10)
  Price=st.number_input('Price',100,1000)
  Discount=st.number_input('Discount',10,35)
  Prod_Rating = st.number_input('Product Rating', min_value=1.00, max_value=5.00, step=0.01, format="%.2f")
  
  data_mdl_tr=data_gen(50000)
  model, scaler, le_dict = model_train(data_mdl_tr)
  if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Category': category_option,
        'Brand': brand_option,
        'Age': Age,
        'Gender': Gender,
        'State': State,
        'Quantity': Quantity,
        'Price': Price,
        'Discount': Discount,
        'Product Rating': Prod_Rating }])
    clm_enc2=['State','Category','Gender', 'Brand']
    for i in clm_enc2:
      input_df[i] = le_dict[i].transform([input_df[i][0]])[0]
      
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    prd=np.where(pred==0,'No','Yes')
    st.success(f"High Return Risk : {prd[0]}")
  
  st.subheader('Prediction using Generated Data')
  num_row_pred=st.number_input("Enter number of rows ",min_value=500,max_value=10000,step=500 )
  dt_pd=data_gen(num_row_pred)
  dt_prd=dt_pd[['Category','Brand','Age','Gender','State','Quantity','Price','Discount','Product Rating']]
  dt_p=dt_prd.copy()
  st.dataframe(dt_p, width=1000, height=200)
  if st.button("Predict Data"):
    clm_enc3=['State','Category','Gender', 'Brand']
    for i in clm_enc3:
      dt_p[i] = le_dict[i].transform(dt_p[i])
    dt_scaled = scaler.transform(dt_p)
    pred = model.predict(dt_scaled)
    prd=np.where(pred==0,'No','Yes')
    st.success(" Prediction Successfully Done, to get the prediction click on 'Export File' button ")
    dt_prd['High_Return_Risk_Prediction']=prd
    st.download_button(label='Export File',data= dt_prd.to_csv().encode('utf-8'),
                   file_name=f'Product_Return_Risk_Prediction.csv',mime='text/csv')

  st.subheader('Prediction using Import files') 
  uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
  if uploaded_file is not None:
    dt_prd2 = pd.read_csv(uploaded_file)
    st.dataframe(dt_prd2, width=1000, height=200)
    dt_p2=dt_prd2.copy() 
    if st.button("Predict Data for File "):
      clm_enc4=['State','Category','Gender', 'Brand']
      for i in clm_enc4:
        dt_p2[i] = le_dict[i].transform(dt_p2[i])
      dt_scaled2 = scaler.transform(dt_p2)
      pred2 = model.predict(dt_scaled2)
      prd2=np.where(pred2==0,'No','Yes')
      st.success(" Prediction Successfully Done, to get the prediction click on 'Export Uploaded File' button ")
      dt_prd2['High_Return_Risk_Prediction']=prd2
      st.download_button(label='Export Uploaded File',data= dt_prd2.to_csv().encode('utf-8'),
                   file_name=f'Product_Return_Risk_Prediction_Uploaded_File.csv',mime='text/csv')
  else:
    st.info("Please upload a CSV file.")
 
with tab4:
  st.subheader("Document Q&A Assistant")
  user_question = st.text_input("Ask about Python, ML, or SQL:")
  if user_question:
    with st.spinner("Searching documents..."):
      response = query_documents(user_question)
    st.text_area("Answer:", value=response, height=200)
    st.markdown("### Available Documentation:")
    st.write("- Python Basics")
    st.write("- Machine Learning Concepts")
    st.write("- SQL Fundamentals")   



    
# Project Overview

# The project focuses on analyzing product return risks in e-commerce using machine learning and data visualization tools. It includes:
# •	Data Generation: Simulates product-related data (e.g., categories, brands, prices, ratings) for analysis.
# •	Model Development: Builds a predictive model using XGBoost to classify whether a product has a high return risk.
# •	Dashboard Creation: Implements an interactive Streamlit dashboard for data exploration, model predictions, and insights.
# •	Document Q&A Assistant: Integrates a Retrieval-Augmented Generation (RAG) chatbot to answer user queries based on uploaded documents.


# Setup and Execution Instructions

# 1.	Environment Setup:
# •	Install required Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, streamlit, imblearn, plotly, fitz, pinecone, sentence-transformers, and transformers.
# •	Configure Pinecone API for document indexing.

# 2.	Execution Steps:
# •	Run the script to generate synthetic product data using the data_gen function.
# •	Train the XGBoost model using the model_train function, which includes preprocessing steps like outlier treatment, label encoding, scaling, and SMOTE-Tomek sampling.
# •	Save the trained model using pickle for future use.
# •	Launch the Streamlit dashboard to interact with the data:
# •	Explore product return risk data in tabular format or through visualizations.
# •	Predict return risk for individual products or bulk datasets.
# •	Use the Q&A assistant to query uploaded documents.

# 3.	Streamlit Dashboard Features:
# •	Tabs for data exploration (Product_Return_Table), visual analytics (Dashboard), model predictions (Model_Prediction), and document-based Q&A (Q&A).
# •	Options to export datasets and predictions as CSV files.


# Model and Tool Explanation

# 1.	Machine Learning Model:
# •	XGBoost Classifier:
# •	Hyperparameters: Learning rate (ηη), gamma, and regularization (αα).
# •	Handles imbalanced data using SMOTE-Tomek sampling.
# •	Outputs predictions on product return risk with high accuracy.

# 2.	Data Preprocessing Tools:
# •	StandardScaler: Normalizes numerical features for consistent scaling.
# •	LabelEncoder: Encodes categorical variables into numeric format.

# 3.	Visualization Tools:
# •	Matplotlib & Seaborn: Used for creating bar charts, pie charts, histograms, etc., to explore trends in categories, brands, gender distribution, and more.
# •	Plotly Express: Generates interactive line plots for predicted values over time.

# 4.	Document Q&A Assistant:
# •	Utilizes Pinecone for vector-based document indexing and SentenceTransformer for embedding text chunks.
# •	GPT-2 model is employed for generating answers based on user queries.

# 5.	Interactive Dashboard:
# •	Built using Streamlit to provide user-friendly interfaces for data exploration, prediction tasks, and document-based Q&A functionalities.

    
  
  