import streamlit as st
import pandas as pd 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
import tkinter as tk
from tkinter import ttk, StringVar
import time

#from query import *
st.set_option('deprecation.showPyplotGlobalUse', False)

#navicon and header
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")  

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#current date
from datetime import datetime
current_datetime = datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d')
formatted_day = current_datetime.strftime('%A')
 
st.header(" MACHINE LEARNING WORKFLOW | RANDOM FOREST ")
st.markdown(
 """
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
 <hr>

<div class="card mb-3">
<div class="card">
  <div class="card-body">
    <h3 class="card-title"style="color:#007710;"><strong>⏱ SƠ LƯỢC VỀ ÁP DỤNG RANDOM FOREST</strong></h3>
    <p class="card-text">Viết đánh giá sơ bộ tại đây</p>
    <p class="card-text"><small class="text-body-secondary"> </small></p>
  </div>
</div>
</div>
 <style>
    [data-testid=stSidebar] {
         color: white;
         text-size:24px;
    }
</style>
""",unsafe_allow_html=True
)



# df=pd.read_csv("advanced_regression.csv")


with st.sidebar:
 st.markdown(f"<h4 class='text-success'>{formatted_day}: {formatted_date}</h4>Analytics Dashboard V: 01/2023<hr>", unsafe_allow_html=True)
 

# # switcher
# year_= st.sidebar.multiselect(
#     "PICK YEAR:",
#     options=df["year"].unique(),
#     default=df["year"].unique()
# )
# month_ = st.sidebar.multiselect(
#     "PICK MONTH:",
#     options=df["month"].unique(),
#     default=df["month"].unique(),
# )

# df_selection = df.query(
#     "month == @month_ & year ==@year_"
# )

#download csv
# with st.sidebar:
#  df_download = df_selection.to_csv(index=False).encode('utf-8')
#  st.download_button(
#     label="Download DataFrame from Mysql",
#     data=df_download,
#     key="download_dataframe.csv",
#     file_name="my_dataframe.csv"
#  )

#drop unnecessary fields
# df_selection.drop(columns=["id","year","month"],axis=1,inplace=True)

df = pd.read_csv('churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Tạo một imputer với chiến lược thay thế là trung bình
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Áp dụng imputer vào cột TotalCharges của df
df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

y = df['Churn']

# Start

with st.expander("⬇ VỀ DỮ LIỆU GỐC"):
    st.write("Examining the correlation between the independent variables (features) and the dependent variable before actually building and training a regression model. This is an important step in the initial data exploration and analysis phase to understand the relationships between variables.")
    st.dataframe(df)  # Hiển thị dữ liệu gốc

df1 = df.drop(['customerID','Churn'], axis=1)
df1 = df.drop(columns=['gender', 'InternetService', 'MultipleLines',
                   'PhoneService', 'StreamingMovies','StreamingTV'])
with st.expander("⬇ DỮ LIỆU SAU KHI ĐÃ XÓA ĐI CÁC CỘT"):
    st.write("Trong bước này nhóm đã xóa đi các cột không cần thiết như 'customerID','Churn' (cột dự đoán), gender', 'InternetService', 'MultipleLines','PhoneService', 'StreamingMovies','StreamingTV ")
    st.dataframe(df)  # Hiển thị dữ liệu gốc

with st.expander("⬇ THỐNG KÊ MÔ TẢ CỦA DỮ LIỆU"):
    st.write("Thêm text gì đó vào đây")
    st.dataframe(df.describe())
    
with st.expander("⬇ KHÁM PHÁ CÁC BIẾN SỐ"):
    st.write("Các biểu đồ histogram và mô tả chi tiết cho các biến số:")
    numeric_cols = [f for f in df.columns if df[f].dtype != "O"]
    for col in numeric_cols:
        st.write(f"### {col}")
        st.write(df[col].value_counts())
        st.write(df[col].describe())
        plt.figure(figsize=(4, 4))
        df[col].hist()
        plt.title(col)
        st.pyplot(plt)
with st.expander(" Cái gì đây ko rõ nữa :V: "):
  st.caption("SeniorCitizen: 0 & 1")
  st.caption("Parter No 0 Yes 1")
  st.caption("Dependents No 0 Yes 1")
  st.caption("OnlineSecurity No 0 No internet service 1 Yes 2")
  st.caption("OnlineBackup No 0 No internet service 1 Yes 2")
  st.caption("DeviceProtection No 0 No internet service 1 Yes 2")
  st.caption("TechSupport No 0 No internet service 1 Yes 2")
  st.caption("Contact Month-to-month 0 One year 1 Two year 2")
  st.caption("PaperlessBilling No 0 Yes 1")
  st.caption("PaymentMethod Bank transfer (automatic) 0 Credit card (automatic) 1 Electronic check 2 Mailed check 3")
  
scaler = StandardScaler()
df = scaler.fit_transform(df)

with st.expander("⬇ XỬ LÝ OUTLIERS VÀ MISSING VALUES"):
    df = pd.DataFrame(df)

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Draw boxplots for each numerical variable
    sns.boxplot(data=df, orient="h", palette="Set2")

    # Add titles and labels
    plt.title("Boxplot of Variables in the Dataset")
    plt.xlabel("Value")

    # Display the plot
    plt.tight_layout()
    st.pyplot(plt)

    # Detect outliers using IQR or Z-score method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Find outliers using IQR
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)

    # Print rows with outliers
    outlier_rows = df[outliers]
    st.write("Rows with outliers:")
    st.dataframe(outlier_rows)

    # Handling outliers: Replace with NaN
    df[outliers] = np.nan

    # Handling missing values: Impute with mean
    df.fillna(df.mean(), inplace=True)

    # Print DataFrame after handling outliers
    st.write("DataFrame after handling outliers:")
    st.dataframe(df)

X = df

# UpSampling
sm = SMOTEENN()
X_res, y_res = sm.fit_resample(X, y)

# Phân tách dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Định nghĩa mô hình K-Nearest Neighbors với các tham số cố định
model = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Tạo một pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Chuẩn hóa đặc trưng
    ('model', model)
])

# Đo thời gian bắt đầu
start_time = time.time()

# Huấn luyện pipeline trên dữ liệu huấn luyện
pipeline.fit(X_train, y_train)

# Đo thời gian kết thúc
end_time = time.time()
elapsed_time = end_time - start_time

# Dự đoán trên tập kiểm thử
y_pred = pipeline.predict(X_test)

# Tính điểm độ chính xác
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

# Hiển thị kết quả
with st.expander("⬇ KẾT QUẢ PHÂN TÍCH MÔ HÌNH K-NEAREST NEIGHBORS"):
    st.write(f"Model: K-Nearest Neighbors")
    st.write(f"Thời gian chạy: {elapsed_time:.4f} giây")
    st.write(f"Độ chính xác trên tập kiểm thử: {accuracy:.4f}")
    st.write("Báo cáo phân loại:")
    st.json(report)
    st.write(f"Diện tích dưới đường cong ROC (ROC AUC): {roc_auc:.4f}")

# End


#theme_plotly = None # None or streamlit

# with st.expander("⬇ EXPLORATORY  ANALYSIS"):
#  st.write("Examining the correlation between the independent variables (features) and the dependent variable before actually building and training a regression model. This is an important step in the initial data exploration and analysis phase to understand the relationships between variables.")
#  col_a,col_b=st.columns(2)
#  with col_a:
#   st.subheader("Interest Vs Unemployment")
#   plt.figure(figsize=(4, 4))
#   sns.regplot(x=df_selection['interest_rate'], y=df_selection['unemployment_rate'],color="#007710")
#   plt.xlabel('Interest Rate')
#   plt.ylabel('Unemployment Rate')
#   plt.title('Interest Rate vs UnemploymentRate: Regression Plot')
#   st.pyplot()
   

# with col_b:
#  plt.figure(figsize=(4, 4))
#  st.subheader("Interest Vs Index Price")
#  sns.regplot(x=df_selection['interest_rate'], y=df_selection['index_price'],color="#007710")
#  plt.xlabel('Interest Rate')
#  plt.ylabel('Unemployment Rate')
#  plt.title('InterestRate vs IndexPrice Regression Plot')
#  st.pyplot()

#  fig, ax = plt.subplots()
#  st.subheader("Variables outliers",)
#  sns.boxplot(data=df, orient='h',color="#FF4B4B")
#  plt.show()
#  st.pyplot()

# with st.expander("⬇ EXPLORATORY VARIABLE DISTRIBUTIONS BY FREQUENCY: HISTOGRAM"):
#   df_selection.hist(figsize=(16,8),color='#007710', zorder=2, rwidth=0.9,legend = ['unemployment_rate']);
#   st.pyplot()

# with st.expander("⬇ EXPLORATORY VARIABLES DISTRIBUTIONS:"):
#  st.subheader("Correlation between variables",)
#  #https://seaborn.pydata.org/generated/seaborn.pairplot.html
#  pairplot = sns.pairplot(df_selection,plot_kws=dict(marker="+", linewidth=1), diag_kws=dict(fill=True))
#  st.pyplot(pairplot)


# #checking null value
# with st.expander("⬇ NULL VALUES, TENDENCY & VARIABLE DISPERSION"):
#  a1,a2=st.columns(2)
#  a1.write("number of missing (NaN or None) values in each column of a DataFrame")
#  a1.dataframe(df_selection.isnull().sum(),use_container_width=True)
#  a2.write("insights into the central tendency, dispersion, and distribution of the data.")
#  a2.dataframe(df_selection.describe().T,use_container_width=True)



# # train and test split
# with st.expander("⬇ DEFAULT CORRELATION"):
#  st.dataframe(df_selection.corr())
#  st.subheader("Correlation",)
#  st.write("correlation coefficients between Interest Rate Rate & Unemployment Rate")
#  plt.scatter(df_selection['interest_rate'], df_selection['unemployment_rate'])
#  plt.ylabel("Unemployment rate")
#  plt.xlabel("Interest rate")
#  st.pyplot()

# try:

#  # independent and dependent features
#  X=df_selection.iloc[:,:-1] #left a last column
#  y=df_selection.iloc[:,-1] #take a last column

#  # train test split
#  from sklearn.model_selection import train_test_split
#  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


#  with st.expander("⬇ UNIFORM  DISTRIBUTION "):
#   st.subheader("Standard Scores (Z-Scores)",)
#   st.write("transform data so that it has a mean (average) of 0 and a standard deviation of 1. This process is also known as [feature scaling] or [standardization.]")
#   from sklearn.preprocessing import StandardScaler
#   scaler=StandardScaler()
#   X_train=scaler.fit_transform(X_train)
#   X_test=scaler.fit_transform(X_test)
#   st.dataframe(X_train)


#  from sklearn.linear_model import LinearRegression
#  regression=LinearRegression()
#  regression.fit(X_train,y_train)

# #cross validation
#  from sklearn.model_selection import cross_val_score
#  validation_score=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=3)

#  col1, col3,col4,col5 = st.columns(4)
#  col1.metric(label="🟡 MEAN VALIDATION SCORE", value=np.mean(validation_score), delta=f"{ np.mean(validation_score):,.0f}")

#  #prediction
#  y_pred=regression.predict(X_test)


# # performance metrics
#  from sklearn.metrics import mean_squared_error, mean_absolute_error
#  meansquareerror=mean_squared_error(y_test,y_pred)
#  meanabsluteerror=mean_absolute_error(y_test,y_pred)
#  rootmeansquareerror=np.sqrt(meansquareerror)

#  col3.metric(label="🟡 MEAN SQUARED ERROR ", value=np.mean(meansquareerror), delta=f"{ np.mean(meansquareerror):,.0f}")
#  col4.metric(label="🟡 MEAN ABSOLUTE ERROR", value=np.mean(meanabsluteerror), delta=f"{ np.mean(meanabsluteerror):,.0f}")
#  col5.metric(label="🟡 ROOT MEAN SQUARED ERROR", value=np.mean(rootmeansquareerror), delta=f"{ np.mean(rootmeansquareerror):,.0f}")


#  with st.expander("⬇ COEFFICIENT OF DETERMINATION | R2"):
#   from sklearn.metrics import r2_score
#   score=r2_score(y_test,y_pred)
#   st.metric(label="🔷 r", value=score, delta=f"{ score:,.0f}")

#  with st.expander("⬇ ADJUSTED CORRERATION COEFFICIENT | R"):
#   #display adjusted R_squared
#   st.metric(label="🔷 Adjusted R", value=((1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))), delta=f"{ ((1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))):,.0f}")
 

#  with st.expander("⬇ CORRERATION COEFFICIENT | r"):
#   #display correlation
#   st.write(regression.coef_)
 

#  #https://seaborn.pydata.org/generated/seaborn.regplot.html
#  c1,c2,c3=st.columns(3)
#  with c1:
#   with st.expander("⬇ LINE OF BEST FIT"):
#    st.write("regression line that best represents the relationship between the independent variable(s) and the dependent variable in a linear regression model. This line is determined through a mathematical process that aims to minimize the error between the observed data points and the predicted values generated by the model.")
#    plt.figure(figsize=(8, 6))
#    sns.regplot(x=y_test, y=y_pred,color="#FF4B4B",line_kws=dict(color="#007710"))
#    plt.xlabel('Interest Rate')
#    plt.ylabel('Unemployment Rate')
#    plt.title('Interest Rate vs Unemployment_Rate Regression Plot')
#    st.pyplot()

#  with c2:
#   with st.expander("⬇ RESIDUAL"):
#    st.write("residuals: refers to the differences between the actual observed values (the dependent variable, often denoted as y) and the predicted values made by a regression model (often denoted as y_pred). These residuals represent how much the model's predictions deviate from the actual data points")
#    residuals=y_test-y_pred
#    st.dataframe(residuals)

#  with c3:
#   with st.expander("⬇ MODEL PERFORMANCE | NORMAL DISTRIBUTION CURVE"):
#    st.write("distribution of a continuous random variable where data tends to be symmetrically distributed around a mean (average) value. It is a fundamental concept in statistics and probability theory.")
#    sns.displot(residuals,kind='kde',legend=True,color="#007710") #kernel density estimator
#    st.pyplot()


#  with st.expander("⬇ OLS, or Ordinary Least Squares Method"): 
#   import statsmodels.api as sm
#   model=sm.OLS(y_train,X_train).fit()
#   st.write(model.summary())

#  st.sidebar.image("data/logo1.png")
#  style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow="#F71938")

# except:
#  st.error("❌ THE AMOUNT OF DATA YOU SELECTED IS NOT ENOUGH FOR THE MODEL TO PERFORM PROPERLY")

 


