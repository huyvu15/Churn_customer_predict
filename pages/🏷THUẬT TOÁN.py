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
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


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
    <h3 class="card-title"style="color:#007710;"><strong>⏱ SƠ LƯỢC VỀ ÁP DỤNG 3 THUẬT TOÁN CHO HIỆU NĂNG CAO NHẤT</strong></h3>
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


with st.sidebar:
 st.markdown(f"<h4 class='text-success'>{formatted_day}: {formatted_date}</h4>Analytics Dashboard V: 01/2023<hr>", unsafe_allow_html=True)


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
with st.expander("mÔ TẢ "):
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

# XGBoost

# Định nghĩa mô hình XGBoost với các tham số cố định
model1 = XGBClassifier(n_estimators=200, learning_rate=0.5)

# Tạo một pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Chuẩn hóa đặc trưng
    ('model', model1)
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
with st.expander("⬇ KẾT QUẢ PHÂN TÍCH MÔ HÌNH XGBOOST"):
    st.write(f"Model: XGBoost")
    st.write(f"Thời gian chạy: {elapsed_time:.4f} giây")
    st.write(f"Độ chính xác trên tập kiểm thử: {accuracy:.4f}")
    st.write("Báo cáo phân loại:")
    st.json(report)
    st.write(f"Diện tích dưới đường cong ROC (ROC AUC): {roc_auc:.4f}")


model2 = RandomForestClassifier(max_depth=None, n_estimators=150)

# Tạo một pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Chuẩn hóa đặc trưng
    ('model', model2)
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
with st.expander("⬇ KẾT QUẢ PHÂN TÍCH MÔ HÌNH RANDOM FOREST"):
    st.write(f"Model: Random Forest")
    st.write(f"Thời gian chạy: {elapsed_time:.4f} giây")
    st.write(f"Độ chính xác trên tập kiểm thử: {accuracy:.4f}")
    st.write("Báo cáo phân loại:")
    st.json(report)
    st.write(f"Diện tích dưới đường cong ROC (ROC AUC): {roc_auc:.4f}")

# End
