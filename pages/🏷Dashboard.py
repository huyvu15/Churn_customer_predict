import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from streamlit_extras.metric_cards import style_metric_cards
from io import StringIO

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

st.header("Dashboard overview")
# st.success("The main objective is to measure if Number of family dependents or Wives may influence a person to supervise many projects")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_data():
    df = pd.read_csv('Churn.csv')
    # original_labels = df['Churn']
    df = df.drop(columns=['customerID'])
    return df

df = load_data()

original_labels = df['Churn']
# df = df.drop(columns=['customerID', 'Churn'])

data_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Assuming 'Projects' is the target variable for regression; this is just for demonstration.
df['Projects'] = df['TotalCharges']  # This line is for example purpose, change it according to your actual data

selected_column = st.selectbox('SELECT INPUT X FEATURE', df.select_dtypes("number").columns)
X = sm.add_constant(df[selected_column])  # Adding a constant for intercept

# Fitting the model
model = sm.OLS(df['Projects'], X, missing='drop').fit()

c1, c2, c3, c4 = st.columns(4)
# Printing general intercept
c1.metric("INTERCEPT:", f"{model.params.iloc[0]:,.4f}")

# Printing R-squared
c2.metric("R SQUARED", f"{model.rsquared:,.2f}", delta="is it strong relationship?")

# Printing adjusted R-squared
c3.metric("ADJUSTED R", f"{model.rsquared_adj:,.3f}")

# Printing standard error
c4.metric("STANDARD ERROR", f"{model.bse.iloc[0]:,.4f}")

# Printing correlation coefficient
style_metric_cards(background_color="#FFFFFF", border_left_color="#686664")

b1, b2 = st.columns(2)
# Printing predicted values
data = {
    'X feature': selected_column,
    'Prediction': model.predict(X),
    'Residuals': model.resid
}

dt = pd.DataFrame(data)
b1.dataframe(dt, use_container_width=True)

with b2:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[selected_column], df['Projects'], label='Actual')
    plt.plot(df[selected_column], model.predict(X), color='red', label='Predicted')
    plt.xlabel(selected_column)
    plt.ylabel('Projects')
    plt.title(f'Line of Best Fit ({selected_column} vs Projects)')
    plt.grid(color='grey', linestyle='--')
    plt.legend()

    # Setting outer border color
    plt.gca().spines['top'].set_color('gray')
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_color('gray')
    plt.gca().spines['right'].set_color('gray')
    st.pyplot(plt)


st.header("Phân tích tổng thể")

# Định nghĩa màu sắc
colors = ['#1f77b4', '#aec7e8']

# Vẽ biểu đồ
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Tổng số lượng khách hàng
axs[0, 0].text(0.5, 0.5, str(df.shape[0]), horizontalalignment='center', verticalalignment='center', fontsize=50)
axs[0, 0].set_title('Total Customer Count')
axs[0, 0].axis('off')

# Tỷ lệ rời bỏ
churn_rate = (original_labels == 'Yes').mean() * 100
axs[0, 1].text(0.5, 0.5, f'{churn_rate:.2f}%', horizontalalignment='center', verticalalignment='center', fontsize=50)
axs[0, 1].set_title('Churn Rate')
axs[0, 1].axis('off')

# Chi phí trung bình hàng tháng
avg_monthly_charges = df['MonthlyCharges'].mean()
axs[0, 2].text(0.5, 0.5, f'{avg_monthly_charges:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=50)
axs[0, 2].set_title('Average Monthly Charges')
axs[0, 2].axis('off')

# Phân phối theo giới tính
labels = df['gender'].value_counts().index
sizes = df['gender'].value_counts().values
axs[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axs[1, 0].set_title('Gender Distribution')

# Tỷ lệ giữ chân khách hàng so với rời bỏ
labels = original_labels.value_counts().index
sizes = original_labels.value_counts().values
axs[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axs[1, 1].set_title('Customer Retention vs. Churn Percentage')

# Phân phối người cao tuổi
bars = axs[1, 2].bar(df['SeniorCitizen'].value_counts().index, df['SeniorCitizen'].value_counts().values, color=colors)
axs[1, 2].set_title('Number of Senior Citizens')
axs[1, 2].set_xticks([0, 1])
axs[1, 2].set_xticklabels(['0', '1'])
axs[1, 2].set_ylabel('Customer Count')
for bar in bars:
    height = bar.get_height()
    axs[1, 2].text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
st.pyplot(fig)

st.header("Tác động của nhân khẩu học đến tỷ lệ rời bỏ")
import seaborn as sns

# Định nghĩa màu sắc
colors = ['#1f77b4', '#aec7e8']  # Xanh và xanh nhạt pastel

# Thiết lập grid của biểu đồ
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Biểu đồ phân phối khách hàng theo giới tính và churn
sns.countplot(x='gender', hue='Churn', data=df, ax=axs[0, 0], palette=colors)
axs[0, 0].set_title('Churned and Retained Customers: Gender Breakdown')
axs[0, 0].set_xlabel('Gender')
axs[0, 0].set_ylabel('Count of Customers')
for p in axs[0, 0].patches:
    axs[0, 0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Biểu đồ phân phối khách hàng có người phụ thuộc và churn
sns.countplot(x='Dependents', hue='Churn', data=df, ax=axs[0, 1], palette=colors)
axs[0, 1].set_title('Churned and Retained Customers: Customer with Dependents')
axs[0, 1].set_xlabel('Customer with Dependents')
axs[0, 1].set_ylabel('Count of Customers')
for p in axs[0, 1].patches:
    axs[0, 1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Biểu đồ phân phối khách hàng theo độ tuổi và churn
sns.countplot(x='SeniorCitizen', hue='Churn', data=df, ax=axs[1, 0], palette=colors)
axs[1, 0].set_title('Churned and Retained Customers based on Age')
axs[1, 0].set_xlabel('Senior Citizen')
axs[1, 0].set_ylabel('Count of Customers')
axs[1, 0].set_xticklabels(['No', 'Yes'])
for p in axs[1, 0].patches:
    axs[1, 0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Biểu đồ phân phối khách hàng có đối tác và churn
sns.countplot(x='Partner', hue='Churn', data=df, ax=axs[1, 1], palette=colors)
axs[1, 1].set_title('Churned and Retained Customers: Customer with Partners')
axs[1, 1].set_xlabel('Customer with Partner')
axs[1, 1].set_ylabel('Count of Customers')
for p in axs[1, 1].patches:
    axs[1, 1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
st.pyplot(fig)

st.header("Tác động loại hợp đồng")

# Thiết lập màu sắc cho biểu đồ
colors = ['#1f77b4', '#aec7e8']  # Xanh và xanh nhạt pastel

# Dashboard layout
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Số lượng khách hàng rời bỏ trong vòng một năm đầu tiên
sns.histplot(data=df[df['tenure'] <= 12], x='tenure', hue='Churn', multiple='stack', palette=colors, ax=axs[0, 0])
axs[0, 0].set_title('Churn within First Year of Signing Up')
axs[0, 0].set_xlabel('Tenure (months)')
axs[0, 0].set_ylabel('Number of Customers')

# Số lượng khách hàng rời bỏ trong tháng đầu tiên và hợp đồng hàng tháng
monthly_contracts = df[(df['tenure'] <= 1) & (df['Contract'] == 'Month-to-month')]
sns.countplot(x='tenure', hue='Churn', data=monthly_contracts, palette=colors, ax=axs[0, 1])
axs[0, 1].set_title('Churn within First Month with Monthly Contract')
axs[0, 1].set_xlabel('Tenure (months)')
axs[0, 1].set_ylabel('Number of Customers')
for p in axs[0, 1].patches:
    axs[0, 1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Tỷ lệ khách hàng rời bỏ theo loại hợp đồng
sns.countplot(x='Contract', hue='Churn', data=df, palette=colors, ax=axs[1, 0])
axs[1, 0].set_title('Churn Rate by Contract Type')
axs[1, 0].set_xlabel('Contract Type')
axs[1, 0].set_ylabel('Number of Customers')
for p in axs[1, 0].patches:
    axs[1, 0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Tỷ lệ khách hàng rời bỏ theo phương thức thanh toán
sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette=colors, ax=axs[1, 1])
axs[1, 1].set_title('Churn Rate by Payment Method')
axs[1, 1].set_xlabel('Payment Method')
axs[1, 1].set_ylabel('Number of Customers')
for p in axs[1, 1].patches:
    axs[1, 1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    
plt.tight_layout()
st.pyplot()


st.header("Sử dụng dịch vụ")

# Định nghĩa màu sắc
colors = ['#1f77b4', '#aec7e8']  # Xanh và xanh nhạt pastel
pie_colors = ['#1f77b4', '#aec7e8', '#17becf']  # Thêm màu xanh khác

# Tính toán tỷ lệ phần trăm
def calculate_percentage(df, column, hue):
    count_df = df.groupby([column, hue]).size().reset_index(name='count')
    total_df = df.groupby(column).size().reset_index(name='total')
    percentage_df = pd.merge(count_df, total_df, on=column)
    percentage_df['percentage'] = (percentage_df['count'] / percentage_df['total']) * 100
    return percentage_df

# Tạo figure và axes cho các biểu đồ
fig, axs = plt.subplots(4, 2, figsize=(18, 24))

# Hàm vẽ biểu đồ cột tỷ lệ phần trăm
def plot_percentage_bar(ax, data, x, hue, title):
    percentage_df = calculate_percentage(data, x, hue)
    sns.barplot(x=x, y='percentage', hue=hue, data=percentage_df, palette=colors, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Percentage of Customers')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Vẽ các biểu đồ
plot_percentage_bar(axs[0, 0], df, 'PhoneService', 'Churn', 'Customer Retention and Churn Percentage: Phone Service')
plot_percentage_bar(axs[1, 0], df, 'MultipleLines', 'Churn', 'Customer Retention and Churn Percentage: Multiple Lines')
plot_percentage_bar(axs[2, 0], df, 'DeviceProtection', 'Churn', 'Customer Churn and Retention Percentages: Device Protection')
plot_percentage_bar(axs[3, 0], df, 'OnlineBackup', 'Churn', 'Customer Churn and Retention Percentages: Online Backup')
plot_percentage_bar(axs[2, 1], df, 'OnlineSecurity', 'Churn', 'Customer Churn and Retention Percentages: Online Security')
plot_percentage_bar(axs[3, 1], df, 'TechSupport', 'Churn', 'Customer Churn and Retention Percentages: Tech Support')

# Biểu đồ tròn cho Internet Service
internet_churn = df[df['Churn'] == 'Yes']['InternetService'].value_counts()
internet_total = df['InternetService'].value_counts()
internet_percentage = (internet_churn / internet_total) * 100

axs[0, 1].pie(internet_churn, labels=internet_churn.index, autopct='%1.1f%%', colors=pie_colors)
axs[0, 1].set_title('Churned Customers vs Internet Services Categories')

plt.tight_layout()
st.pyplot()

st.header("Tác động của tài chính và phương thức thanh toán")

# Thêm cột phân loại chi phí hàng tháng và tổng chi phí
df['MonthlyChargesRange'] = pd.cut(df['MonthlyCharges'], bins=[0, 20, 40, 60, 80, 100, 120], labels=['0-20', '21-40', '41-60', '61-80', '81-100', '101-120'])
df['TotalChargesRange'] = pd.cut(df['TotalCharges'], bins=[0, 2000, 4000, 6000, 8000, 10000], labels=['0-2000', '2001-4000', '4001-6000', '6001-8000', '8001-10000'])

# Định nghĩa màu sắc
bar_color = '#1f77b4'  # Màu xanh dương đậm cho biểu đồ cột
pie_colors = ['#1f77b4', '#aec7e8', '#17becf', '#7f7f7f', '#ff7f0e']  # Thêm màu xanh khác cho Credit card

# Tạo figure và axes cho các biểu đồ
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Hàm vẽ biểu đồ cột
def plot_bar(ax, df, column, title):
    churn_data = df[df['Churn'] == 'Yes'][column].value_counts().sort_index()
    sns.barplot(x=churn_data.index, y=churn_data.values, color=bar_color, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Count of Customers')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Hàm vẽ biểu đồ tròn
def plot_pie(ax, data, column, title):
    churn_data = df[df['Churn'] == 'Yes'][column].value_counts()
    ax.pie(churn_data, labels=churn_data.index, autopct='%1.1f%%', colors=pie_colors)
    ax.set_title(title)

# Vẽ các biểu đồ
plot_bar(axs[0, 0], df, 'MonthlyChargesRange', 'Churned Customers by Monthly Charges Ranges')
plot_bar(axs[1, 0], df, 'TotalChargesRange', 'Churned Customers by Total Charges Range')
plot_pie(axs[0, 1], df, 'PaymentMethod', 'Churned Customers by Type of Payment Method')

# Loại bỏ biểu đồ trống và điều chỉnh layout
axs[1, 1].axis('off')
plt.tight_layout()
st.pyplot()

st.header("Thống kê dữ liệu mẫu")

st.caption("Thông tin dữ liệu")

# Bắt nội dung của df.info() vào một string
buffer = StringIO()
df.info(buf=buffer)
info_string = buffer.getvalue()

# Hiển thị nội dung df.info() trên Streamlit
st.text(info_string)

st.caption("Thống kê mô tả")

# Bắt nội dung của df.info() vào một string
# df.describe().T
# info_string1 = buffer.getvalue()

# Hiển thị nội dung df.info() trên Streamlit
st.text(df.describe().T)


plt.figure(figsize=(8, 6))
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')

# Hiển thị đồ thị trên Streamlit
st.pyplot(plt)



st.subheader('Monthly Charges vs. Total Charges')

# 17. Monthly Charges vs. Total Charges
plt.figure(figsize=(8, 15))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df)
plt.title('Monthly Charges vs. Total Charges')

st.pyplot(plt)



# Lọc các cột số
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Tính toán ma trận tương quan
correlation_matrix = df_numeric.corr()

# Tính toán tương quan với 'Churn' (nếu có cột 'Churn')
if 'Churn' in df_numeric.columns:
    churn_correlations = correlation_matrix['Churn'].sort_values()

# Vẽ biểu đồ heatmap
plt.figure(figsize=(14, 16))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')

# Hiển thị biểu đồ heatmap trên Streamlit
st.pyplot(plt)

# # Vẽ biểu đồ thanh ngang cho tương quan với 'Churn'
# plt.figure(figsize=(10, 8))
# churn_correlations.drop('Churn').plot(kind='barh', color='skyblue')  # Exclude self-correlation
# plt.title('Correlation with Churn')
# plt.xlabel('Correlation Coefficient')
# plt.ylabel('Attributes')
# plt.grid(True)

# # Hiển thị biểu đồ thanh ngang trên Streamlit
# st.pyplot(plt)