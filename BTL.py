import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


# Đọc dữ liệu lên DataFrame
dataFrame = pd.read_csv('cardekho.csv', sep=';')

# Hiển thị 15 dòng dữ liệu đầu tiên
print(dataFrame.head(15))

# In thông tin các cột trong DF
dataFrame['max_power'] = pd.to_numeric(dataFrame['max_power'], errors='coerce')
dataFrame.info()
print('Shape tập dữ liệu: ', dataFrame.shape)


def plt_categories_counts(label_name):
    counts = dataFrame[label_name].value_counts()
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar')
    plt.title(f'Số lượng giá trị duy nhất của cột {label_name}')
    plt.xlabel(label_name)
    plt.ylabel('Counts')
    plt.show()


# Đếm các giá trị duy nhất của các cột có Dtype phi số
plt_categories_counts('owner')
plt_categories_counts('fuel')
plt_categories_counts('seller_type')
plt_categories_counts('transmission')


def categories_counts(label_name):
    print(f'-----------------------------------------------')
    print(f'Số các giá trị duy nhất của cột {label_name}:')
    print(dataFrame[label_name].value_counts())


# Đếm các giá trị duy nhất của các cột có Dtype phi số
categories_counts('owner')
categories_counts('fuel')
categories_counts('seller_type')
categories_counts('transmission')


def statistical_missing_data(df):
    missing_data = df.isnull().sum()
    duplicate_data = df.duplicated().sum()

    print("Số liệu thiếu trong mỗi cột:")
    print(missing_data)
    print("\nSố liệu trùng lặp:")
    print(duplicate_data)


# Thống kê dữ liệu khuyết
statistical_missing_data(dataFrame)


def plt_statistics_missing_data(dataFrame):
    # Dữ liệu khuyết
    missing_data = dataFrame.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    plt.figure(figsize=(10, 5))
    missing_data.plot(kind='bar')
    plt.title('Dữ liệu khuyết trong mỗi cột')
    plt.xlabel('Tên cột')
    plt.ylabel('Số lượng dữ liệu khuyết')
    plt.show()
    # Dữ liệu lặp
    duplicate_data = dataFrame.duplicated().sum()
    print("\nSố liệu trùng lặp:", duplicate_data)


# Thống kê dữ liệu khuyết
plt_statistics_missing_data(dataFrame)

print('Shape tập dữ liệu trước khi xóa cột:\t', dataFrame.shape)
# Bỏ các cột không còn cần thiết
dataFrame = dataFrame.drop('seller_type', axis=1)
print('Shape tập dữ liệu sau khi xóa cột:\t', dataFrame.shape)
# Kết quả sau khi xóa cột
dataFrame.head()

# Trích xuất các cột giá trị số
data_numeric = dataFrame.select_dtypes(include=['number'])
print(data_numeric)
# Thống kê dữ liệu khuyết của DL số
plt_statistics_missing_data(data_numeric)
# Thống kê dữ liệu khuyết sau khi trích xuất giá trị số
statistical_missing_data(data_numeric)

# Đếm các giá trị duy nhất của các cột có giá trị số
categories_counts('max_power')
categories_counts('seats')
categories_counts('engine')
categories_counts('mileage(km/ltr/kg)')


# Điền khuyết bằng giá trị Mode
def fillna_with_mode(data_numeric):
    df_missing = data_numeric.isnull()
    df_filled = data_numeric.fillna(df_missing.mean())
    print(df_filled)
    print('SỐ giá trị khuyết của các cột sau khi điền khuyết')
    print(df_filled.isnull().sum())


fillna_with_mode(data_numeric)
# Kết quả
dataFrame.head()

# Số các giá trị duy nhất của các cột có giá trị
categories_counts('mileage(km/ltr/kg)')
categories_counts('engine')
categories_counts('max_power')
categories_counts('seats')

# Tìm các giá trị duy nhất để chuẩn bị từ điển mapping
categories_counts('fuel')
categories_counts('transmission')
categories_counts('owner')


# Ánh xạ các giá trị phân loại thành nhãn kiểu số
def map_categorical(col_name, dictionary):
    dataFrame[col_name] = dataFrame[col_name].map(dictionary)


fuel_type_mapping = {'CNG': 1, 'Diesel': 2, 'Petrol': 3, 'LPG': 4}
transmission_type_mapping = {'Manual': 1, 'Automatic': 2}
owner_type_mapping = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}

map_categorical('fuel', fuel_type_mapping)
map_categorical('transmission', transmission_type_mapping)
map_categorical('owner', owner_type_mapping)

# Kết quả sau khi ánh xạ
dataFrame.head()


def describe_data(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    data_describe = {
        'Count': numeric_df.count(),
        'Min': numeric_df.min(),
        'Max': numeric_df.max(),
        'Mean': numeric_df.mean(),
        'Mode': numeric_df.mode().iloc[0],  # mode() return a DataFrame
        '25%': numeric_df.quantile(0.25),
        '50%': numeric_df.median(),
        '75%': numeric_df.quantile(0.75),
        'IQR': numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
        'Variance': numeric_df.var(),
        'STD': numeric_df.std()
    }
    return pd.DataFrame(data_describe).T


describe_data(dataFrame)

# Tạo bảng thống kê bằng hàm cho 1 cột
print(data_numeric['selling_price'].describe())


def min_max_nor(data, column):
    col = np.array(data[column]).reshape(-1, 1)
    scaler = MinMaxScaler()
    data[column] = scaler.fit_transform(col)
    return data


# Example usage:
df_minmax_nor = min_max_nor(data_numeric, 'selling_price')
df_minmax_nor = min_max_nor(data_numeric, 'mileage(km/ltr/kg)')
print(df_minmax_nor.head(10))

column1 = data_numeric['engine']
column2 = data_numeric['selling_price']
plt.scatter(column1, column2, label='Dữ liệu', color='blue', marker='o')
plt.title('Biểu đồ Scatter')
plt.xlabel('engine')
plt.ylabel('selling_price')
plt.legend()
plt.show()

sns.heatmap(data_numeric.corr())

split_name = dataFrame["name"].str.split(" ", expand=True)
dataFrame["Manufacturer"] = split_name[0]
# Đặt màu cho các cột
colors = sns.color_palette("husl", len(dataFrame["Manufacturer"].unique()))
# Tạo các cột với màu chỉ định
plt.figure(figsize=(10, 6))
plot = sns.countplot(x='Manufacturer', data=dataFrame, hue='Manufacturer', palette=colors, legend=False)
plt.xticks(rotation=90)
# Lấy các giá trị trên các cột
for p in plot.patches:
    plot.annotate(
        p.get_height(),
        (p.get_x() + p.get_width() / 2000.00, p.get_height()),
    )
plt.title("SỐ LƯỢNG XE ĐÃ BÁN THEO HÃNG SẢN XUẤT")
plt.xlabel("HÃNG SẢN XUẤT")
plt.ylabel("SỐ LƯỢNG XE")
plt.show()

average_prices = data_numeric.groupby('year')['selling_price'].mean().reset_index()
# Vẽ biểu đồ cột
plt.figure(figsize=(12, 8))
sns.barplot(data=average_prices, x='year', y='selling_price', hue='year', palette='viridis')
plt.title('Giá trung bình của xe theo từng năm')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize = (20,8))
sns.heatmap(dataFrame.corr(numeric_only = True), annot = True, cmap = plt.cm.Blues)
plt.show()

# Tính trung bình giá xe của mỗi loại xe cho mỗi năm
average_prices = dataFrame.groupby(['year', 'Manufacturer'])['selling_price'].mean().reset_index()
# Tạo biểu đồ heatmap
plt.figure(figsize=(30, 20))
heatmap_data = average_prices.pivot_table(index="Manufacturer", columns="year", values="selling_price", aggfunc="mean")
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Average Price'})
plt.title('Heatmap of Average Car Prices by Manufacturer and Year')
plt.xlabel('Year')
plt.ylabel('Manufacturer')
plt.tight_layout()
plt.show()

average_prices = dataFrame.groupby('transmission')['selling_price'].mean().reset_index().sort_values(by='selling_price', ascending=False)
# Vẽ biểu đồ boxplot cho trung bình giá
plt.figure(figsize=(10, 6))
sns.boxplot(x='transmission', y='selling_price', data=dataFrame, hue='transmission',  order=average_prices['transmission'], palette="Set3")
plt.title('Boxplot of Average Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print(dataFrame)

X = data_numeric[['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']]
Y = data_numeric['selling_price']
X['Fuel'] = dataFrame['fuel']
X['Transmission'] = dataFrame['transmission']
X['Owner'] = dataFrame['owner']
# Xử lý dữ liệu trước khi sử dụng np.polyfit
data_cleaned = pd.concat([X, Y], axis=1).dropna() # Loại bỏ hàng có giá trị NaN
X_cleaned = data_cleaned.iloc[:, :-1]
Y_cleaned = data_cleaned.iloc[:, -1]
# Vẽ scatter plot và hồi quy
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for i, ax in enumerate(axes.flat):
    if i < len(X_cleaned.columns):  # Kiểm tra không vượt quá số cột
        x = X_cleaned.iloc[:, i]
        y = Y_cleaned
        # Kiểm tra xem cột hiện tại có đủ dữ liệu khác biệt không
        if x.nunique() > 1:  # Đảm bảo cột có ít nhất 2 giá trị khác biệt
            ax.scatter(x, y)
            fit = np.polyfit(x, y, deg=1)
            ax.plot(x, fit[0] * x + fit[1], color='red')
            ax.set_xlabel(X_cleaned.columns[i])
            ax.set_ylabel('Price')
        else:
            ax.set_title(f"Not enough variation in {X_cleaned.columns[i]}")
            ax.axis('off')

plt.tight_layout()
plt.show()

# Thêm cột hằng số vào X
X = dataFrame[['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']]
# Chuyển đổi tất cả các cột của X thành dạng số, xử lý lỗi bằng cách ép kiểu thành NaN
X = X.apply(pd.to_numeric, errors='coerce')
# Loại bỏ các hàng có giá trị NaN trong X
X = X.dropna()
X = sm.add_constant(X)  # Thêm cột hằng số (intercept) vào X
# Biến phụ thuộc
Y = dataFrame['selling_price']
# Đảm bảo Y cũng ở dạng số
Y = pd.to_numeric(Y, errors='coerce')
# Lọc Y để khớp với các hàng trong X sau khi loại bỏ NaN
Y = pd.to_numeric(Y, errors='coerce')[X.index]  # Đồng bộ chỉ số
# Khởi tạo và fit mô hình OLS
model_OLS = sm.OLS(Y, X).fit()
# Hiển thị bảng OLS
print(model_OLS.summary())

# chia tập train và test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 )

# tạo model
model = LinearRegression()

# train model
model.fit(X_train, Y_train)

# ktra độ chính xác
Y_pred = model.predict(X)
print(Y_pred.shape)


def my_function(Year_input, Kilometers_Driven_input, Mileage_input, Seats_input, Engine_Number_input, Power_Number_input, Transmission_input):
    # Tạo mảng dữ liệu đầu vào
    x = np.array([[Year_input, Kilometers_Driven_input, Mileage_input, Seats_input, Engine_Number_input,
                   Power_Number_input, Transmission_input]])
    x = sm.add_constant(x)
    # Dự đoán giá trị
    result = model.predict(x)
    # Y_pred_test = model.predict(X_test)

    # Tính toán các chỉ số hiệu suất
    r2 = r2_score(Y_test, model.predict(X_test))  # R^2 score
    r_square = model.score(X, Y)  # R^2 từ model
    slope = model.coef_  # Hệ số góc
    intercept = model.intercept_  # Hằng số

    # In kết quả
    print(f"Dự đoán giá cho các thông số đầu vào: {result[0]}")
    # print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"R Square: {r_square}")
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")


Year_input = float(input("Nhập Year: "))
Kilometers_Driven_input = float(input("Nhập Kilometers_Driven: "))
Mileage_input = float(input("Nhập Mileage: "))
Seats_input = int(input("Nhập Seats: "))
Engine_Number_input = float(input("Nhập Engine_Number: "))
Power_Number_input = float(input("Nhập Power_Number: "))
Transmission_input = int(input("Nhập Transmission('Manual': 1, 'Automatic': 2): "))

my_function(Year_input, Kilometers_Driven_input, Mileage_input, Seats_input, Engine_Number_input, Power_Number_input, Transmission_input)


# # Test Ridge Regression
# # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # Huấn luyện mô hình RandomForestRegressor
# n_estimators_value = 100  # Số cây trong rừng, có thể điều chỉnh để tối ưu mô hình
# model_2 = RandomForestRegressor(n_estimators=n_estimators_value, random_state=42)
# model_2.fit(X_train, Y_train)
#
# # Lưu mô hình
# # dump(model, 'random_forest_model.joblib')
#
# # Dự đoán trên tập kiểm tra và in ra các metric đánh giá
# Y_pred = model_2.predict(X_test)
# mae = mean_absolute_error(Y_test, Y_pred)
# mse = mean_squared_error(Y_test, Y_pred)
# r2 = r2_score(Y_test, Y_pred)
#
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"R^2 Score: {r2:.2f}")
