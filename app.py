import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the trained RandomForest model
model = 'dudoanhotel3.pkl'
rfc = pickle.load(open(model, 'rb'))

# Title of the application
st.title('Dự đoán khách hàng rời bỏ hotel')

# Sidebar for user input
st.sidebar.title('Nhập các thuộc tính để dự đoán')

# Number inputs for the features
required_car_parking_space = st.sidebar.selectbox('Khách hàng sử dụng bãi đỗ xe', ['Không', 'Có'])
required_car_parking_space = 1 if required_car_parking_space == 'Có' else 0

repeated_guest = st.sidebar.selectbox('Khách hàng đã từng đặt phòng', ['Không', 'Có'])
repeated_guest = 1 if repeated_guest == 'Có' else 0

market_segment_type = st.sidebar.selectbox('Phân khúc khách hàng', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
market_segment_map = {
    'Offline': 0,
    'Online': 1,
    'Corporate': 2,
    'Aviation': 3,
    'Complementary': 4
}
market_segment_type_value = market_segment_map[market_segment_type]

avg_price_per_room = st.sidebar.number_input('Giá phòng trung bình trong ngày', min_value=0, step=1)
lead_time = st.sidebar.number_input('Số ngày khách hàng đặt phòng', min_value=18, max_value=100, step=1)
no_of_previous_cancellations = st.sidebar.number_input('Số lần đặt chỗ khách hàng đã hủy trước đó', min_value=0, step=1)
no_of_previous_bookings_not_canceled = st.sidebar.number_input('Số lượng đặt chỗ khách hàng không hủy trước', min_value=0, step=1)
no_of_special_requests = st.sidebar.number_input('Số lượng yêu cầu dịch vụ đặc biệt', min_value=0, step=1)

# Create a DataFrame from user input
data = {
    'required_car_parking_space': [required_car_parking_space],
    'repeated_guest': [repeated_guest],
    'lead_time': [lead_time],
    'market_segment_type': [market_segment_type_value],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'avg_price_per_room': [avg_price_per_room],
    'no_of_special_requests': [no_of_special_requests]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Load and preprocess the dataset
data_path = 'Hotel Reservations.csv'
data = pd.read_csv(data_path)

# Encoding categorical variables
lb_make = LabelEncoder()
data['Booking_ID'] = lb_make.fit_transform(data['Booking_ID'])
data['room_type_reserved'] = lb_make.fit_transform(data['room_type_reserved'])
data['type_of_meal_plan'] = lb_make.fit_transform(data['type_of_meal_plan'])
data['market_segment_type'] = lb_make.fit_transform(data['market_segment_type'])
data['booking_status'] = lb_make.fit_transform(data['booking_status'])
data['avg_price_per_room'] = lb_make.fit_transform(data['avg_price_per_room'])

# Selecting features and target
feature_cols = ['required_car_parking_space', 'repeated_guest', 'lead_time', 'market_segment_type',
                'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                'avg_price_per_room', 'no_of_special_requests']
target_col = 'booking_status'

X = data[feature_cols]
y = data[target_col]

# Splitting into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prediction on the input data
input_data_scaled = scaler.transform(df)
prediction = rfc.predict(input_data_scaled)

# Display the prediction result
st.write('## Kết quả dự đoán:')
st.write('Khách hàng sẽ rời bỏ khách sạn' if prediction[0] == 1 else 'Khách hàng sẽ không rời bỏ khách sạn')
