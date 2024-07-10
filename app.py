import streamlit as st
import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Đọc và hiển thị 10 dòng đầu của bộ dữ liệu
data = pd.read_csv('Hotel Reservations.csv')

# Mã hóa dữ liệu với LabelEncoder
lb_make = LabelEncoder()
data['Booking_ID'] = lb_make.fit_transform(data['Booking_ID'])
data['room_type_reserved'] = lb_make.fit_transform(data['room_type_reserved'])
data['type_of_meal_plan'] = lb_make.fit_transform(data['type_of_meal_plan'])
data['market_segment_type'] = lb_make.fit_transform(data['market_segment_type'])
data['booking_status'] = lb_make.fit_transform(data['booking_status'])
data['avg_price_per_room'] = lb_make.fit_transform(data['avg_price_per_room'])

# Xác định các thuộc tính mô tả và thuộc tính dự đoán
feature = ['required_car_parking_space', 'repeated_guest', 'lead_time', 'market_segment_type',
           'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
           'avg_price_per_room', 'no_of_special_requests']
target = ['booking_status']

X = data[feature]
y = data[target]

# Phân tích dữ liệu và mô hình hóa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_scaled, y_train)

# Streamlit App
st.title("Dự đoán sự rời bỏ của khách hàng khách sạn")
st.sidebar.title("Nhập các thuộc tính để dự đoán")

required_car_parking_space = st.sidebar.selectbox('Khách hàng sử dụng bãi đỗ xe', ['Không', 'Có'])
required_car_parking_space = 1 if required_car_parking_space == 'Có' else 0

repeated_guest = st.sidebar.selectbox('Khách hàng đã từng đặt phòng', ['Không', 'Có'])
repeated_guest = 1 if repeated_guest == 'Có' else 0

market_segment_map = {
    'Offline': 0,
    'Online': 1,
    'Corporate': 2,
    'Aviation': 3,
    'Complementary': 4
}

market_segment_type = st.sidebar.selectbox('Phân khúc khách hàng', list(market_segment_map.keys()))
market_segment_type_value = market_segment_map[market_segment_type]

avg_price_per_room = st.sidebar.number_input('Giá phòng trung bình trong ngày', min_value=0, step=1)
lead_time = st.sidebar.number_input('Số ngày khách hàng đặt phòng', min_value=1, max_value=1000, step=1)
no_of_previous_cancellations = st.sidebar.number_input('Số lần đặt chỗ khách hàng đã hủy trước đó', min_value=0, step=1)
no_of_previous_bookings_not_canceled = st.sidebar.number_input('Số lượng đặt chỗ khách hàng không hủy trước', min_value=0, step=1)
no_of_special_requests = st.sidebar.number_input('Số lượng yêu cầu dịch vụ đặc biệt', min_value=0, step=1)

if st.button("Dự đoán"):
    input_data = np.array([[required_car_parking_space, repeated_guest, lead_time, market_segment_type_value,
                            no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                            avg_price_per_room, no_of_special_requests]])
    input_data_scaled = scaler.transform(input_data)
    prediction = rfc.predict(input_data_scaled)
    predicted_class = "Rời bỏ" if prediction[0] == 0 else "Ở lại"
    st.write(f"Dự đoán: {predicted_class}")
