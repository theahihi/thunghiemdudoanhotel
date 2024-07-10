import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained KNN model

model = 'dudoanhotel3.pkl'

rfc = pickle.load(open(model, 'rb'))


# Title of the application
st.title('Dự đoán khách hàng rời bỏ hotel')

# Sidebar for user input
st.sidebar.title('Nhập các thuộc tính để dự đoán')

# Number inputs for the features




import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Widget để lựa chọn có hoặc không
required_car_parking_space = st.sidebar.selectbox('Khách hàng sử dụng bãi đỗ xe', ['Không', 'Có'])

# Chuyển đổi lựa chọn sang số
if required_car_parking_space == 'Có':
    required_car_parking_space = 1
else:
    required_car_parking_space = 0

# Widget để lựa chọn có hoặc không
repeated_guest = st.sidebar.selectbox('Khách hàng đã từng đặt phòng', ['Không', 'Có'])

# Chuyển đổi lựa chọn sang số
if repeated_guest == 'Có':
    repeated_guest = 1
else:
    repeated_guest = 0

# Widget để chọn phân khúc khách hàng
market_segment_type = st.sidebar.selectbox('Phân khúc khách hàng', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])

# Chuyển đổi lựa chọn sang giá trị tương ứng
market_segment_map = {
    'Offline': 0.0,
    'Online': 0.25,
    'Corporate': 0.5,
    'Aviation': 0.75,
    'Complementary': 1.0
}
market_segment_type_value = market_segment_map[market_segment_type]

# Widget để nhập giá phòng trung bình trong ngày
avg_price_per_room = st.sidebar.number_input('Giá phòng trung bình trong ngày', min_value=0, step=1)

# Hiển thị các widget còn lại với định dạng số
lead_time = st.sidebar.number_input('Số ngày khách hàng đặt phòng', min_value=18, max_value=100, step=1)
no_of_previous_cancellations = st.sidebar.number_input('Số lần đặt chỗ khách hàng đã hủy trước đó', min_value=0, step=1)
no_of_previous_bookings_not_canceled = st.sidebar.number_input('Số lượng đặt chỗ khách hàng không hủy trước', min_value=0, step=1)
no_of_special_requests = st.sidebar.number_input('Số lượng yêu cầu dịch vụ đặc biệt', min_value=0, step=1)

# Tạo DataFrame từ dữ liệu người dùng
# Khởi tạo các scaler cho từng đặc trưng
scaler_required_car_parking_space = MinMaxScaler(feature_range=(0, 1))
scaler_repeated_guest = MinMaxScaler(feature_range=(0, 1))
scaler_lead_time = MinMaxScaler(feature_range=(0, 1))
scaler_market_segment_type = MinMaxScaler(feature_range=(0, 1))
scaler_no_of_previous_cancellations = MinMaxScaler(feature_range=(0, 1))
scaler_no_of_previous_bookings_not_canceled = MinMaxScaler(feature_range=(0, 1))
scaler_avg_price_per_room = MinMaxScaler(feature_range=(0, 1))
scaler_no_of_special_requests = MinMaxScaler(feature_range=(0, 1))

# Giả sử các giá trị cần dự đoán đã được lưu vào các biến riêng biệt
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

# Chuyển đổi từng đặc trưng thành dạng mảng numpy và áp dụng MinMaxScaler
scaled_required_car_parking_space = scaler_required_car_parking_space.fit_transform(np.array(data['required_car_parking_space']).reshape(-1, 1))
scaled_repeated_guest = scaler_repeated_guest.fit_transform(np.array(data['repeated_guest']).reshape(-1, 1))
scaled_lead_time = scaler_lead_time.fit_transform(np.array(data['lead_time']).reshape(-1, 1))
scaled_market_segment_type = scaler_market_segment_type.fit_transform(np.array(data['market_segment_type']).reshape(-1, 1))
scaled_no_of_previous_cancellations = scaler_no_of_previous_cancellations.fit_transform(np.array(data['no_of_previous_cancellations']).reshape(-1, 1))
scaled_no_of_previous_bookings_not_canceled = scaler_no_of_previous_bookings_not_canceled.fit_transform(np.array(data['no_of_previous_bookings_not_canceled']).reshape(-1, 1))
scaled_avg_price_per_room = scaler_avg_price_per_room.fit_transform(np.array(data['avg_price_per_room']).reshape(-1, 1))
scaled_no_of_special_requests = scaler_no_of_special_requests.fit_transform(np.array(data['no_of_special_requests']).reshape(-1, 1))

# Kết hợp tất cả các đặc trưng đã chuẩn hóa thành một mảng numpy
data = np.hstack([
    scaled_required_car_parking_space,
    scaled_repeated_guest,
    scaled_lead_time,
    scaled_market_segment_type,
    scaled_no_of_previous_cancellations,
    scaled_no_of_previous_bookings_not_canceled,
    scaled_avg_price_per_room,
    scaled_no_of_special_requests
])






# Dự đoán trên dữ liệu đã chuẩn hóa
prediction = rfc.predict(scaled_required_car_parking_space,scaled_repeated_guest,scaled_lead_time,scaled_market_segment_type,scaled_no_of_previous_cancellations,scaled_no_of_previous_bookings_not_canceled,scaled_avg_price_per_room,scaled_no_of_special_requests)

# Hiển thị kết quả dự đoán
st.write('## Kết quả dự đoán:')
st.write('Khách hàng sẽ rời bỏ khách sạn' if prediction[0] == 1 else 'Khách hàng sẽ không rời bỏ khách sạn')
