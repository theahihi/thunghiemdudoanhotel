import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained RandomForest model
model_path = 'dudoanhotel3.pkl'
rfc = pickle.load(open(model_path, 'rb'))

# Title of the application
st.title('Dự đoán sự rời bỏ của khách hàng khách sạn')

# Sidebar for user input
st.sidebar.title('Nhập các thuộc tính để dự đoán')

# Input fields for features
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

# Create a numpy array from user input
input_data = np.array([[required_car_parking_space, repeated_guest, lead_time, market_segment_type_value,
                        no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                        avg_price_per_room, no_of_special_requests]])

# Prediction logic
if st.sidebar.button('Dự đoán'):
    # Apply MinMaxScaler to the input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Make prediction using the loaded model
    prediction = rfc.predict(input_data_scaled.reshape(1, -1))
    
    # Display prediction result
    if prediction[0] == 1:
        st.write('Khách hàng có khả năng rời bỏ khách sạn.')
    else:
        st.write('Khách hàng có khả năng ở lại khách sạn.')
