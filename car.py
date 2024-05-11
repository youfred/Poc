import streamlit as st
import pandas as pd 
from datetime import datetime

car_recall = pd.read_csv("car_recall.csv")
car_recall['리콜개시일'] = pd.to_datetime(car_recall['리콜개시일'])
car_recall["리콜개시달"] = car_recall["리콜개시일"].dt.month

st.title(":car: 자동차 리콜 현황 :car:")

option = st.selectbox(
    "먼저 리콜 현황을 보고 싶은 회사를 골라주세요",
    car_recall['제작자'].unique())

with st.container(border= True): 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("차종별 리콜 순위")
        filter = (car_recall["제작자"] == option) 
        df1 = car_recall[filter]["차명"].value_counts().sort_values(ascending=False)  
        st.dataframe(data=df1, use_container_width=True)
    with col2:
        filter = (car_recall["제작자"] == option) 
        df = car_recall[filter].groupby("리콜개시달")
        st.subheader("월별 리콜 횟수(2022년 기준)")
        recalls_by_month = df.size()
        st.bar_chart(recalls_by_month)

option1 = st.selectbox(
    "고르신 회사 안에서 보고 싶은 차종을 골라주세요",
    car_recall[car_recall["제작자"] == option]['차명'].unique(),    
    index=None)



st.subheader("회사 차종별 리콜 상세 내용 및 생산기간")
with st.container(border= True): 
    filtered_data = car_recall[(car_recall["제작자"] == option) & (car_recall["차명"] == option1)][["차명", "리콜사유"]]  
    pd.set_option('display.max_colwidth', None)
    st.table(data=filtered_data)
    filter = (car_recall["제작자"] == option) & (car_recall["차명"] == option1) 
    df2 = car_recall[filter][["생산기간(부터)", "생산기간(까지)"]]
    st.dataframe(data=df2, use_container_width=True, hide_index= True)  



