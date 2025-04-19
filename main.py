from model import yiyudisease
from utils import fill_info, standarscaler, ai
import streamlit as st
import streamlit_shap
import pandas as pd
import shap
import joblib

import shap 

import matplotlib.pyplot as plt

st.divider()

with st.expander('click me to fill your information'):
    column1, column2, column3, column4 = st.columns([1,1,1,1])
    with column1:
        st.write('Demographic characteristic')
        age = st.number_input('age', min_value=45, max_value=100, step=1)
        gender = st.selectbox(label='gender',options=['Male', 'Female'])
        edu = st.selectbox(label='Education level', options=['小学及以下', '高中', '大专及以上'])
        marital = st.selectbox( label='婚姻状况', options=['已婚', '未婚'])
        urban_nbs = st.selectbox( label='居住地', options=['城市', '乡村'])
        reg = st.selectbox( label='地区', options=['中部地区', '东部地区', '东北地区', '西部地区'])
    

    with column2:
        st.write('Medical history')
        Dyslipidemia = st.selectbox(label='Dyslipidemia', options=['Yes', 'No'])
        Liver_disease = st.selectbox(label='Liver_disease', options=['Yes', 'No'])
        Kidney_disease = st.selectbox(label='Kidney_disease', options=['Yes', 'No'])
        stomach = st.selectbox(label='stomach', options=['Yes', 'No'])
        asthma = st.selectbox(label='asthma', options=['Yes', 'No'])
        Heart_attack = st.selectbox(label='Heart_attack', options=['Yes', 'No'])
        lung = st.selectbox(label='lung', options=['Yes', 'No'])
        arthritis = st.selectbox(label='arthritis', options=['Yes', 'No']) 
        emotional = st.selectbox(label='emotional', options=['Yes', 'No'])    
        pain = st.selectbox(label='pain', options=['Yes', 'No'])   
        disability = st.selectbox(label='disability', options=['Yes', 'No'])   


    with column3:
        st.write('Labortary test')
        FG = st.number_input('FG (mmol/L)')
        SBP = st.number_input('SBP (mmhg)')
        muscle_mass = st.number_input(' ')
        low_Grip = st.selectbox('grip status', ['Yes', 'No']) 

    with column4:
        st.write('Lifestyle')
        N32 = st.number_input('N32', min_value=0.0, max_value=100.0, step=1.0)
        sedentary_minutes = st.number_input('Sedentary minutes', min_value=0, max_value=1000, step=10)
        met_ca = st.selectbox('身体活动强度', options=['Inactive',  'active', 'highly active'])
        sleep = st.selectbox('Sleep quality', ['<6h', '6-8h', '>8h'])
        wusleep = st.selectbox('午睡 Sleep quality', ['无', '<30分钟', '30-90分钟', '>90分钟'])
        drinking = st.selectbox('Drinking status', ['No drinking','drinking<1', 'drinking>1'])  
        smoking = st.selectbox('Smoking status', ['No smoking', 'Smoking'])  
        IADL = st.selectbox('IADL', ['No', 'Yes'])  
        BADL = st.selectbox('BADL', ['No', 'Yes'])  
        satification = st.selectbox('satification', ['差', '良', '好'])  
        self_health = st.selectbox('self_health', ['差', '良', '好'])  

prediction_button = st.button('预测抑郁风险')

info = pd.DataFrame(columns=['age', 'N32', 'FG', 'SBP', 'muscle_mass', 'sleep'], data=None)

st.session_state['info'] = info
st.session_state['orinial_info'] = st.session_state['info'] = fill_info(info=info, Dyslipidemia=Dyslipidemia, Liver_disease=Liver_disease,
                                                                        Kidney_disease=Kidney_disease,stomach=stomach,asthma=asthma,Heart_attack=Heart_attack, 
                                                                        lung=lung,arthritis=arthritis,emotional=emotional,SBP=SBP, FG=FG, age=age,
                                                                        gender=gender,marital=marital,urban_nbs=urban_nbs, reg=reg, edu=edu,sleep=sleep, pain=pain,
                                                                        disability=disability,muscle_mass=muscle_mass, IADL=IADL, BADL=BADL, met_ca=met_ca,
                                                                        N32=N32, wusleep=wusleep, drinking=drinking, smoking=smoking, 
                                                                        self_health=self_health,satification=satification, low_Grip=low_Grip)

st.session_state['info'] = standarscaler(df=st.session_state['info'])
# 预测
if prediction_button:
    st.session_state['prediction_button'] = True
    st.write('抑郁风险')
    st.session_state['yiyu_risk'] = round(yiyudisease(info=st.session_state['info']), 2)

 
    st.write('您的十年抑郁风险为 {}'.format(round(st.session_state['yiyu_risk'], 2)))

    yiyu_explainer = shap.TreeExplainer(joblib.load('best_XGC.pkl'))
    yiyu_shap_values = yiyu_explainer.shap_values(st.session_state['info'])
    
    st.session_state['yiyu_shap_values'] = yiyu_shap_values
    
    shap_fig = shap.force_plot(
        yiyu_explainer.expected_value, 
        yiyu_shap_values[0,:], 
        st.session_state['info'].columns.tolist(),
        matplotlib=True,

        )
    streamlit_shap.st_shap(shap_fig)
    
ai_button = st.button('获取Ai建议')
if ai_button:

    if not st.session_state['prediction_button']:
        st.info('Please click on Prediction button first!')
        st.stop()
    with st.spinner('Please wait a moment, suggestions are being generated'):
        response = ai(
            yiyu_risk = st.session_state['yiyu_risk'],
            yiyu_shap=st.session_state['yiyu_shap_values'],
        )
        st.info(response)