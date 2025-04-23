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

with st.expander('点击填写您的个人基本信息'):
    column1, column2, column3, column4 = st.columns([1,1,1,1])
    with column1:
        st.write('基本信息')
        age = st.number_input('年龄', min_value=45, max_value=100, step=1)
        gender = st.selectbox(label='性别',options=['Male', 'Female'])
        edu = st.selectbox(label='Education level', options=['小学及以下', '高中', '大专及以上'])
        marital = st.selectbox( label='婚姻状况', options=['已婚', '未婚'])
        urban_nbs = st.selectbox( label='居住地', options=['城市', '乡村'])
        reg = st.selectbox( label='地区', options=['中部地区', '东部地区', '东北地区', '西部地区'])
    

    with column2:
        st.write('患病情况')
        Dyslipidemia = st.selectbox(label='血脂异常', options=['Yes', 'No'])
        Liver_disease = st.selectbox(label='肝脏疾病', options=['Yes', 'No'])
        Kidney_disease = st.selectbox(label='肾脏疾病', options=['Yes', 'No'])
        stomach = st.selectbox(label='胃部疾病', options=['Yes', 'No'])
        asthma = st.selectbox(label='哮喘', options=['Yes', 'No'])
        Heart_attack = st.selectbox(label='心脏病', options=['Yes', 'No'])
        lung = st.selectbox(label='肺部疾病', options=['Yes', 'No'])
        arthritis = st.selectbox(label='关节炎', options=['Yes', 'No']) 
        emotional = st.selectbox(label='情感类疾病', options=['Yes', 'No'])    
        pain = st.selectbox(label='身体疼痛', options=['Yes', 'No'])   
        disability = st.selectbox(label='残疾', options=['Yes', 'No'])   


    with column3:
        st.write('实验室检查')
        FG = st.number_input('空腹血糖 (mmol/L)')
        SBP = st.number_input('收缩压 (mmHg)')
        muscle_mass = st.number_input('肌肉质量 ')
        low_Grip = st.selectbox('低握力情况', ['Yes', 'No']) 

    with column4:
        st.write('生活方式')
        N32 = st.number_input('认知得分', min_value=0.0, max_value=100.0, step=1.0)
        sedentary_minutes = st.number_input('久坐时间（分钟）', min_value=0, max_value=1000, step=10)
        met_ca = st.selectbox('身体活动强度', options=['Inactive',  'active', 'highly active'])
        sleep = st.selectbox('睡眠时间', ['<6h', '6-8h', '>8h'])
        wusleep = st.selectbox('午睡 时间', ['无', '<30分钟', '30-90分钟', '>90分钟'])
        drinking = st.selectbox('饮酒情况', ['No drinking','drinking<1', 'drinking>1'])  
        smoking = st.selectbox('吸烟情况', ['No smoking', 'Smoking'])  
        IADL = st.selectbox('日常活动能力', ['No', 'Yes'])  
        BADL = st.selectbox('基本生活活动能力', ['No', 'Yes'])  
        satification = st.selectbox('生活满意度', ['差', '良', '好'])  
        self_health = st.selectbox('自评健康状况', ['差', '良', '好'])  

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
        matplotlib=False,

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
