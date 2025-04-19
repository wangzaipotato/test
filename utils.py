import pandas as pd
import joblib
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import ChatPromptTemplate
import os



def fill_info(info, Dyslipidemia=None, Liver_disease=None,Kidney_disease=None,stomach=None,asthma=None,Heart_attack=None,lung=None,arthritis=None,emotional=None,SBP=None, FG=None,  age=None,
       gender=None,marital=None,urban_nbs=None, reg=None, edu=None,sleep=None,drinking=None, pain=None,disability=None,
       muscle_mass=None, IADL=None, BADL=None, met_ca=None,
       N32=None, wusleep=None,  smoking=None, self_health=None,satification=None, low_Grip=None):
    
    if Dyslipidemia:
        if Dyslipidemia == 'Yes':
            Dyslipidemia = 1
        else:
            Dyslipidemia = 0
        info.loc[0, 'Dyslipidemia'] = Dyslipidemia   
    if Liver_disease:
        if Liver_disease == 'Yes':
            Liver_disease = 1
        else:
            Liver_disease = 0
        info.loc[0, 'Liver_disease'] = Liver_disease
    if Kidney_disease:
        if Kidney_disease == 'Yes':
            Kidney_disease = 1
        else:
            Kidney_disease = 0
        info.loc[0, 'Kidney_disease'] = Kidney_disease        
    if stomach:
        if stomach == 'Yes':
            stomach = 1
        else:
            stomach = 0
        info.loc[0, 'stomach'] = stomach 
    if asthma:
        if asthma == 'Yes':
            asthma = 1
        else:
            asthma = 0
        info.loc[0, 'asthma'] = asthma 
    if Heart_attack:
        if Heart_attack == 'Yes':
            Heart_attack = 1
        else:
            Heart_attack = 0
        info.loc[0, 'Heart_attack'] = Heart_attack 
    if lung:
        if lung == 'Yes':
            lung = 1
        else:
            lung = 0
        info.loc[0, 'lung'] = lung
    if arthritis:
        if arthritis == 'Yes':
            arthritis = 1
        else:
            arthritis = 0
        info.loc[0, 'arthritis'] = arthritis
    if emotional:
        if emotional == 'Yes':
            emotional = 1
        else:
            emotional = 0
        info.loc[0, 'emotional'] = emotional
    if pain:
        if pain == 'Yes':
            pain = 1
        else:
            pain = 0
        # info.loc[0, 'pain'] = pain
    if disability:
        if disability == 'Yes':
            disability = 1
        else:
            disability = 0
        info.loc[0, 'disability'] = disability
     
    if SBP:
        info.loc[0, 'SBP'] = SBP
    if FG:
        info.loc[0, 'FG'] = FG
    if age:
        info.loc[0, 'age'] = age
    if gender:
        if gender == 'Female':
            gender1 = 0
        else:
            gender1 = 1
        info.loc[0, 'gender'] = gender1
    if marital:
        if marital == '未婚':
            marital = 0
        else:
            marital1 = 1
        info.loc[0, 'marital'] = marital1    
    if urban_nbs:
        if urban_nbs == '乡村':
            urban_nbs = 0
        else:
            urban_nbs1 = 1
        info.loc[0, 'urban_nbs'] = urban_nbs1      
    if reg:
        lst = ['中部地区', '东部地区', '东北地区', '西部地区']
        reg1 = lst.index(reg)
        info.loc[0, 'reg'] = reg1
    if edu:
        lst = ['小学及以下', '高中', '大专及以上']
        edu = lst.index(edu)
        info.loc[0, 'edu'] = edu
    if drinking:
        lst = ['No drinking','drinking<1', 'drinking>1']
        drinking1 = lst.index(drinking)
        info.loc[0, 'drinking'] = drinking1
    if muscle_mass:
        info.loc[0, 'muscle_mass'] = muscle_mass
    if IADL:
        if IADL == 'Yes':
            IADL1=0
        else: 
            IADL1=1
        info.loc[0, 'IADL'] = IADL1
    if BADL:
        if BADL == 'Yes':
            BADL1=0
        else: 
            BADL1=1
        info.loc[0, 'BADL'] = BADL1
    if met_ca:
        lst = ['Inactive',  'active', 'highly active']
        met_ca1 = lst.index(met_ca)
        info.loc[0, 'met_ca'] = met_ca1
    if N32:
        info.loc[0, 'N32'] = N32
    if sleep:
        lst = ['<6h', '6-8h', '>8h']
        sleep1 = lst.index(sleep)
        info.loc[0, 'sleep'] = sleep1    
    if wusleep:
        lst = ['无', '<30分钟', '30-90分钟', '>90分钟']
        wusleep1 = lst.index(wusleep)
        info.loc[0, 'wusleep'] = wusleep1
    if smoking:
        lst = ['No smoking', 'Smoking']
        smoking = lst.index(smoking)
        info.loc[0, 'smoking'] = smoking
    if low_Grip:
        lst = ['Yes', 'No']
        low_Grip1 = lst.index(low_Grip)
        info.loc[0, 'low_Grip'] = low_Grip1   
    if self_health:
        lst = ['差', '良', '好']
        self_health = lst.index(self_health)
        info.loc[0, 'self_health'] = self_health
    if satification:
        lst = ['差', '良', '好']
        satification = lst.index(satification)
        info.loc[0, 'satification'] = satification

    feature_order = ['age', 'N32', 'FG', 'SBP', 'muscle_mass', 'sleep', 'gender', 'marital', 'edu', 
                 'urban_nbs', 'smoking', 'drinking', 'Dyslipidemia', 'Liver_disease', 'Kidney_disease', 
                 'stomach', 'Heart_attack', 'lung', 'self_health', 'satification', 'wusleep', 'emotional', 
                 'arthritis', 'asthma', 'met_ca', 'IADL', 'BADL', 'disability', 'reg', 'low_Grip']
    info = info[feature_order]
    
    return info

def standarscaler(df):
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    StandardScaler = joblib.load('StandardScaler-yiyu.pkl')
    num_features = df.loc[:, ['age', 'N32', 'FG', 'SBP', 'muscle_mass', 'sleep']]
    num_features_transformed = pd.DataFrame(StandardScaler.transform(num_features), columns=StandardScaler.get_feature_names_out())

    for feature in ['age', 'N32', 'FG', 'SBP', 'muscle_mass', 'sleep']:
        df.loc[:, feature] = num_features_transformed.loc[:,feature]
    
    df = df.astype(float)

    return df

def ai(yiyu_risk, yiyu_shap):
    import os
    api_key = os.getenv('ZHIPUAI_API_KEY')
    model = ChatZhipuAI(model='glm-4-flash', api_key=api_key, temperature=1)

    prompt_template = ChatPromptTemplate([
        ('system',"""你是一个机器学习和医学专家，现在我有一个基于生活方式的10年抑郁症风险的预测模型，
         我会将该个体的预测风险以及每个变量对应的shap值给你，请你对其进行分析，提出针对于该个体的合理化建议，尤其是对于生活方式方面的。
         回答的内容用markdown格式书写，尽量美观。 回答使用中文。
         
         --------
         回答的格式为，
         标题
         一、概述
         二、评估结果 （这部分简要描述即可）
         三、 建议 （多聚焦于生活方式）

         --------

         其特征分别为：
         1.是否患有高血脂：1.有， 0.无',
         2. 是否患有肝脏疾病：1.有， 0.无',
         3.是否患有肾脏疾病：1.有， 0.无',
         4.是否患有胃部疾病：1.有， 0.无',
         5.是否患有哮喘：1.有， 0.无',
         6.是否患有心脏疾病：1.有， 0.无',
         7.是否患有肺部疾病：1.有， 0.无',
         8.是否患有关节炎：1.有， 0.无',
         9.是否患有情感疾病：1.有， 0.无',
         10.是否有身体疼痛：1.有， 0.无',
         11.是否有残疾：1.有， 0.无',
         12.收缩压；
         13.空腹血糖值：mmol/L;
         14.年龄；
         15.性别：1.男， 0女；
         16.婚姻状况：1.已婚， 2未婚；
         17.居住地：1.城市，2乡村；
         18.地区： 1.Central，2.Eastern，  3.Noeastern，  4.Western；
         19.受教育水平：1.小学及以下，2.高中，3.大专及以上；
         20. 饮酒情况：1.不饮酒 ，2.少于1个月1次，3.大于1个月1次；
         21.肌肉质量指数；
         22.工具性日常生活活动能力：  1.有，0.无；
         23.基本生活活动能力： 1.有，0.无；
         24.体力活动强度：0.不活跃，1.活跃，2.高度活跃；
         25.认知能力得分；
         26.睡眠时间： 1.'< 6h',2.'6-8h',3.'>8h';
         27.午睡时间：0.无，1.'<30分钟'，2.'30-90分钟'，3.'>90分钟'；
         28.吸烟情况：0.不吸烟，1.吸烟；
         29.是否有握力较低： 0.无，1.有；
         30.自评健康状况： 0.差，1.良，2.优；
         31.生活满意度：0.差，1.良，3.优；
         """),

        ('user',"""
        该个体的10年抑郁症风险为{yiyu_risk}, 在预测全因死亡方面，其特征所对应的shap值为{yiyu_shap}。
""")
    ])

    chain = prompt_template | model
    response = chain.invoke({
        'yiyu_risk':yiyu_risk,
        'yiyu_shap': yiyu_shap,
    })

    return response.content
