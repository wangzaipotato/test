def yiyudisease(info):
    import joblib
    clf = joblib.load('best_XGC.pkl')

    death_proba = clf.predict_proba(info)[0][0]

    return death_proba