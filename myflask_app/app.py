from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
cancer_model = pickle.load(open('cancer_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming data is sent as JSON

    # Extracting data for Diabetes Prediction
    pregnancies = data.get('Pregnancies')
    glucose = data.get('Glucose')
    blood_pressure = data.get('BloodPressure')
    skin_thickness = data.get('SkinThickness')
    insulin = data.get('Insulin')
    bmi = data.get('BMI')
    diabetes_pedigree_function = data.get('DiabetesPedigreeFunction')
    age = data.get('Age')

    # Perform Diabetes prediction
    diab_prediction = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    if diab_prediction[0] == 1:
        diab_diagnosis = 'Patient is diabetic, please consult your doctor immediately!'
    else:
        diab_diagnosis = 'Patient is not diabetic, no need to worry!'

    # Extracting data for Heart Disease Prediction
    age = data.get('Age')
    sex = data.get('Sex')
    cp = data.get('ChestPain')
    trestbps = data.get('RestingBloodPressure')
    chol = data.get('Cholesterol')
    fbs = data.get('FastingBloodSugar')
    restecg = data.get('RestingECG')
    thalach = data.get('MaxHeartRate')
    exang = data.get('ExerciseAngina')
    oldpeak = data.get('Oldpeak')
    slope = data.get('Slope')
    ca = data.get('MajorVessels')
    thal = data.get('Thal')

    # Perform Heart Disease prediction
    heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    if heart_prediction[0] == 1:
        heart_diagnosis = 'Patient has a high risk of Heart Disease, please consult your doctor immediately!'
    else:
        heart_diagnosis = 'Patient has a low risk of Heart Disease, no need to worry!'

    # Extracting data for Breast Cancer Prediction
    concavepoints_mean = data.get('ConcavePoints')
    area_mean = data.get('Area')
    radius_mean = data.get('Radius')
    perimeter_mean = data.get('Perimeter')
    concavity_mean = data.get('Concavity')
    texture_mean = data.get('Texture')
    smoothness_mean = data.get('Smoothness')
    compactness_mean = data.get('Compactness')
    symmetry_mean = data.get('Symmetry')
    fractaldimension_mean = data.get('FractalDimension')

    # Perform Breast Cancer prediction
    cancer_prediction = cancer_model.predict([[concavepoints_mean, area_mean, radius_mean, perimeter_mean, concavity_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractaldimension_mean]])
    if cancer_prediction[0] == 1:
        cancer_diagnosis = 'Patient has a high risk of Cancer Disease, please consult your doctor immediately!'
    else:
        cancer_diagnosis = 'Patient has a low risk of Cancer Disease, no need to worry!'

    # Extracting data for Parkinson's Disease Prediction
    fo = data.get('MDVP:Fo(Hz)')
    fhi = data.get('MDVP:Fhi(Hz)')
    flo = data.get('MDVP:Flo(Hz)')
    jitter_percent = data.get('MDVP:Jitter(%)')
    jitter_abs = data.get('MDVP:Jitter(Abs)')
    rap = data.get('MDVP:RAP')
    ppq = data.get('MDVP:PPQ')
    ddp = data.get('Jitter:DDP')
    shimmer = data.get('MDVP:Shimmer')
    shimmer_db = data.get('MDVP:Shimmer(dB)')
    apq3 = data.get('Shimmer:APQ3')
    apq5 = data.get('Shimmer:APQ5')
    apq = data.get('MDVP:APQ')
    dda = data.get('Shimmer:DDA')
    nhr = data.get('NHR')
    hnr = data.get('HNR')
    rpde = data.get('RPDE')
    dfa = data.get('DFA')
    spread1 = data.get('spread1')
    spread2 = data.get('spread2')
    d2 = data.get('D2')
    ppe = data.get('PPE')

    # Perform Parkinson's Disease prediction
    parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
    if parkinsons_prediction[0] == 1:
        parkinsons_diagnosis = "Patient has a high risk of Parkinson's Disease, please consult your doctor immediately!"
    else:
        parkinsons_diagnosis = "Patient has a low risk of Parkinson's Disease, no need to worry!"

    return jsonify({
        "Diabetes Prediction": diab_diagnosis,
        "Heart Disease Prediction": heart_diagnosis,
        "Breast Cancer Prediction": cancer_diagnosis,
        "Parkinson's Prediction": parkinsons_diagnosis
    })

if __name__ == '_main_':
    app.run(debug=True)