import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pandas as pd
from flask import Flask, request, render_template #for modifying html template with python output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
np.set_printoptions(suppress=True) #suppressing scientific notation
pd.set_option('display.max_colwidth', None) # Display the DataFrame with the long string

model = load_model('model_predict_DOOR_unscaled3_FINAL_reduced.h5', compile=False)
model.compile()

#specifying class names for Y_combined
class_names = ["Survival; adequate clinical response; no severe adverse events, and no recurrent CDI", 
               "Survival; adequate clinical response; ICU transfer due to sepsis without need for pressors or surgery", 
               "Survival; inadequate clinical response; ICU transfer due to sepsis requiring pressors, without surgery", 
               "Survival; inadequate clinical response; need for surgery (colectomy or ileostomy) for CDI, without death", 
               "Death (in-hospital, CDI-attributable)",
               "No severe outcomes of index infection but subsequent recurrent CDI",
              "Severe adverse outcome plus recurrent infection"]

# def all_prediction_results(predictions):
#     data = []
#     [data.append({'Outcome': class_names[i], 'Probability':round(predictions[0][i],4)}) for i in range(7)]
#     return pd.DataFrame(data).reset_index(drop=True)

#pandas styling function
def highlight_row(s, row_index, color='bold'):
    return ['font-weight: %s' % color if i==row_index else '' for i in range(len(s))]

expected_probabilities = [1162/1660, 29/1660, 10/1660, 4/1660, 88/1660, 345/1660, 22/1660]
expected_probabilities = [ round(i*100, 2) for i in expected_probabilities ]
expected_all_severe_outcomes = (expected_probabilities[1]+expected_probabilities[2]+expected_probabilities[3]+expected_probabilities[4]+expected_probabilities[6])
expected_all_recurrence = (expected_probabilities[5]+expected_probabilities[6])

def all_prediction_results(predictions):
    data = []
    [data.append({'CDI-Attributable Outcome': class_names[i], 'Probability (%)':round(predictions[0][i]*100,3), 'Expected (based on historical UVA cases)':round(expected_probabilities[i],3)}) for i in range(7)]
    #adding a blank row..
    data.append({'CDI-Attributable Outcome': '', 'Probability (%)':0})   
    data.append({'CDI-Attributable Outcome': 'Any Recurrent Infection', 'Probability (%)':(data[5]['Probability (%)']+data[6]['Probability (%)']), 'Expected (based on historical UVA cases)':expected_all_recurrence})
    data.append({'CDI-Attributable Outcome': 'Any CDI-Attributable Severe Outcome (ICU, Surgery, Death)', 'Probability (%)':(data[1]['Probability (%)']+data[2]['Probability (%)']+data[3]['Probability (%)']+data[4]['Probability (%)']+data[6]['Probability (%)']), 'Expected (based on historical UVA cases)':expected_all_severe_outcomes})
    data = pd.DataFrame(data).reset_index(drop=True)
    data['Predicted/Expected'] = data['Probability (%)'] /data['Expected (based on historical UVA cases)'] 
    data['Probability (%)'] = data['Probability (%)'].astype('float64', errors='ignore').round(3).astype(str)
    data['Probability (%)'][7] = ''
    data['Expected (based on historical UVA cases)'] = data['Expected (based on historical UVA cases)'].astype('float64', errors='ignore').round(2).astype(str)
    data['Expected (based on historical UVA cases)'][7] = ''
    data['Predicted/Expected'] = data['Predicted/Expected'].astype('float64', errors='ignore').round(3).astype(str).apply(lambda x: "{}{}".format(x, 'x'))
    data['Predicted/Expected'][7] = ''
    data = data.style.apply(lambda s: highlight_row(s, 8)).apply(lambda s: highlight_row(s, 9)).hide(axis='index').set_table_styles([{'selector':'th',
                            'props':[('word-wrap', ' break-word'),
                                     ('max-width','130px'),
                                     ( 'text-align', 'left')
                                    ]
                           }])
    return data
    
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
#gathering continuous inputs...
        age = request.form['age']
        recurrence_number = request.form['recurrence_number']
        previous_hospital_duration = request.form['previous_hospital_duration']
        antibiotic_days = request.form['antibiotic_days']
        pcr_ct = request.form['pcr_ct']

#interpreting checkboxes...
        if request.form.get('fever'):
            fever = 1
        else:
            fever = 0  
        if request.form.get('hypotension'): #use 'get' because it won't exist if unchecked
            hypotension = 1
        else:
            hypotension = 0     
        if request.form.get('pressors'):
            pressors = 1
        else:
            pressors = 0 
        if request.form.get('wbc_greater_15'):
            wbc_greater_15 = 1
        else:
            wbc_greater_15 = 0   
        if request.form.get('creatinine_greater_1_5'):
            creatinine_greater_1_5 = 1
        else:
            creatinine_greater_1_5 = 0   
        if request.form.get('lactate_greater_1_9'):
            lactate_greater_1_9 = 1
        else:
            lactate_greater_1_9 = 0  

#encoding the one-hot coding...        
        if request.form['onset'] == 'community_onset':
            community_onset_value = 1
            community_onset_healthcare_associated_value = 0
            hospital_onset_value = 0
        elif request.form['onset'] == 'community_onset_healthcare_associated':
            community_onset_value = 0
            community_onset_healthcare_associated_value = 1
            hospital_onset_value = 0
        elif request.form['onset'] == 'hospital_onset':
            community_onset_value = 0
            community_onset_healthcare_associated_value = 0
            hospital_onset_value = 1
        else:
            print("No onset selection found.")

        if request.form['therapy'] == 'vancomycin_monotherapy':
            vancomycin_monotherapy_value = 1
            fidaxomicin_monotherapy_value = 0
            metronidazole_monotherapy_value = 0
            dual_therapy_value = 0
        elif request.form['therapy'] == 'fidaxomicin_monotherapy':
            vancomycin_monotherapy_value = 0
            fidaxomicin_monotherapy_value = 1
            metronidazole_monotherapy_value = 0
            dual_therapy_value = 0
        elif request.form['therapy'] == 'metronidazole_monotherapy':
            vancomycin_monotherapy_value = 0
            fidaxomicin_monotherapy_value = 0
            metronidazole_monotherapy_value = 1
            dual_therapy_value = 0
        elif request.form['therapy'] == 'dual_therapy':
            vancomycin_monotherapy_value = 0
            fidaxomicin_monotherapy_value = 0
            metronidazole_monotherapy_value = 0
            dual_therapy_value = 1
        else:
            print("No onset selection found.")

        pred = model.predict(np.array([[int(age), int(recurrence_number), int(pressors),int(hypotension), int(previous_hospital_duration), int(wbc_greater_15),int(creatinine_greater_1_5), int(lactate_greater_1_9), 
                                        int(fever),np.float64(pcr_ct), int(antibiotic_days), int(community_onset_value),int(community_onset_healthcare_associated_value), int(hospital_onset_value), int(vancomycin_monotherapy_value),  int(fidaxomicin_monotherapy_value), int(metronidazole_monotherapy_value), 
                                        int(dual_therapy_value)],]))
        return render_template('index7.html', pred=all_prediction_results(pred).to_html(index=False, index_names=False,  classes='table table-striped table-hover', header = "true", justify = "left")) 
    
    return render_template('index7.html')

if __name__ == '__main__':
    app.run(debug=True)