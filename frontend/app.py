import streamlit as st
from joblib import load

st.set_page_config(page_title='mushroom prediction')

dialog_options = {
    'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'],
    'cap-surface': ['fibrous','grooves','scaly','smooth'],
    'cap-color': ['brown','buff','cinnamon','gray','green','pink','purple','red','white','yellow'],
    'bruises': ['bruises','no'],
    'odor': ['almond','anise','creosote','fishy','foul','musty','none','pungent','spicy'],
    'gill-attachment': ['attached', 'descending', 'free', 'notched'],
    'gill-spacing': ['close', 'crowded', 'distant'],
    'gill-size': ['broad', 'narrow'],
    'gill-color': ['black','brown','buff','chocolate','gray','green','orange','pink','purple','red','white','yellow'],
    'stalk-shape': ['enlarging','tapering'],
    'stalk-root': ['bulbous','club','cup','equal','rhizomorphs','rooted','missing'],
    'stalk-surface-above-ring': ['fibrous','scaly','silky','smooth'],
    'stalk-surface-below-ring': ['fibrous','scaly','silky','smooth'],
    'stalk-color-above-ring': ['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],
    'stalk-color-below-ring': ['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],
    'veil-type': ['partial','universal'],
    'veil-color': ['brown','orange','white','yellow'],
    'ring-number': ['none','one','two'],
    'ring-type': ['cobwebby','evanescent','flaring','large','none','pendant','sheathing','zone'],
    'spore-print-color': ['black','brown','buff','chocolate','green','orange','purple','white','yellow'],
    'population': ['abundant','clustered','numerous','scattered','several','solitary'],
    'habitat': ['grasses','leaves','meadows','paths','urban','waste','woods'],
}

def intro():
    import streamlit as st
    st.markdown(
"""
# Welcome to ECS 171 Project: Mushroom Prediction

You can choose the model on the left. We have

1. logistic
2. multi-layer perception classifier
3. bayesian classifier

""")
    
encoder = load("./frontend/encoder.joblib")

def model_demo():
    import streamlit as st
    left_column, right_column = st.columns(2)

    choices = dict()
    for i, v in dialog_options.items():
        choices[i] = left_column.selectbox(i, v)

    with right_column:
        choices_arr = encoder.transform([list(choices.values())])
        predicted = "edible" if models[model_name].predict(choices_arr)[0] >= 0.5 else "poisonous"
        st.write('Predicted value: ', predicted)
        st.write('You selected: ', choices)
        st.write('Encoded input: ', choices_arr[0])

page_names_to_funcs = {
    "Intro": intro,
    "Model demo": model_demo,
}


lr = None
mlp = None
nb = None

models = {
    "logistic": load("./frontend/lg.joblib"),
    "mlp": load("./frontend/mlp.joblib"),
    "naive bayes": load("./frontend/nb.joblib"),
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
model_name = st.sidebar.selectbox("Choose a model", models.keys())
page_names_to_funcs[demo_name]()