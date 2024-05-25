import numpy as np
import pickle
import streamlit as st 

# Load the model
model_path = 'predictor_saved.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

# Creating a function for Prediction
def completionrate_prediction(input_data):
    # Changing the input_data to numpy array and converting to float
    input_data_as_float = [float(x) for x in input_data]
    input_data_as_numpy_array = np.asarray(input_data_as_float)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    # Giving a title
    st.title('School Completion Rate Prediction Application')

    # Getting the input data from the user
    pspmlsrbs = st.text_input('pspmlsrbs', '0.0')
    gip = st.text_input('gip', '0.0')
    ofsalssa = st.text_input('ofsalssa', '0.0')
    ppp_2_cg = st.text_input('ppp_2_cg', '0.0')
    pc_24_59_thlp = st.text_input('pc_24_59_thlp', '0.0')
    nerece = st.text_input('nerece', '0.0')
    nerpp = st.text_input('nerpp', '0.0')
    pyalmlpdl = st.text_input('pyalmlpdl', '0.0')
    earcpehp_twentyfiveplus = st.text_input('earcpehp_twentyfiveplus', '0.0')
    ptmrepeagpi = st.text_input('ptmrepeagpi', '0.0')
    geeduv = st.text_input('geeduv', '0.0')
    pcypepfm = st.text_input('pcypepfm', '0.0')
    pcypelsfr = st.text_input('pcypelsfr', '0.0')

    # Code for Prediction
    if st.button('Predict school completion rate'):
        input_data = [pspmlsrbs, gip, ofsalssa, ppp_2_cg, pc_24_59_thlp, nerece, nerpp, pyalmlpdl, earcpehp_twentyfiveplus,
                       ptmrepeagpi, geeduv, pcypepfm, pcypelsfr]
        prediction = completionrate_prediction(input_data)
        st.success(f'Predicted school completion rate is : {prediction}')

if __name__ == '__main__':
    main()
