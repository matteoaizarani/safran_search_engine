import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt 
import os


df = pd.read_excel("data_test1.xlsx")
#overview of the excel file 
df.head(1)

#creating a copy of the original file to modify it 
df2 = df.copy()
df2.head(1)

#keeping the product name column before removing it 
target = df2['Product Name']

#removing the columns where we have missing values or where Yann still need to give infos 
df2 = df2.drop(['Product Name', 'Export control', 'Operating Temperature', 
                'Run-up time', 'Shocks', 'Vibration (random)', 'Power supply', 'TSF accuracy at 20°C (PPM)', 
                'G² sensitive drift (°/h/g2)', 'Size (cm3)', 'Minimum operating temperature', 'Weight (g)', 
                'Minimum delivery time (months)','Maximum operating temperature', 'Technology', 'Output', 
                'Number of axis', 'Scale factor accuracy (PPM)', 'G sensitive drift (°/h/g)', 
                'Bias over temperature (°/h rms)'],
               axis = 1)
#encoding the categorical variables because our model struggles with non numerical data 
#one =OneHotEncoder(sparse_output= False)
#encoded_cols = one.fit_transform(df2[["Technology", "Output"]])
#encoded_df = pd.DataFrame(encoded_cols, columns=one.get_feature_names_out(['Technology', 'Output']))

#df2 = pd.concat([df2.drop(['Technology', 'Output'], axis=1), encoded_df], axis=1)

df2.head()

#feeding our scaler function our dataset so it can normalise our data and reduce biases (for example:
#scale factor = 500 but bias over temperature = 10 the difference is too big so we will have to normalise our data.
scaler = MinMaxScaler()
df2_norm = pd.DataFrame(scaler.fit_transform(df2), columns = df2.columns) # we have to transform the np.array into a dataframe to use it later on linalg

#same principle as before, we are normalizing our client input to efficiently compare it to our normalized dataset 
def normalize_client_input(client_input_dict):
    input_df = pd.DataFrame([client_input_dict], columns= df2.columns) # Convert dict to DataFrame (client input is a dict)
    #input_df = input_df[df2.columns] #we could use this line if there are missing keys in our client dict but its not the case here so we are using , columns= df2.columns)
    input_scaled = scaler.transform(input_df)     # Apply scaler for normalization 
    return input_scaled, input_df

#client input = the specs the client is searching for when wanting a new product

def main():
    st.title("Gyroscope finder")
    st.header("Enter client specifications")
    client_input = {
      'Measurement range (°/s)': st.number_input('Measurement range (°/s)', min_value=0.0, max_value=440.0, value=0.0),
      'In run\nBias instability (Allan variance) (°/h)': st.number_input('In run Bias instability (°/h)', min_value=0.0, max_value=1.8, value=0.0),
      'Bandwidth (Hz)': st.number_input('Bandwidth (Hz)', min_value = 0.0 , max_value= 262.0),
      'Angle random walk (°/√hr)': st.number_input('Angle random walk (°/√hr)', min_value=0.0000, max_value=0.17, 
                                                   value=0.0000, step=0.001, format="%.4f")
    
      } 
    
    st.header("🎚️ Feature Importance Weights (Total = 100%)")
    bandwidth_weight = st.slider("Importance of Bandwidth", 0, 100, 25)
    mr_weight = st.slider("Importance of Measurement range", 0, 100, 25)
    bias_weight = st.slider("Importance of Bias Instability", 0, 100, 25)
    angle_weight = st.slider("Importance of Angle Random walk", 0, 100, 25)
    
    total_weight = bandwidth_weight + mr_weight + angle_weight + bias_weight
    if total_weight == 0:
        st.warning("assign weight > 0")
        return
    elif total_weight != 100:
        st.warning("Please ensure that the total weight adds up to 100%")
        st.stop()
    
    
    #testing with 100, if not good try again with total_weight 
    weights = {
        'Measurement range (°/s)': mr_weight / 100,
        'In run\nBias instability (Allan variance) (°/h)': bias_weight / 100,
        'Bandwidth (Hz)': bandwidth_weight / 100,
        'Angle random walk (°/√hr)': angle_weight / 100
        
    }

    
    if st.button('Find the products that are the closest to my specifications', type = 'primary'):
        nci, input_df = normalize_client_input(client_input) #applying the normalisation function to our client input
        
        weight_array = np.array([weights[col] if col in weights else 0.0 for col in df2.columns])
        diff = df2_norm.values - nci
        
        # Function for the calculation of the weighted distance 
        # (different from normal euclidien distance calculation because we add the weigt )
        weighted_distance = np.sqrt(np.sum((diff ** 2) * weight_array, axis=1))
        
        #calculating the distance between the real data and the input of the client 
        distance = np.linalg.norm(df2_norm.values - nci, axis = 1)
        max_distance = distance.max() if distance.max() != 0 else 1e-10  # éviter division par zéro
        similarity = 100 * (1 - weighted_distance / max_distance)

        #sorting the top 3 results depending on the distance (the lower the distance the closer the client input is to the actual product)
        result = weighted_distance.argsort()[:3]
        
        
        
        
        st.subheader("Top 3 closest products:")
        pdf_folder = "produits" #where my pdf files are located
        st.write("📂 Contenu de", pdf_folder, ":", os.listdir(pdf_folder))
        for idx in result:
            product_name = target[idx]
            pdf_filename = (f"{product_name}.pdf")
            st.write(f"Matching product {target[idx]} --- similarity score: {similarity[idx]:.2f}%")
            
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as i:
                    st.download_button(label = f"Download PDF file of {product_name}",
                                       data = i,
                                       file_name = f"{product_name}.pdf",
                                       mime= "application/pdf")
            else:
                st.warning(f"PDF not found for product: {product_name}")
            
            
        st.subheader("Similarity score comparison")
        fig, ax = plt.subplots()
        product_names = [target[i] for i in result]
        sim_scores = [similarity[i] for i in result]
        ax.barh(product_names, sim_scores, color='skyblue')
        ax.invert_yaxis()
        ax.set_xlabel("Similarity (%)")
        ax.set_title("Top 3 Matching Products")
        st.pyplot(fig)
        
        #a supprimer si ca marche pas 
        best_match_index = result[0]
        best_match = df.iloc[best_match_index][df2.columns]  # On récupère les colonnes utiles dans le DataFrame original
        client_raw = pd.Series(client_input)  # Utilise les vraies valeurs du client
        client_vs_best = pd.DataFrame({
            "Client Input": client_raw,
            "Best Match": best_match,
            "Difference (abs)": (client_raw - best_match).abs()
            })

        st.subheader("🔍 Feature-by-feature comparison with best match:")
        st.dataframe(client_vs_best.style.format(precision=4))
        
       
        
if __name__ == "__main__":
    main()
    
