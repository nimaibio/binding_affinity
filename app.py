import os
from flask import Flask, request, redirect, url_for, render_template,session, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle
import uuid
from features_functions import *


ALLOWED_EXTENSIONS = set(['pdb','sdf'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# engine=sqlalchemy.create_engine("mysql+pymysql://root:12345@localhost/login")
# conn=engine.connect()

UPLOAD_FOLDER_INPUT = os.path.join(os.getcwd(),'static\\input\\')
UPLOAD_FOLDER_OUTPUT = os.path.join(os.getcwd(),'static\\output\\')


app = Flask(__name__,static_folder='static')
app.secret_key='mohit'
app.config['UPLOAD_FOLDER_INPUT'] = UPLOAD_FOLDER_INPUT
app.config['UPLOAD_FOLDER_OUTPUT'] = UPLOAD_FOLDER_OUTPUT

# home route
@app.route('/', methods=['POST', 'GET'])
def basic():
    return render_template('about_project.html')

# about team route
@app.route('/team', methods=['POST', 'GET'])
def help_render():
    return render_template('about_us.html')

# about team route
@app.route('/help', methods=['POST', 'GET'])
def team_render():
    return render_template('help.html')

# virtual screening route
@app.route('/vs', methods=['GET', 'POST'])
def home():
    try:
        dpnp_selected=request.form['DPNP']
        print(dpnp_selected)
        model_name=request.form['BA_model']
        print(model_name)

        if(dpnp_selected=="No"):

            file_sdf = request.files['file_ligand']
            if (allowed_file(file_sdf.filename)):
                filename = secure_filename(file_sdf.filename)
                # Append a unique identifier to avoid overwriting
                unique_filename_sdf = f"{uuid.uuid4()}_{filename}"
                print(unique_filename_sdf)
                file_sdf.save(os.path.join(app.config['UPLOAD_FOLDER_INPUT'], unique_filename_sdf))
            else:
                return render_template('input_new.html', error='Please select a SDF of PDB file only', ex=True)
            sdf_file_path=os.path.join(app.config['UPLOAD_FOLDER_INPUT'], unique_filename_sdf)

        file_pdb = request.files['file_protein']

        if (allowed_file(file_pdb.filename)):
            filename = secure_filename(file_pdb.filename)
            # Append a unique identifier to avoid overwriting
            unique_filename_pdb = f"{uuid.uuid4()}_{filename}"
            print(unique_filename_pdb)
            file_pdb.save(os.path.join(app.config['UPLOAD_FOLDER_INPUT'], unique_filename_pdb))
        else:
            return render_template('input_new.html', error='Please select a SDF of PDB file only', ex=True)
        pdb_file_path = os.path.join(app.config['UPLOAD_FOLDER_INPUT'], unique_filename_pdb)
        if(dpnp_selected=="Yes"):
            ligand_features=pd.read_csv('dpnp_ligand_features.csv')
        else:
            # read sdf and pdb files and get the features
            ligand_features = process_sdf_folder(sdf_file_path)
        protein_3d_features = process_pocket_files(pdb_file_path)
        protein_2d_features = get_2d_protein_features(pdb_file_path)
        protein_features = pd.concat([protein_3d_features, protein_2d_features], axis=1)
        all_features = pd.concat([ligand_features, protein_features], axis=1)
        mol_name = all_features['Molecule Name'].astype(str)
        all_features = all_features.drop(columns=['Molecule Name'])
        print(all_features.columns)
        print(all_features.shape)

        if(model_name=="Random Forest"):
            model=pickle.load(open("random_forest_model.sav", "rb"))
        elif(model_name=="Gradient Boosting"):
            model = pickle.load(open("gradient_boosting_model.sav", "rb"))

        elif (model_name == "XG"):
            model = pickle.load(open("xgb_model.pkl", "rb"))

        predictions=model.predict(all_features.values)
        predictions=np.round(predictions, 3)
        predictions=-predictions
        predictions_df=pd.DataFrame()
        predictions_df['Compound Name/PubChem ID']=mol_name
        predictions_df['Binding_Affintiy']=predictions
        predictions_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER_OUTPUT'], unique_filename_pdb+'.csv'))
        print(predictions)
        return render_template('output.html',predictions=predictions_df.values,file_name=unique_filename_pdb+'.csv')
    except Exception as e:
        error='Please select a SDF or PDB file only!'
        print(e)
        return render_template('input_new.html',error=error,ex=True)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_OUTPUT'], filename, as_attachment=True)


@app.route('/download_sdf')
def download_sdf_file():
    return send_from_directory('static', '1a1e_ligand.sdf', as_attachment=True)

@app.route('/download_pdb')
def download_pdb_file():
    return send_from_directory('static', '1a1e_pocket.pdb', as_attachment=True)

@app.route('/download_active_site_file')
def download_active_site_file():
    return send_from_directory('static', '1a1e_protein.pdb', as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True,)


