from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet,configure_uploads,IMAGES,ALL,DATA
from flask_sqlalchemy import SQLAlchemy






app = Flask(__name__)
Bootstrap(app)

# Configuration

files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadstorage'
configure_uploads(app,files)

import os
import datetime
import time


# Packages
import pandas as pd 
import numpy as np 

#ML Regression Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


#ML  Classifiers Packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
standard_scaler_object = StandardScaler()
labelencoder_object = LabelEncoder()

db = SQLAlchemy(app)
# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'



# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)

# ML Packages
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/datauploads',methods=['GET','POST'])
def datauploads():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		#Quering the target column from the user
		target_column_value = request.form.get('target_column')
		filename=file.filename


		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB',filename))
		fullfile = os.path.join('static/uploadsDB',filename)


		#file.save(os.path.join('static/uploadstorage',filename))
		data_set = pd.read_csv(os.path.join('static/uploadsDB',filename))
		data_set_original = pd.read_csv(os.path.join('static/uploadsDB',filename))
		
		#Data-Set Operation Function
		df_size = data_set.size
		df_shape = data_set.shape
		df_columns = list(data_set.columns)

		#Handling Categorical Features
		
		for column in list(data_set.columns):
			if(data_set[column].dtype == 'O' and data_set[column].nunique()<100):
				data_set[column]=labelencoder_object.fit_transform(data_set[column].fillna('0'))
                
				data_set[column]=pd.get_dummies(data_set[column])
            
			elif(data_set[column].dtype == 'O' and data_set[column].nunique()>100):
				data_set = data_set.drop(column,axis=1)


		for column in list(data_set.columns):
            
			if(data_set[column].isna().sum()!=0):
        	    
				data_set[column] = data_set[column].fillna(data_set[column].mean())

        
		
		df_targetnames = data_set[data_set.columns[-1]].name
		X = data_set.drop(target_column_value,axis = 1)
		Y = data_set[target_column_value]
        #Feature Scaling of X_features
		X = standard_scaler_object.fit_transform(X)
        
		
		seed = 102


		models = []
		models.append(('Logistic Regression:', LogisticRegression()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('Decission Tree: ', DecisionTreeClassifier()))
		models.append(('Naive Bayes', GaussianNB()))
		models.append(('SVM', SVC()))
		

		results = []
		names = []
		allmodels = []
		scoring = 'accuracy'
		for name,model in models:
			kfold = model_selection.KFold(n_splits=10, random_state = seed)
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			#Mean Result of the K-Fold Cross Validation is used for evaluating the model
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			allmodels.append(msg)
			model_results = allmodels
			model_names = names

			

		#Saving all the details in the database

		newfile = FileContents(name=file.filename,data=file.read(),modeldata=msg)
		db.session.add(newfile)
		#db.session.commit()






		

	
		
	
	return render_template('details.html',filename = filename,data_set = data_set,
	       target_column_value = target_column_value,
			df_size = df_size,
			df_shape= df_shape,
			df_columns = df_columns,
			df_targetnames = df_targetnames,
		    model_results = model_results,
			model_names = model_names,
		    fullfile = fullfile,
		    dfplot = data_set_original )




if __name__ == '__main__':
	app.run(debug=True)