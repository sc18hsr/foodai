from flask import Flask,render_template,url_for,redirect,send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user,LoginManager,login_required, logout_user
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_uploads import UploadSet,IMAGES, configure_uploads
from flask_bcrypt import Bcrypt
from wtforms import StringField,PasswordField,SubmitField, FileField
from wtforms.validators import InputRequired,Length, ValidationError

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

from flask_bootstrap import Bootstrap5

#images folder
IMAGE_FOLDER = 'static'

 # db intitialized here
app = Flask(__name__)
bootstrap = Bootstrap5(app)

#hash function
bcrypt = Bcrypt(app)

#database is stored in instance folder
db = SQLAlchemy()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

#setting up secret key
app.config['SECRET_KEY'] = "secret"

#images folder
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER 

#download folder
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
db.init_app(app)

#setting up our image upload
photos = UploadSet('photos',IMAGES)
configure_uploads(app,photos)

#loading our classification model
loaded_model = load_model('VGG_veg_0001.h5')

#initializing login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(20),nullable = False,unique=True)
    password = db.Column(db.String(20),nullable = False)

class RegistrationForm(FlaskForm):
    username = StringField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Username"})
    password = StringField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Password"})
    submit = SubmitField("Register")

    def validate_username(self,username):
        existing_user_username = User.query.filter_by(username=username.data).first()

        if existing_user_username:
            raise ValidationError("Username exists")

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos,'Only images are allowed'),
            FileRequired('This should not be empty')
        ]
    )
    submit = SubmitField('Upload')
    

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Username"})
    password = StringField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Password"})
    submit = SubmitField("Login")

@app.route('/')
def home():
    img_filename = os.path.join(app.config['IMAGE_FOLDER'], 'im-121422.jpeg')
    return render_template('home.html', banner_image = img_filename)

@app.route('/dashboard', methods = ['GET','POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods = ['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html',form = form)

@app.route('/login', methods = ['GET','POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods = ['GET','POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password= hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form = form)

def predict(image_url):
    tomato_img = image_url
    label_map = {'Bean': 0,
                'Brinjal': 1,
                'Broccoli': 2,
                'Bottle_Gourd': 3,
                'Bitter_Gourd': 4,
                'Pumpkin': 5,
                'Radish': 6,
                'Cucumber': 7,
                'Carrot': 8,
                'Potato': 9,
                'Cauliflower': 10,
                'Cabbage': 11,
                'Capsicum': 12,
                'Papaya': 13,
                'Tomato': 14}
    img2 = load_img(tomato_img, target_size=(224, 224))
    img_array = img_to_array(img2)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    probs = loaded_model.predict(img_array)

    predicted_classes = np.argmax(probs, axis=1)
    class_names = {v: k for k, v in label_map.items()}
    predicted_class_names = [class_names[i] for i in predicted_classes]

    return predicted_class_names
    
@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/upload', methods=['GET','POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        labels = predict(os.path.join("uploads",filename))
    else:
        file_url = None
        labels = None
    return render_template('index.html',form=form, file_url = file_url,labels = labels)

if __name__ == '__main__':
    app.run(debug=True)

