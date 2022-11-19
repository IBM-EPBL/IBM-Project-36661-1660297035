import warnings, os, cv2, numpy as np

warnings.filterwarnings('ignore')
from os.path import join, dirname, realpath
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = join(dirname(realpath(__file__)), os.getcwd() + '/UPLOAD_FOLDER')
ALLOWED_EXTENSIONS = {'png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF'}
print("model loading starts")
body_model = load_model(os.getcwd() + '/Model/bodyl.h5')
level_model = load_model(os.getcwd() + '/Model/levell.h5')
print("model loaded ")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/UPLOAD_FOLDER'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # max upload - 10MB
app.secret_key = 'secret'
body_labels = ['Front', 'Rear', 'Side']
level_labels = ['Minor', 'Moderate', 'Severe']


def model_detection(model, frame, labels):
    img = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
    if np.max(img) > 1:
        img = img / 225.0
    img = np.array([img])
    prediction = model.predict(img)
    return labels[np.argmax(prediction)]


def detect(image_path):
    image = cv2.imread(image_path)
    x = model_detection(model=body_model, frame=image, labels=body_labels),
    print(x)
    print(type(x))
    print([x])
    print([x][0])
    y = model_detection(model=level_model, frame=image, labels=level_labels)
    result = {'gate1': 'Car validation check: ',
              'gate1_result': 1,
              'gate1_message': {0: None, 1: None},
              'gate2': 'Damage presence check: ',
              'gate2_result': 1,
              'gate2_message': {0: None, 1: None},
              'location': x,
              'severity': y,
              'final': 'Damage assessment complete!'}
    return result


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/<a>')
def available(a):
    flash('{} coming soon!'.format(a))
    return render_template('index.html', result=None, scroll='third')


@app.route('/assessment')
def assess():
    return render_template('index.html', result=None, scroll='third')


@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('assess'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('assess'))
        if file and allowed_file(file.filename):
            filename = secure_filename(
                file.filename)  # used to secure a filename before storing it directly on the filesystem
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model_results = detect(file_path)
            return render_template('results.html', result=model_results, scroll='third', filename=filename)

    flash('Invalid file format - please try your upload again.')
    return redirect(url_for('assess'))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
