from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from process import process

app = Flask(__name__)

# Defines the upload folder
UPLOAD_FOLDER = 'uploads'
MEDIA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# checks if the upload folder exists, if not, creates it
def create_upload_folder():
  if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Route for the homepage
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload_file():
  create_upload_folder()
  if 'file' not in request.files:
    return redirect(request.url)
  file = request.files['file']
  if file.filename == '':
    return redirect(request.url)
  if file:
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # process uploaded file
    try:
        os.remove('./results/output.webm')
    except:
        pass
    
    crit=process(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('result.html',critique=crit)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)

