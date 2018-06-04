from flask import Flask, render_template

UPLOAD_FOLDER = 'sapi/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    
    return render_template('home.html')
        
if __name__ == '__main__':
    app.run(debug=True)