from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    prompt = request.form.get('prompt')
    return f'You entered: {prompt}'

if __name__ == '__main__':
    app.run(debug=True)