from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def hello2():
    return render_template('hello.html')

@app.route('/hello', methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting' : 'Hello, ' + name + '!'
    }
    return jsonify(response)