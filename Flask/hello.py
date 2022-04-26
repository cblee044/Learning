from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "<H1>This is Flask</H1>"