from flask import Flask
from src.logger import logging

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    logging.info("we are testing second method of logging")
    return "welcome buddy"

if __name__ == "__main__":
    app.run(debug=True)