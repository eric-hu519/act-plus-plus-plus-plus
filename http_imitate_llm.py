
import zlib
import threading
from flask import Flask, request, Response,jsonify
import requests
import os
import pickle


app = Flask(__name__)

@app.route('/vision', methods=['GET', 'POST'])

def get_straw():
    return "task:transfer_straw"


if __name__ == '__main__':

    app.run(host='192.168.31.109', port=1116)
