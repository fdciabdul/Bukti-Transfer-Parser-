from flask import Flask, request, jsonify
import os
from ocr.ocr_processor import process_ocr

app = Flask(__name__)

@app.route("/parse", methods=["POST"])
def ocr():
    file = request.files['file']
    filepath = os.path.join('output', file.filename)
    file.save(filepath)

    result = process_ocr(filepath)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
