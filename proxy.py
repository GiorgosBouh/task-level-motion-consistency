from flask import Flask, request, Response
import requests
from flask_cors import CORS

app = Flask(__name__)
# Επιτρέπουμε τα πάντα (CORS) για να μη σε μπλοκάρει ο Browser
CORS(app, resources={r"/*": {"origins": "*"}})

OLLAMA_URL = "http://127.0.0.1:11434"

@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(path):
    if request.method == 'OPTIONS':
        return Response(), 200
    
    resp = requests.request(
        method=request.method,
        url=f"{OLLAMA_URL}/{path}",
        headers={key: value for (key, value) in request.headers if key != 'Host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False)

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    response = Response(resp.content, resp.status_code, headers)
    return response

if __name__ == '__main__':
    # Τρέχουμε στην πόρτα 5000
    app.run(host='0.0.0.0', port=5000)
