from flask import Flask, request, jsonify
from services.orchestrator import orchestrate_health

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    result = orchestrate_health(data)
    return jsonify({"status": "success", "data": result})

if __name__ == "__main__":
    app.run(debug=True)
