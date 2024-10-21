# dashboard_server.py

import os
import sys
import logging
from flask import Flask, send_from_directory
from waitress import serve
from flask import request, abort
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Adjust the path to find the 'ui' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_folder = os.path.join(current_dir, "../ui/dist")

app = Flask(__name__, static_folder=ui_folder)
CORS(app)  # Enable CORS


# Remove X-Frame-Options header
@app.after_request
def add_header(response):
    response.headers.pop("X-Frame-Options", None)
    return response


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/shutdown", methods=["POST"])
def shutdown():
    if request.remote_addr not in ["127.0.0.1", "::1"]:
        abort(403)
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    logging.info("Dashboard server shutting down...")
    return "Server shutting down..."


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard Server")
    parser.add_argument("--port", type=int, default=3000, help="Port number")
    args = parser.parse_args()

    port = args.port

    # Start the server
    logging.info(f"Starting dashboard server on port {port}")
    serve(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
