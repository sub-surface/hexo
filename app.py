"""
app.py — HexGo dashboard launcher.

Starts the FastAPI server (server.py) and opens the dashboard in the
default browser. The server runs until Ctrl+C.

Usage: python app.py [--port 7860] [--no-browser]
"""

import argparse
import threading
import webbrowser

import uvicorn

from server import app


def main():
    parser = argparse.ArgumentParser(description="HexGo Dashboard")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"HexGo Dashboard -> {url}")

    if not args.no_browser:
        # Open browser after a short delay so the server is ready
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
