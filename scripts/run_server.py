"""
Launcher script to run the QAFD-RAG interactive web application server.
Usage:
    python scripts/run_server.py [--port 8000] [--host 127.0.0.1]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import uvicorn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_server")


def main():
    parser = argparse.ArgumentParser(description="Start QAFD-RAG Interactive Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("STARTING QAFD-RAG INTERACTIVE WEB APPLICATION")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 65 + "\n")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
