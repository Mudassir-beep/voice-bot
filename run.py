import os
import subprocess

# Get port from Railway
port = int(os.environ.get("PORT", 7860))

# Run Streamlit - app.py already has the WebSocket server
subprocess.run([
    "streamlit", "run", "app.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])
