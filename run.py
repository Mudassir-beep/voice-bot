import threading
import time
import os
import sys

# Start WebSocket server in background
def start_websocket():
    import asyncio
    import websockets
    import json
    import base64
    import queue
    import logging
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("reem")
    
    audio_queue = queue.Queue()
    
    async def deepgram_stream():
        # ... (implementation from app.py)
        pass
    
    async def handle_websocket(websocket):
        # ... (implementation from app.py)
        pass
    
    async def start_websocket_server():
        port = int(os.environ.get("WS_PORT", 8765))
        async with websockets.serve(handle_websocket, "0.0.0.0", port):
            log.info(f"WebSocket server started on port {port}")
            await asyncio.Future()
    
    asyncio.run(start_websocket_server())

# Start WebSocket in thread
ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

# Give WebSocket time to start
time.sleep(2)

# Import and run Streamlit
import subprocess
port = int(os.environ.get("PORT", 10000))
subprocess.run([
    "streamlit", "run", "app.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])