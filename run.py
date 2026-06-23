import os
import subprocess
import threading
import asyncio
import websockets
import json
import base64
import queue
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reem")

# Global queue for audio chunks
audio_queue = queue.Queue()

async def handle_websocket(websocket):
    """Handle WebSocket connections from the browser"""
    log.info("🟢 Client connected")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    # Receive audio from browser
                    audio_data = base64.b64decode(data.get("data", ""))
                    audio_queue.put(audio_data)
                    log.info(f"📥 Received audio chunk: {len(audio_data)} bytes")
                    
                elif msg_type == "start":
                    log.info("🎤 Audio stream started")
                    
                elif msg_type == "stop":
                    log.info("⏹️ Audio stream stopped")
                    audio_queue.put(None)
                    
            except Exception as e:
                log.error(f"❌ Error processing message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        log.info("🔴 Client disconnected")
    except Exception as e:
        log.error(f"❌ WebSocket error: {e}")

async def start_websocket_server():
    """Start the WebSocket server"""
    port = int(os.environ.get("WS_PORT", 8765))
    async with websockets.serve(handle_websocket, "0.0.0.0", port):
        log.info(f"✅ WebSocket server started on port {port}")
        await asyncio.Future()  # Run forever

def run_websocket():
    """Run WebSocket server in background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())

# ── Main ──────────────────────────────────────────────────────────────────────

log.info("🚀 Starting Reem Voice Agent...")

# Start WebSocket in background thread
ws_thread = threading.Thread(target=run_websocket, daemon=True)
ws_thread.start()
log.info("✅ WebSocket thread started")

# Give WebSocket time to start
time.sleep(2)

# Start Streamlit
port = int(os.environ.get("PORT", 7860))
log.info(f"🎯 Starting Streamlit on port {port}")

subprocess.run([
    "streamlit", "run", "app.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])
