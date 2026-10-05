"""Low-level WebSocket client for TradingView real-time and historical protocols."""
import json
import ssl
import websocket
from typing import Dict, Any, List, Optional, Tuple

HEADERS = {
    "Origin": "https://www.tradingview.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def format_message(func: str, args: list) -> str:
    """Encodes an RPC function call into a TradingView WebSocket frame."""
    msg = json.dumps({"m": func, "p": args}, separators=(',', ':'))
    return f"~m~{len(msg)}~m~{msg}"

def parse_message(raw_data: str) -> List[Any]:
    """Decodes TradingView packet frames into JSON payloads or heartbeat strings."""
    messages = []
    while raw_data:
        try:
            if not raw_data.startswith("~m~"):
                break
            parts = raw_data.split("~m~")
            if len(parts) < 3:
                break
            length = int(parts[1])
            start = len("~m~") + len(parts[1]) + len("~m~")
            payload = raw_data[start:start + length]
            if payload.startswith("~h~"):
                messages.append(payload)
            else:
                messages.append(json.loads(payload))
            raw_data = raw_data[start + length:]
        except Exception:
            break
    return messages

class TradingViewSocketClient:
    """Manages persistent WebSocket connection, framing, heartbeat responses, and authentication."""

    def __init__(
        self,
        sessionid: str,
        sessionid_sign: str = "",
        jwt_token: str = "unauthorized_user_token",
        server: str = "data",
        timeout: float = 15.0
    ):
        self.sessionid = sessionid
        self.sessionid_sign = sessionid_sign
        self.jwt_token = jwt_token
        self.server = server
        self.timeout = timeout
        self.ws: Optional[websocket.WebSocket] = None
        self._build_url_and_headers()

    def _build_url_and_headers(self):
        self.ws_url = f"wss://{self.server}.tradingview.com/socket.io/websocket?from=chart&type=chart"
        cookie_str = f"sessionid={self.sessionid};"
        if self.sessionid_sign:
            cookie_str += f" sessionid_sign={self.sessionid_sign};"
        self.headers = dict(HEADERS)
        self.headers["Cookie"] = cookie_str

    def update_credentials(
        self,
        sessionid: str,
        sessionid_sign: str = "",
        jwt_token: str = "unauthorized_user_token",
        server: str = "data"
    ):
        """Rotates credentials and switches server endpoints cleanly."""
        self.close()
        self.sessionid = sessionid
        self.sessionid_sign = sessionid_sign
        self.jwt_token = jwt_token
        self.server = server
        self._build_url_and_headers()

    def connect(self) -> websocket.WebSocket:
        """Establishes WebSocket connection and sends auth token."""
        if self.ws is None or not getattr(self.ws, "connected", False):
            self.close()
            self.ws = websocket.create_connection(
                self.ws_url,
                header=self.headers,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                timeout=self.timeout
            )
            self.ws.recv()  # Initial handshake frame
            self.send("set_auth_token", [self.jwt_token])
        return self.ws

    def send(self, func: str, args: list):
        """Formats and sends an RPC command over the active WebSocket."""
        ws = self.connect()
        ws.send(format_message(func, args))

    def recv_messages(self, timeout: float = 2.0) -> List[Any]:
        """Receives a frame, replies to heartbeats automatically, and returns decoded JSON messages."""
        if not self.ws or not getattr(self.ws, "connected", False):
            self.connect()

        self.ws.settimeout(timeout)
        raw = self.ws.recv()
        messages = parse_message(raw)
        result = []
        for m in messages:
            if isinstance(m, str) and m.startswith("~h~"):
                # Immediately respond to heartbeat ping to maintain connection
                self.ws.send(f"~m~{len(m)}~m~{m}")
            elif isinstance(m, dict):
                result.append(m)
        return result

    def keep_alive(self):
        """Handles any incoming ping frames while idle."""
        if self.ws and getattr(self.ws, "connected", False):
            try:
                self.ws.settimeout(0.02)
                raw = self.ws.recv()
                for m in parse_message(raw):
                    if isinstance(m, str) and m.startswith("~h~"):
                        self.ws.send(f"~m~{len(m)}~m~{m}")
            except Exception:
                pass

    def close(self):
        """Closes the WebSocket connection gracefully."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
