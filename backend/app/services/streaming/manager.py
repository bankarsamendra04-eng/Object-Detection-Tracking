import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional
import uuid
from fastapi import WebSocket, WebSocketDisconnect

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.streaming.session import StreamSession
from backend.app.api.schemas.websocket import (
    ClientCommand,
    ConnectionAckData,
    ServerMessageType,
    StreamErrorData,
    WebSocketMessage,
)

logger = get_logger("streaming.manager")


class WebSocketManager:
    """
    Manages active WebSocket client connections and streaming sessions.
    Ensures safe concurrency, multi-client isolation, and complete resource cleanup on disconnect.
    """

    def __init__(self):
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = asyncio.Lock()

    @property
    def active_connections_count(self) -> int:
        return len(self._sessions)

    async def connect(self, websocket: WebSocket) -> StreamSession:
        """
        Accepts incoming WebSocket connection, registers a new isolated StreamSession,
        and sends the connection acknowledgment message.
        Enforces maximum concurrent active sessions limit.
        """
        async with self._lock:
            if len(self._sessions) >= settings.MAX_CONCURRENT_WS_SESSIONS:
                logger.warning(
                    f"WebSocket connection rejected: concurrent session limit reached ({settings.MAX_CONCURRENT_WS_SESSIONS})"
                )
                await websocket.close(code=1008, reason="Concurrent streaming session limit exceeded.")
                raise WebSocketDisconnect(code=1008)

        await websocket.accept()
        session_id = str(uuid.uuid4())

        session = StreamSession(session_id=session_id, websocket=websocket)

        async with self._lock:
            self._sessions[session_id] = session

        logger.info(f"WebSocket client connected. Active sessions: {len(self._sessions)} (session={session_id})")

        # Send connection_ack
        ack_message = WebSocketMessage(
            type=ServerMessageType.CONNECTION_ACK,
            session_id=session_id,
            data=ConnectionAckData(
                session_id=session_id,
                message="Connected to Real-Time Computer Vision Streaming Engine",
                supported_sources=["webcam", "video"],
                server_time=datetime.now(timezone.utc).isoformat(),
            ),
        )
        await session.enqueue_message(ack_message)
        # Start sender task for immediate message delivery
        if session.sender_task is None or session.sender_task.done():
            session.sender_task = asyncio.create_task(session._sender_loop())

        return session

    async def disconnect(self, session_id: str) -> None:
        """Cleans up resources for a disconnected session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)

        if session:
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error during session close for {session_id}: {e}")
            logger.info(f"WebSocket client disconnected and resources released (session={session_id}). Active sessions: {len(self._sessions)}")

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        return self._sessions.get(session_id)

    async def close_all(self) -> None:
        """Gracefully closes all active sessions and releases hardware handles and threads."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error closing session during shutdown: {e}")


# Global singleton instance
_manager_instance: Optional[WebSocketManager] = None


def get_stream_manager() -> WebSocketManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = WebSocketManager()
    return _manager_instance
