import json
from pydantic import ValidationError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.streaming.manager import get_stream_manager
from backend.app.api.schemas.websocket import (
    ClientCommand,
    ServerMessageType,
    StreamErrorData,
    WebSocketMessage,
)

logger = get_logger("api.websocket")
router = APIRouter()


@router.websocket("/ws/stream")
@router.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-Time WebSocket Streaming Endpoint for live Computer Vision processing.
    Connects clients to an isolated StreamSession capable of streaming from:
    1. Hardware webcam (index-based)
    2. Video files (.mp4, .avi, etc.)

    Message Protocol:
    - Connection returns 'connection_ack' with assigned session_id.
    - Client issues JSON commands:
      - {"action": "start", "source": "webcam", "camera_index": 0, "fps_limit": 15}
      - {"action": "start", "source": "video", "path": "data/samples/sample_real_bus.mp4"}
      - {"action": "pause"} / {"action": "resume"}
      - {"action": "stop"}
      - {"action": "ping"}
    """
    manager = get_stream_manager()
    session = await manager.connect(websocket)

    try:
        while True:
            raw_text = await websocket.receive_text()

            # 0. Enforce maximum message size
            if len(raw_text) > settings.MAX_WS_MESSAGE_SIZE_BYTES:
                err_msg = WebSocketMessage(
                    type=ServerMessageType.ERROR,
                    session_id=session.session_id,
                    data=StreamErrorData(
                        code="MESSAGE_TOO_LARGE",
                        message=f"WebSocket payload size ({len(raw_text)} bytes) exceeds limit of {settings.MAX_WS_MESSAGE_SIZE_BYTES} bytes.",
                    ),
                )
                await session.enqueue_message(err_msg)
                continue

            # 1. Parse JSON format
            try:
                raw_json = json.loads(raw_text)
            except json.JSONDecodeError as err:
                err_msg = WebSocketMessage(
                    type=ServerMessageType.ERROR,
                    session_id=session.session_id,
                    data=StreamErrorData(
                        code="INVALID_JSON",
                        message="Payload must be valid JSON.",
                        details={"raw": raw_text[:100]},
                    ),
                )
                await session.enqueue_message(err_msg)
                continue

            # 2. Validate via Pydantic ClientCommand
            try:
                cmd = ClientCommand.model_validate(raw_json)
            except ValidationError as val_err:
                clean_errors = [
                    {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
                    for err in val_err.errors()
                ]
                err_msg = WebSocketMessage(
                    type=ServerMessageType.ERROR,
                    session_id=session.session_id,
                    data=StreamErrorData(
                        code="INVALID_COMMAND",
                        message="Command validation failed.",
                        details={"errors": clean_errors},
                    ),
                )
                await session.enqueue_message(err_msg)
                continue

            # 3. Dispatch validated command to session
            await session.handle_command(cmd)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected normally (session={session.session_id})")
    except Exception as exc:
        logger.error(f"WebSocket error in session={session.session_id}: {exc}", exc_info=True)
    finally:
        await manager.disconnect(session.session_id)
