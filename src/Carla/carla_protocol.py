"""Small, versioned socket protocol for the CARLA--DREAM process boundary.

The installed CARLA 0.9.14 client runs under Python 3.7 while the DREAM
planning stack runs in a newer Python environment.  This module intentionally
uses only the Python standard library and Python 3.7 syntax so both processes
can import the same framing and validation code.

Each wire frame is a four-byte unsigned network-order payload length followed
by one UTF-8 JSON document.  Messages use the envelope::

    {
        "schema": "dream.carla.protocol",
        "version": 1,
        "type": "observation",
        "payload": {...}
    }

The helpers validate the envelope, not the scenario-specific payload fields.
Payload schemas can therefore evolve independently while the transport keeps
strict framing, type, and version checks.
"""

import json
import socket
import struct

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - compatibility with older Python.
    from collections import Mapping


SCHEMA_ID = "dream.carla.protocol"
PROTOCOL_VERSION = 1
SUPPORTED_MESSAGE_TYPES = frozenset(
    ("hello", "observation", "plan", "heartbeat", "error", "shutdown")
)

# Large enough for numerical trajectories and diagnostics, while preventing a
# corrupt or hostile prefix from causing an unbounded allocation/read.
DEFAULT_MAX_MESSAGE_BYTES = 4 * 1024 * 1024

_HEADER = struct.Struct("!I")


class ProtocolError(Exception):
    """Base class for CARLA protocol failures."""


class ConnectionClosed(ProtocolError):
    """The peer closed the connection before a complete frame was received."""


class ProtocolTimeout(ProtocolError):
    """A socket operation timed out before a complete frame was transferred."""


class FrameTooLarge(ProtocolError):
    """A frame exceeds the configured message-size limit."""


class MalformedMessage(ProtocolError):
    """A frame is not valid UTF-8 JSON."""


class MessageValidationError(ProtocolError):
    """A JSON message does not satisfy the protocol envelope contract."""


def _validate_max_message_bytes(max_message_bytes):
    if isinstance(max_message_bytes, bool) or not isinstance(max_message_bytes, int):
        raise ValueError("max_message_bytes must be an integer")
    if max_message_bytes <= 0 or max_message_bytes > 0xFFFFFFFF:
        raise ValueError("max_message_bytes must be in [1, 2**32 - 1]")


def make_message(message_type, payload, schema=SCHEMA_ID, version=PROTOCOL_VERSION):
    """Build and validate one protocol envelope."""

    message = {
        "schema": schema,
        "version": version,
        "type": message_type,
        "payload": payload,
    }
    validate_message(message)
    return message


def validate_schema_version(
    message,
    expected_schema=SCHEMA_ID,
    expected_version=PROTOCOL_VERSION,
):
    """Validate the envelope's schema identifier and integer version.

    Returns ``(schema, version)`` on success so callers can log the negotiated
    values without reading the envelope twice.
    """

    if not isinstance(message, Mapping):
        raise MessageValidationError("message must be a JSON object")

    schema = message.get("schema")
    if not isinstance(schema, str) or not schema:
        raise MessageValidationError("message.schema must be a non-empty string")
    if schema != expected_schema:
        raise MessageValidationError(
            "unsupported schema {!r}; expected {!r}".format(schema, expected_schema)
        )

    version = message.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise MessageValidationError("message.version must be an integer")
    if version != expected_version:
        raise MessageValidationError(
            "unsupported protocol version {!r}; expected {!r}".format(
                version, expected_version
            )
        )
    return schema, version


def validate_message_type(
    message,
    expected_type=None,
    supported_types=SUPPORTED_MESSAGE_TYPES,
):
    """Validate and return the envelope's message type."""

    if not isinstance(message, Mapping):
        raise MessageValidationError("message must be a JSON object")

    message_type = message.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise MessageValidationError("message.type must be a non-empty string")

    if supported_types is not None and message_type not in supported_types:
        raise MessageValidationError(
            "unsupported message type {!r}".format(message_type)
        )
    if expected_type is not None and message_type != expected_type:
        raise MessageValidationError(
            "unexpected message type {!r}; expected {!r}".format(
                message_type, expected_type
            )
        )
    return message_type


def validate_message(
    message,
    expected_type=None,
    expected_schema=SCHEMA_ID,
    expected_version=PROTOCOL_VERSION,
    supported_types=SUPPORTED_MESSAGE_TYPES,
):
    """Validate the complete transport envelope and return ``message``.

    The payload must be a JSON object.  More specific observation and plan
    payload validation belongs in their respective adapters.
    """

    validate_schema_version(
        message,
        expected_schema=expected_schema,
        expected_version=expected_version,
    )
    validate_message_type(
        message,
        expected_type=expected_type,
        supported_types=supported_types,
    )

    if "payload" not in message:
        raise MessageValidationError("message.payload is required")
    if not isinstance(message["payload"], Mapping):
        raise MessageValidationError("message.payload must be a JSON object")
    return message


def encode_message(message, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES, validate=True):
    """Serialize a message into one length-prefixed wire frame."""

    _validate_max_message_bytes(max_message_bytes)
    if validate:
        validate_message(message)
    try:
        payload = json.dumps(
            message,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise MalformedMessage("message is not JSON serializable: {}".format(error))

    if not payload:
        raise MalformedMessage("encoded JSON payload cannot be empty")
    if len(payload) > max_message_bytes:
        raise FrameTooLarge(
            "encoded frame is {} bytes; limit is {} bytes".format(
                len(payload), max_message_bytes
            )
        )
    return _HEADER.pack(len(payload)) + payload


def send_message(sock, message, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES, validate=True):
    """Encode and send one complete message, returning its wire byte count."""

    frame = encode_message(
        message,
        max_message_bytes=max_message_bytes,
        validate=validate,
    )
    try:
        sock.sendall(frame)
    except socket.timeout as error:
        raise ProtocolTimeout("timed out while sending a protocol frame") from error
    except (BrokenPipeError, ConnectionResetError) as error:
        raise ConnectionClosed("peer closed the connection while sending") from error
    return len(frame)


def _recv_exact(sock, size, label):
    chunks = []
    received = 0
    while received < size:
        try:
            chunk = sock.recv(size - received)
        except socket.timeout as error:
            raise ProtocolTimeout(
                "timed out while receiving {} ({}/{} bytes)".format(
                    label, received, size
                )
            ) from error
        except ConnectionResetError as error:
            raise ConnectionClosed(
                "peer reset the connection while receiving {} ({}/{} bytes)".format(
                    label, received, size
                )
            ) from error

        if not chunk:
            raise ConnectionClosed(
                "peer closed the connection while receiving {} ({}/{} bytes)".format(
                    label, received, size
                )
            )
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_message(
    sock,
    max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES,
    validate=True,
    expected_type=None,
):
    """Receive, decode, and optionally validate one protocol frame."""

    _validate_max_message_bytes(max_message_bytes)
    header = _recv_exact(sock, _HEADER.size, "frame header")
    payload_size = _HEADER.unpack(header)[0]
    if payload_size == 0:
        raise MalformedMessage("zero-length JSON frames are not allowed")
    if payload_size > max_message_bytes:
        raise FrameTooLarge(
            "incoming frame declares {} bytes; limit is {} bytes".format(
                payload_size, max_message_bytes
            )
        )

    raw_payload = _recv_exact(sock, payload_size, "frame payload")
    try:
        text = raw_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MalformedMessage("frame payload is not valid UTF-8") from error
    try:
        message = json.loads(text)
    except (TypeError, ValueError) as error:
        raise MalformedMessage("frame payload is not valid JSON") from error

    if validate:
        validate_message(message, expected_type=expected_type)
    return message


__all__ = [
    "SCHEMA_ID",
    "PROTOCOL_VERSION",
    "SUPPORTED_MESSAGE_TYPES",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "ProtocolError",
    "ConnectionClosed",
    "ProtocolTimeout",
    "FrameTooLarge",
    "MalformedMessage",
    "MessageValidationError",
    "make_message",
    "validate_schema_version",
    "validate_message_type",
    "validate_message",
    "encode_message",
    "send_message",
    "recv_message",
]
