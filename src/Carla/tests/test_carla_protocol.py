import socket
import struct
import sys
from pathlib import Path

import pytest

# Import the packaged shared transport module directly. Importing the full
# planning stack is intentionally unnecessary in the lightweight CARLA env.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_protocol import (  # noqa: E402
    ConnectionClosed,
    FrameTooLarge,
    MalformedMessage,
    MessageValidationError,
    PROTOCOL_VERSION,
    ProtocolTimeout,
    encode_message,
    make_message,
    recv_message,
    send_message,
    validate_message,
    validate_message_type,
    validate_schema_version,
)


class _ChunkedReceiver(object):
    """Socket facade that guarantees short reads for framing tests."""

    def __init__(self, sock, chunk_size):
        self._sock = sock
        self._chunk_size = chunk_size

    def recv(self, size):
        return self._sock.recv(min(size, self._chunk_size))


def test_socketpair_round_trip_preserves_utf8_payload():
    sender, receiver = socket.socketpair()
    try:
        message = make_message(
            "observation",
            {
                "frame_id": 42,
                "simulation_time_s": 1.5,
                "label": "occluded vehicle 车辆",
            },
        )
        wire_size = send_message(sender, message)
        received = recv_message(receiver, expected_type="observation")

        assert wire_size == len(encode_message(message))
        assert received == message
    finally:
        sender.close()
        receiver.close()


def test_partial_header_and_payload_reads_are_reassembled():
    sender, receiver = socket.socketpair()
    try:
        message = make_message(
            "plan",
            {"source_frame_id": 7, "controls": [{"a": -1.0, "delta": 0.02}]},
        )
        sender.sendall(encode_message(message))

        received = recv_message(
            _ChunkedReceiver(receiver, chunk_size=1),
            expected_type="plan",
        )
        assert received == message
    finally:
        sender.close()
        receiver.close()


def test_oversized_declared_frame_is_rejected_before_payload_read():
    sender, receiver = socket.socketpair()
    try:
        limit = 32
        sender.sendall(struct.pack("!I", limit + 1))
        with pytest.raises(FrameTooLarge, match="declares 33 bytes"):
            recv_message(receiver, max_message_bytes=limit)
    finally:
        sender.close()
        receiver.close()


def test_oversized_encoded_frame_is_rejected_on_send_side():
    message = make_message("heartbeat", {"padding": "x" * 128})
    with pytest.raises(FrameTooLarge, match="limit is 32 bytes"):
        encode_message(message, max_message_bytes=32)


@pytest.mark.parametrize("raw_payload", [b"{not-json", b"\xff\xfe\xfd"])
def test_malformed_json_or_utf8_is_rejected(raw_payload):
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack("!I", len(raw_payload)) + raw_payload)
        with pytest.raises(MalformedMessage):
            recv_message(receiver)
    finally:
        sender.close()
        receiver.close()


def test_schema_version_and_type_validation_helpers():
    message = make_message("observation", {"frame_id": 1})

    assert validate_schema_version(message)[1] == PROTOCOL_VERSION
    assert validate_message_type(message, expected_type="observation") == "observation"
    assert validate_message(message, expected_type="observation") is message

    wrong_version = dict(message, version=PROTOCOL_VERSION + 1)
    with pytest.raises(MessageValidationError, match="unsupported protocol version"):
        validate_schema_version(wrong_version)

    wrong_schema = dict(message, schema="unrelated.schema")
    with pytest.raises(MessageValidationError, match="unsupported schema"):
        validate_schema_version(wrong_schema)

    with pytest.raises(MessageValidationError, match="unexpected message type"):
        validate_message_type(message, expected_type="plan")

    unknown_type = dict(message, type="arbitrary")
    with pytest.raises(MessageValidationError, match="unsupported message type"):
        validate_message_type(unknown_type)


def test_payload_must_be_a_json_object():
    message = {
        "schema": "dream.carla.protocol",
        "version": PROTOCOL_VERSION,
        "type": "heartbeat",
        "payload": [],
    }
    with pytest.raises(MessageValidationError, match="payload must be a JSON object"):
        validate_message(message)


def test_clean_eof_reports_connection_closed():
    sender, receiver = socket.socketpair()
    sender.close()
    try:
        with pytest.raises(ConnectionClosed, match="frame header"):
            recv_message(receiver)
    finally:
        receiver.close()


def test_partial_frame_eof_reports_received_byte_count():
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(b"\x00\x00")
        sender.close()
        with pytest.raises(ConnectionClosed, match=r"2/4 bytes"):
            recv_message(receiver)
    finally:
        receiver.close()


def test_receive_timeout_is_wrapped_as_protocol_timeout():
    sender, receiver = socket.socketpair()
    try:
        receiver.settimeout(0.05)
        with pytest.raises(ProtocolTimeout, match="frame header"):
            recv_message(receiver)
    finally:
        sender.close()
        receiver.close()
