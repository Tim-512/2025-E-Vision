import threading
import time

import pytest

from ev_vision.camera.latest_frame import LatestFrameBuffer
from ev_vision.models import Frame


def test_latest_frame_overwrites_unconsumed_frame() -> None:
    buffer = LatestFrameBuffer()
    buffer.publish(Frame(sequence=1, captured_ns=10, image="old"))
    buffer.publish(Frame(sequence=2, captured_ns=20, image="new"))
    assert buffer.wait_next(after_sequence=0, timeout_s=0.01).sequence == 2


def test_wait_next_blocks_until_newer_sequence() -> None:
    buffer = LatestFrameBuffer()

    def publish() -> None:
        time.sleep(0.02)
        buffer.publish(Frame(sequence=3, captured_ns=30, image="frame"))

    thread = threading.Thread(target=publish)
    thread.start()
    frame = buffer.wait_next(after_sequence=2, timeout_s=0.2)
    thread.join()
    assert frame.sequence == 3


def test_wait_next_times_out() -> None:
    buffer = LatestFrameBuffer()
    with pytest.raises(TimeoutError, match="latest frame"):
        buffer.wait_next(after_sequence=0, timeout_s=0.01)


def test_sequence_must_increase() -> None:
    buffer = LatestFrameBuffer()
    buffer.publish(Frame(sequence=2, captured_ns=20, image="frame"))
    with pytest.raises(ValueError, match="sequence"):
        buffer.publish(Frame(sequence=2, captured_ns=30, image="duplicate"))
