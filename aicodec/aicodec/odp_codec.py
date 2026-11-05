"""One-D-Piece token codec for aiortc (video/tokenIdODP).

Variable-length sibling of the liquid-era ``video/tokenIdLLM`` codec:

* framing is IDENTICAL to :mod:`aicodec.tokenid` (VP8-like payload
  descriptor + 12-byte token header ``(picture_id, timestamp, pos, size)``),
  so the jitter buffer / depayload / NACK handling all carry over;
* the PAYLOAD differs: One-D-Piece codebook ids are 12-bit (4096 entries,
  matching the paper's 1.5 bytes/token), so ``ODPTokenEncoder`` packs them
  12-bit big-endian (2 tokens per 3 bytes) instead of the uint16 layout
  used by liquid's ``TokenIdEncoder`` — 256 tokens = 384 B, not 512 B;
* the receive side reassembles by the per-chunk ``pos``/``size`` header
  fields (``recover_token_from_data`` in tokenid.py assumes a FIXED
  2048-byte liquid frame and cannot be reused) and unpacks the 12-bit
  stream back to token ids, recovering the EXACT frame length.

Phase-1 behaviour: reassembled frames are pushed as
``(picture_id, timestamp, tokens)`` tuples onto a module-level asyncio
queue (``get_odp_queue()``) for the cloud server to consume (dump/metrics/
optional reconstruction); ``decode()`` returns ``[]`` (no video frames),
mirroring ``AsyncTokenIdDecoder``. ``decode()`` runs on aiortc's receiver
thread and MUST stay cheap — any heavy work (detokenize, LLM) belongs in
the queue consumer, not here.

Phase-2: the consumer of ``get_odp_queue()`` swaps the dump for a
connector+LLM call — this codec and the edge side never change.
"""

import asyncio
import logging
import random
from struct import pack, unpack_from
from typing import Optional

from aiortc.codecs.base import Decoder, Encoder
from aiortc.mediastreams import VIDEO_TIME_BASE, convert_timebase

logger = logging.getLogger(__name__)

TOKEN_HEADER_LENGTH = 12  # picture_id(4) + timestamp(4) + pos(2) + size(2)

#: bits per token id (codebook 4096 = 2**12 -> 1.5 bytes/token, per paper)
ODP_TOKEN_BITS = 12


def packed_size(num_tokens: int) -> int:
    """Byte length of ``num_tokens`` 12-bit-packed token ids."""
    return (num_tokens * ODP_TOKEN_BITS + 7) // 8


def pack_tokens_12bit(tokens) -> bytes:
    """Pack token ids (each < 4096) into a 12-bit big-endian bitstream.

    Layout: two tokens per 3 bytes —
      byte0 = t0[11:4], byte1 = t0[3:0]<<4 | t1[11:8], byte2 = t1[7:0].
    An odd trailing token occupies 2 bytes with its low nibble zero-padded,
    so the count is exactly recoverable from the byte length.
    """
    out = bytearray()
    it = iter(tokens)
    for t0 in it:
        t1 = next(it, None)
        if t1 is None:
            out += bytes(((t0 >> 4) & 0xFF, (t0 & 0xF) << 4))
        else:
            out += bytes(((t0 >> 4) & 0xFF,
                          ((t0 & 0xF) << 4) | ((t1 >> 8) & 0xF),
                          t1 & 0xFF))
    return bytes(out)


def unpack_tokens_12bit(data: bytes) -> tuple:
    """Inverse of :func:`pack_tokens_12bit`; count derived from byte length.

    len%3==0 -> 2 tokens per 3 bytes; len%3==2 -> one extra 2-byte token.
    A trailing partial byte (loss) is dropped.
    """
    n_pairs = len(data) // 3
    tokens = []
    for i in range(n_pairs):
        b0, b1, b2 = data[3 * i], data[3 * i + 1], data[3 * i + 2]
        tokens.append((b0 << 4) | (b1 >> 4))
        tokens.append(((b1 & 0xF) << 8) | b2)
    rem = len(data) - 3 * n_pairs
    if rem == 2:
        tokens.append((data[-2] << 4) | (data[-1] >> 4))
    return tuple(tokens)


_odp_queue: Optional[asyncio.Queue] = None


def get_odp_queue() -> asyncio.Queue:
    """Module-level queue of received token frames (created on first use)."""
    global _odp_queue
    if _odp_queue is None:
        _odp_queue = asyncio.Queue(maxsize=30)
    return _odp_queue


def odp_recover_tokens(frame_data: bytes):
    """Reassemble a variable-length 12-bit token frame from depayloaded data.

    ``frame_data`` is the concatenation of per-packet chunks, each
    ``token_header (12B) + payload``. Chunks may be missing (RTP loss);
    gaps leave zeros, exactly like the fixed-size liquid path, but the
    recovered length is the true frame size (max pos+size), not a constant.

    Returns ``(picture_id, timestamp, tokens_tuple)``.
    """
    picture_id = -1
    timestamp = 0
    chunks = []
    max_end = 0
    pointer = 0
    data_len = len(frame_data)
    while pointer + TOKEN_HEADER_LENGTH <= data_len:
        pid, ts, pos, size = unpack_from("!IIHH", frame_data, pointer)
        pointer += TOKEN_HEADER_LENGTH
        actual = min(size, data_len - pointer)
        chunks.append((pos, frame_data[pointer:pointer + actual]))
        pointer += actual
        picture_id = pid
        timestamp = ts
        max_end = max(max_end, pos + actual)

    filled = bytearray(max_end)
    for pos, chunk in chunks:
        end = min(pos + len(chunk), max_end)
        filled[pos:end] = chunk[:end - pos]

    tokens = unpack_tokens_12bit(bytes(filled)) if max_end else ()
    return picture_id, timestamp, tokens


# ==============================
# Send-side encoder (12-bit packed tokens)
# ==============================
class ODPTokenEncoder(Encoder):
    """``video/tokenIdODP`` encoder: tokenize + 12-bit pack + packetize.

    Mirrors ``TokenIdEncoder`` (shared bridge, VP8-like descriptor, 12-byte
    token header, ``PACKET_MAX`` chunking) with the uint16 payload replaced
    by the 12-bit bitstream. Frames larger than one RTP payload are split
    into chunks carrying ``pos``/``size`` for loss-tolerant reassembly.
    """

    bridge = None

    def __init__(self) -> None:
        self.picture_id = random.randint(0, (1 << 15) - 1)
        self.__target_bitrate = 500000

    @classmethod
    def preload_bridge(cls):
        if cls.bridge is not None:
            print(f"directly return {cls.bridge}")
            return True
        import aicodec.shared as shared
        from aicodec.models import get_bridge
        cls.bridge = get_bridge(shared.get_model(), "enc")
        print(f"odp tokenizer preloaded, {cls.bridge}")
        return True

    @classmethod
    def cleanup_bridge(cls):
        cls.bridge = None

    @classmethod
    def require_bridge(cls):
        if cls.bridge is None:
            raise RuntimeError("bridge not initialized")
        return cls.bridge

    def encode(self, frame, force_keyframe: bool = False):
        # step 1. frame -> PIL image -> tokenizer -> 12-bit ids
        img = frame.to_image()
        bridge = ODPTokenEncoder.require_bridge()
        tokens = bridge.tokenize(img)
        # step 2. 12-bit pack (1.5 B/token)
        data = pack_tokens_12bit(tokens)
        # step 3. packetize following the tokenid framing
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        payloads = self._packetize_tokenheader(data, self.picture_id, timestamp)
        self.picture_id = (self.picture_id + 1) % (1 << 15)
        return payloads, timestamp

    def pack(self, packet):
        # direct packetize (backup path, mirrors TokenIdEncoder.pack)
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        payloads = self._packetize_tokenheader(bytes(packet), self.picture_id, timestamp)
        self.picture_id = (self.picture_id + 1) % (1 << 15)
        return payloads, timestamp

    @property
    def target_bitrate(self) -> int:
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        self.__target_bitrate = max(250000, min(bitrate, 1500000))

    @classmethod
    def _packetize_tokenheader(cls, buffer: bytes, picture_id: int,
                               timestamp: int) -> list:
        from aicodec.tokenid import VpxPayloadDescriptor, PACKET_MAX

        payloads = []
        descr = VpxPayloadDescriptor(
            partition_start=1, partition_id=0, picture_id=picture_id)
        pos = 0
        length = len(buffer)
        while pos < length:
            descr_bytes = bytes(descr)
            size = min(length - pos,
                       PACKET_MAX - len(descr_bytes) - TOKEN_HEADER_LENGTH)
            token_header = pack("!IIHH", picture_id, timestamp, pos, size)
            payloads.append(descr_bytes + token_header + buffer[pos:pos + size])
            descr.partition_start = 0
            pos += size
        if not payloads:  # empty frame: still emit one (header-only) packet
            descr_bytes = bytes(descr)
            token_header = pack("!IIHH", picture_id, timestamp, 0, 0)
            payloads.append(descr_bytes + token_header)
        return payloads


class ODPTokenDecoder(Decoder):
    """Receive-side decoder for ``video/tokenIdODP``.

    Cheap by design: reassemble + enqueue only. The queue consumer (cloud
    server) owns any heavy processing (reconstruction, Phase-2 LLM).
    """

    def __init__(self) -> None:
        self.queue = get_odp_queue()

    def decode(self, encoded_frame):
        picture_id, timestamp, tokens = odp_recover_tokens(encoded_frame.data)
        try:
            self.queue.put_nowait((picture_id, timestamp, tokens))
        except asyncio.QueueFull:
            logger.warning("ODP queue full, dropping frame ts=%s", timestamp)
        return []
