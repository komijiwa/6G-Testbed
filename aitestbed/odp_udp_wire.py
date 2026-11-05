"""Wire framing for the One-D-Piece image link over UDP.

Shared by the UDP clients (fragmented / whole-image) and the UDP server so
the three files can never drift apart on the byte layout.

Why UDP at all
--------------
The HTTP/TCP sibling (``odp_image_http_client.py`` /
``odp_image_http_server.py``) cannot express packet loss: TCP retransmits
every lost segment, so a netem ``loss`` profile only ever shows up as added
LATENCY and the token stream handed to the decoder is always bit-perfect.
Over UDP a dropped datagram is a genuinely missing slice of the 12-bit token
stream, which is exactly the degradation the offline token-drop study
(``run_mmbench.sh --keep-ratio/--drop-mode``) models analytically.

Datagram layout
---------------
    APP_HEADER (18 B) | NAME (name_len B) | TOKEN_HEADER (12 B) | CHUNK

APP_HEADER = ``!4sBBHIIH``
    magic        4s  b"ODPU"
    version      B   = 1
    nchunks      B   total datagrams for this image (1..255)
    name_len     H   length of the UTF-8 filename that follows
    index        I   dataset index of the image
    frame_bytes  I   total 12-bit-packed bytes of the WHOLE image (e.g. 384)
    chunk_seq    H   0-based datagram sequence within the image

TOKEN_HEADER is byte-for-byte the RTP link's header (``!IIHH`` =
picture_id, timestamp, pos, size), so the server can concatenate whatever
chunks survived and hand them straight to
:func:`aicodec.odp_codec.odp_recover_tokens` with no translation layer.

Two deliberate design choices
-----------------------------
* **The filename rides in EVERY datagram.** UDP has no control plane and no
  ordering guarantee, so a "first datagram carries the metadata" scheme
  loses the entire image whenever that one datagram drops. Repeating ~30 B
  of name per datagram is cheap and makes every datagram self-describing.
* **Chunks split on EVEN token counts only.** The 12-bit packing stores two
  tokens per three bytes; an odd split would straddle a byte boundary and
  corrupt the token pair that spans the seam.

Control datagram (session end)
------------------------------
    ``!4sBB`` = magic b"ODPC", version, kind   (kind 1 = SESSION_COMPLETE)

Sent a few times by the client because it, too, can be lost; the server
treats it idempotently.
"""

import json
import struct
from pathlib import Path
from typing import Optional

#: image file extensions recognized by discover_images()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: data datagram magic
ODP_UDP_MAGIC = b"ODPU"
#: control datagram magic
ODP_CTL_MAGIC = b"ODPC"
ODP_UDP_VERSION = 1

#: control kinds
CTL_SESSION_COMPLETE = 1

APP_HEADER_FMT = "!4sBBHIIH"  ##格式字符串，最后的H是Chunk Sequence Number分片序号
APP_HEADER_LENGTH = struct.calcsize(APP_HEADER_FMT)       # 18
CTL_HEADER_FMT = "!4sBB"
CTL_HEADER_LENGTH = struct.calcsize(CTL_HEADER_FMT)       # 6

#: mirrors aicodec.odp_codec.TOKEN_HEADER_LENGTH (picture_id, ts, pos, size)
TOKEN_HEADER_FMT = "!IIHH"
TOKEN_HEADER_LENGTH = struct.calcsize(TOKEN_HEADER_FMT)   # 12

#: 12-bit ids -> 2 tokens per 3 bytes
ODP_TOKEN_BITS = 12


def chunk_bytes_for_tokens(tokens_per_datagram: int) -> int:
    """Byte length of ``tokens_per_datagram`` 12-bit-packed ids.

    Rejects odd counts: two tokens share three bytes, so an odd split would
    cut a token pair in half and corrupt it on recovery.
    """
    if tokens_per_datagram <= 0:
        raise ValueError("tokens_per_datagram must be positive")
    if tokens_per_datagram % 2:
        raise ValueError(
            f"tokens_per_datagram must be EVEN (got {tokens_per_datagram}): "
            "12-bit packing stores 2 tokens per 3 bytes, so an odd split "
            "straddles a byte boundary and corrupts the straddling pair")
    return tokens_per_datagram * ODP_TOKEN_BITS // 8


def plan_chunks(frame_len: int, chunk_size: int):
    """``[(pos, size)]`` slices covering ``frame_len`` bytes."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out = []
    pos = 0
    while pos < frame_len:
        out.append((pos, min(chunk_size, frame_len - pos)))
        pos += chunk_size
    return out


def build_datagram(*, index: int, name: str, frame_bytes: int, nchunks: int,
                   chunk_seq: int, picture_id: int, timestamp: int,
                   pos: int, chunk: bytes) -> bytes:
    """Assemble one data datagram (see module docstring for the layout)."""
    if not 1 <= nchunks <= 255:
        raise ValueError(f"nchunks must fit one byte, got {nchunks}")
    raw_name = name.encode("utf-8")
    if len(raw_name) > 0xFFFF:
        raise ValueError("filename too long for the 16-bit name_len field")
    app = struct.pack(APP_HEADER_FMT, ODP_UDP_MAGIC, ODP_UDP_VERSION,
                      nchunks, len(raw_name), index, frame_bytes, chunk_seq)
    tok = struct.pack(TOKEN_HEADER_FMT, picture_id & 0xFFFFFFFF,
                      timestamp & 0xFFFFFFFF, pos, len(chunk))
    return app + raw_name + tok + chunk


def parse_datagram(data: bytes) -> dict:
    """Decode one data datagram. 数据

    Raises ``ValueError`` on anything malformed so the caller can count it
    as a corrupt datagram instead of crashing the receive loop.
    """
    if len(data) < APP_HEADER_LENGTH:
        raise ValueError(f"short datagram: {len(data)} B")
    magic, version, nchunks, name_len, index, frame_bytes, chunk_seq = \
        struct.unpack(APP_HEADER_FMT, data[:APP_HEADER_LENGTH])
    if magic != ODP_UDP_MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != ODP_UDP_VERSION:
        raise ValueError(f"unsupported version {version}")

    off = APP_HEADER_LENGTH
    end_name = off + name_len
    if len(data) < end_name + TOKEN_HEADER_LENGTH:
        raise ValueError("datagram truncated inside name/token header")
    name = data[off:end_name].decode("utf-8", errors="replace")

    tok = data[end_name:end_name + TOKEN_HEADER_LENGTH]
    picture_id, timestamp, pos, size = struct.unpack(TOKEN_HEADER_FMT, tok)
    chunk = data[end_name + TOKEN_HEADER_LENGTH:]
    if len(chunk) != size:
        raise ValueError(
            f"chunk length {len(chunk)} != declared size {size}")

    return {"nchunks": nchunks, "index": index, "frame_bytes": frame_bytes,
            "chunk_seq": chunk_seq, "name": name, "picture_id": picture_id,
            "timestamp": timestamp, "pos": pos, "size": size,
            "chunk": chunk, "token_header": tok}


def build_control(kind: int = CTL_SESSION_COMPLETE) -> bytes:
    """Assemble a control datagram."""
    return struct.pack(CTL_HEADER_FMT, ODP_CTL_MAGIC, ODP_UDP_VERSION, kind)


def parse_control(data: bytes):
    """Return the control ``kind``, or ``None`` if this is not a control
    datagram (so the receive loop can fall through to :func:`parse_datagram`).
    """
    if len(data) != CTL_HEADER_LENGTH:
        return None
    magic, version, kind = struct.unpack(CTL_HEADER_FMT, data)
    if magic != ODP_CTL_MAGIC or version != ODP_UDP_VERSION:
        return None
    return kind


def max_datagram_bytes(name: str, chunk_size: int) -> int:
    """Worst-case on-wire size, for MTU sanity checks."""
    return (APP_HEADER_LENGTH + len(name.encode("utf-8"))
            + TOKEN_HEADER_LENGTH + chunk_size)


def discover_images(image_dir: str, manifest_path: Optional[str] = None,
                     max_images: Optional[int] = None) -> list:
    """Ordered list of (index, name, source_abs_path).

    Shared dataset-discovery helper for both UDP clients (whole/chunked) and
    the orchestrator scenario layer, so all three agree on image order.

    Order of preference:
      1. explicit ``manifest_path`` jsonl;
      2. ``<image_dir>/manifest.jsonl`` (the dataset manifest, e.g.
         MMBench_DEV_EN ships index/image/question/answer per line) --
         this is the DATASET order, not filename-lexicographic order;
      3. filename-sorted directory scan (fallback).
    ``max_images``: cap; 0/None = no cap (full dataset).
    """
    base = Path(image_dir)
    manifest = Path(manifest_path) if manifest_path else None
    if manifest is None:
        auto = base / "manifest.jsonl"
        if auto.exists():
            manifest = auto
    if manifest is not None and manifest.exists():
        out = []
        with open(manifest, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rel = rec.get("image") or rec.get("path") or rec.get("name")
                if not rel:
                    continue
                p = Path(rel) if Path(rel).is_absolute() else base / rel
                if not p.exists():
                    continue
                try:
                    idx = int(rec.get("index", i))
                except (TypeError, ValueError):
                    idx = i
                out.append((idx, p.name, str(p)))
                if max_images and len(out) >= max_images:
                    break
        if out:
            return out
    files = sorted(
        (p for p in base.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name)
    out = [(i, p.name, str(p)) for i, p in enumerate(files)]
    if max_images:
        out = out[:max_images]
    return out
