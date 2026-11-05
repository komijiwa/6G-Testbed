# One-D-Piece (ODP) <-> Qwen3-VL MMBench, lossy UDP link

End-to-end lossy-link image experiment: One-D-Piece L-256 tokenizes MMBench
images, sends them over UDP (chunked or whole-image mode, no retransmission),
the cloud server reconstructs the survivors, and Qwen3-VL scores the
reconstructed folder offline. Only this UDP path is kept; the earlier
HTTP and realtime-WebRTC ODP links were dropped as superseded.

## Repo layout (ODP-related parts only)

```
<repo root>/
  run_odp_mmbench_pipeline.py          # one-click pipeline: server -> orchestrator
                                        # (UDP link) -> Qwen3-VL eval -> summary
  aicodec/aicodec/
    odp_codec.py                       # token recover/pack helpers (shared by
                                        # the UDP server and the wire format)
    models/one_d_piece/
      __init__.py                      # registers OneDPieceBridge as "one_d_piece"
      bridge.py                        # aicodec bridge (enc/dec); loads the
                                        # architecture + weights via ODP_REPO /
                                        # ODP_CHECKPOINT / ODP_CONFIG (below)
      checkpoints/
        pytorch_model.bin              # ODP L-256 weights (NOT in git)
  One-D-Piece/                         # architecture code + config + data/
                                        # eval assets (NOT a Python package
                                        # under aicodec; kept at repo root)
    modeling/                          # vendored One-D-Piece model source
      one_d_piece.py, titok.py, modules/, quantizer/
    odp_l256.yaml                      # model config (vq/vit hyperparams)
    MMBench_DEV_EN/                    # dataset: manifest.jsonl (in git) +
                                        # images (NOT in git, see below)
    Qwen3-VL-30B-A3B-Instruct/         # Qwen3-VL weights (NOT in git)
    mmbench_qwen3vl/
      requirements.txt                 # pip-installable eval env deps
      fetch_odp_assets.sh              # data check + optional weight download
      mmbench_eval.py                  # MMBench MCQ eval driver (Qwen3-VL)
      run_odp_mmbench.sh               # CLI wrapper: ODP-reconstructed/baseline
                                        # folder -> Qwen3-VL accuracy table
  aitestbed/
    server/odp_image_udp_server.py     # cloud: UDP receive + ODP decode + recon
    clients/odp_image_udp_client_{whole,chunked}.py   # edge: ODP encode + UDP send
    odp_udp_wire.py                    # shared datagram framing
    scenarios/odp_image_udp.py         # orchestrator scenario (provider "odp-image-udp")
    configs/scenarios_odp_image_udp.yaml
    tools/check_odp_image_recon.py     # optional PSNR/SSIM acceptance check
    orchestrator.py                    # registers scenario "odp_image_udp" +
                                        # client provider "odp-image-udp"
```

## Why the model package only has `bridge.py` + `checkpoints/`

`bridge.py` already supports pointing at an external One-D-Piece checkout /
weights / config via env vars (no code changes needed):

| Env var           | Default (if unset)                                   | Used here as                              |
|--------------------|-------------------------------------------------------|--------------------------------------------|
| `ODP_REPO`         | `<model pkg>/vendor`                                   | `<repo root>/One-D-Piece`                   |
| `ODP_CONFIG`       | `<model pkg>/odp_l256.yaml`                            | `<repo root>/One-D-Piece/odp_l256.yaml`     |
| `ODP_CHECKPOINT`   | `<ODP_REPO>/checkpoints/pytorch_model.bin`             | `<repo root>/aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin` |

`ODP_REPO` must resolve to a folder containing `modeling/one_d_piece.py`
directly (no `vendor/` wrapper) -- that's exactly the `One-D-Piece/modeling/`
layout above, because the vendored code's own imports are all
`from modeling.xxx import ...` (absolute, assuming `modeling` is a top-level
package once `ODP_REPO` is pushed onto `sys.path`).

`run_odp_mmbench_pipeline.py` sets all three env vars for its server/orchestrator
subprocesses automatically (see `odp_bridge_env()`); if you run the UDP
server or `orchestrator.py` by hand, export them yourself first:

```bash
export ODP_REPO="$(pwd)/One-D-Piece"
export ODP_CONFIG="$(pwd)/One-D-Piece/odp_l256.yaml"
export ODP_CHECKPOINT="$(pwd)/aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin"
```

## Data and weights (NOT tracked in git -- migrate manually)

* `One-D-Piece/MMBench_DEV_EN/manifest.jsonl` is in git; the ~4300 jpgs are
  NOT (too large / no LFS). Copy them from an existing checkout or the team
  share into `One-D-Piece/MMBench_DEV_EN/`.
* `aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin` (ODP
  L-256 tokenizer, ~2.4 GB) and `One-D-Piece/Qwen3-VL-30B-A3B-Instruct/`
  (~60 GB) are NOT in git. Fetch with:
  ```bash
  bash One-D-Piece/mmbench_qwen3vl/fetch_odp_assets.sh --weights
  ```
  or place them manually at the paths above.

## Setup

```bash
python -m venv ~/envs/qwen3vl && source ~/envs/qwen3vl/bin/activate
pip install -r One-D-Piece/mmbench_qwen3vl/requirements.txt
# installs torch/omegaconf/torchvision for the ODP bridge itself, if not
# already present in your orchestrator env:
pip install torch torchvision omegaconf einops huggingface_hub

bash One-D-Piece/mmbench_qwen3vl/fetch_odp_assets.sh --weights
```

## Running

```bash
# one-click: server + orchestrator (UDP whole-image scenario) + Qwen3-VL eval
python run_odp_mmbench_pipeline.py \
    --profiles lossy --interface lo --max-images 0 \
    --eval-conda-env qwen3vl --eval-gpu 1

# or score an existing folder directly
bash One-D-Piece/mmbench_qwen3vl/run_odp_mmbench.sh --baseline-only --limit 20
bash One-D-Piece/mmbench_qwen3vl/run_odp_mmbench.sh --summary
```

## Removed (superseded by this UDP link, do not re-add)

* `clients/realtime_webrtc_odp_client.py`, `server/realtime_odp_token_server.py`,
  `scenarios/realtime_odp.py`, `aicodec/aicodec/register_one_d_piece.py`
  (realtime WebRTC/RTP token streaming -- a different registration path from
  `aicodec.models.one_d_piece`, only needed by this realtime link).
* `clients/odp_image_http_client.py`, `server/odp_image_http_server.py`,
  `scenarios/odp_image_http.py`, `configs/scenarios_odp*.yaml` (HTTP variant --
  TCP retransmits lost segments, so it cannot model packet loss; UDP replaces it).
