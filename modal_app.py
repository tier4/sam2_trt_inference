"""Modal pipeline for SAM2 TensorRT engine preparation and bbox-guided inference on L40S."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal

REPO_ROOT = Path("/workspace/sam2_trt_inference")
ASSETS_ROOT = Path("/root/sam2_trt_assets")
ONNX_ROOT = ASSETS_ROOT / "onnx"
ENGINE_ROOT = ASSETS_ROOT / "engines"
DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "L40S")
DEFAULT_ENCODER_ENGINE = "sam2_encoder_fp16.engine"
DEFAULT_DECODER_ENGINE = "sam2_decoder_fp16.engine"
MODAL_TIMEOUT_SECONDS = 7200

_local_repo_root = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.01-py3")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("cmake", "libboost-all-dev", "libopencv-dev", "pkg-config")
    .pip_install("google-cloud-storage==3.6.0", "numpy==2.2.6", "opencv-python-headless==4.12.0.88")
    .add_local_dir(str(_local_repo_root), remote_path=str(REPO_ROOT), copy=True)
    .run_commands(
        f"cmake -S {REPO_ROOT} -B {REPO_ROOT / 'build'}",
        f"cmake --build {REPO_ROOT / 'build'} -j$(nproc)",
        f"cp {REPO_ROOT / 'build' / 'libtrt_sam2_infer.so'} /usr/local/lib/libtrt_sam2_infer.so",
        "ldconfig",
        f"mkdir -p {ONNX_ROOT} {ENGINE_ROOT}",
    )
    .env({"PYTHONPATH": f"{REPO_ROOT / 'python'}", "LD_LIBRARY_PATH": "/usr/local/lib:${LD_LIBRARY_PATH}"})
)

app = modal.App(name="sam2-tensorrt-inference", image=image)
assets_volume = modal.Volume.from_name("sam2-tensorrt-assets", create_if_missing=True)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a GCS URI into bucket and blob path."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected GCS URI, got {uri}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def _configure_gcp_credentials() -> str | None:
    """Configure GOOGLE_APPLICATION_CREDENTIALS when MODEL_TRAINING_SA_KEY is available."""
    sa_key = os.environ.get("MODEL_TRAINING_SA_KEY")
    if not sa_key:
        return None

    gcp_key_data = json.loads(sa_key)
    with tempfile.NamedTemporaryFile(delete=False) as temp_key_file:
        temp_key_file.write(json.dumps(gcp_key_data).encode())
        temp_key_path = temp_key_file.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path
    return temp_key_path


def _download_gcs_blob(source_uri: str, destination: Path) -> None:
    """Download a single GCS object to a local path."""
    from google.cloud import storage

    bucket_name, blob_name = _parse_gcs_uri(source_uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(destination))


def _materialize_input(source: str, destination: Path) -> None:
    """Download a GCS object or copy a local file into place."""
    if source.startswith("gs://"):
        _download_gcs_blob(source, destination)
        return
    destination.write_bytes(Path(source).read_bytes())


def _run_command(args: list[str]) -> None:
    """Run a subprocess and raise with captured output on failure."""
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or f"exit_code={completed.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{details}")


def _encode_png_base64(image_array: "np.ndarray") -> str:
    """Encode a numpy image array as base64 PNG."""
    import cv2

    success, encoded = cv2.imencode(".png", image_array)
    if not success:
        raise RuntimeError("Failed to encode PNG output.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _resolve_engine_pair(
    encoder_engine_name: str,
    decoder_engine_name: str,
) -> tuple[Path, Path]:
    """Resolve the engine pair under the shared assets volume."""
    return ENGINE_ROOT / encoder_engine_name, ENGINE_ROOT / decoder_engine_name


@app.function(
    gpu=DEFAULT_GPU,
    timeout=MODAL_TIMEOUT_SECONDS,
    volumes={str(ASSETS_ROOT): assets_volume},
    secrets=[modal.Secret.from_name("model-training", environment_name="main")],
    retries=0,
)
def prepare_engines(
    encoder_onnx_uri: str,
    decoder_onnx_uri: str,
    encoder_engine_name: str = DEFAULT_ENCODER_ENGINE,
    decoder_engine_name: str = DEFAULT_DECODER_ENGINE,
    precision: str = "fp16",
    min_batch: int = 1,
    opt_batch: int = 128,
    max_batch: int = 200,
    workspace_mb: int = 4096,
    force_rebuild: bool = False,
) -> dict[str, str]:
    """Download ONNX weights and build TensorRT engines on an L40S worker."""
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"precision must be fp16 or fp32, got {precision}")

    encoder_engine_path, decoder_engine_path = _resolve_engine_pair(encoder_engine_name, decoder_engine_name)
    if encoder_engine_path.exists() and decoder_engine_path.exists() and not force_rebuild:
        return {"encoder_engine_path": str(encoder_engine_path), "decoder_engine_path": str(decoder_engine_path)}

    temp_key_path = _configure_gcp_credentials()
    try:
        encoder_onnx_path = ONNX_ROOT / Path(encoder_onnx_uri).name
        decoder_onnx_path = ONNX_ROOT / Path(decoder_onnx_uri).name
        _materialize_input(encoder_onnx_uri, encoder_onnx_path)
        _materialize_input(decoder_onnx_uri, decoder_onnx_path)

        _run_command(
            [
                "bash",
                str(REPO_ROOT / "tools" / "generate_encoder_trt.sh"),
                str(encoder_onnx_path),
                str(encoder_engine_path),
                "--precision",
                precision,
                "--workspace",
                str(workspace_mb),
            ]
        )
        _run_command(
            [
                "bash",
                str(REPO_ROOT / "tools" / "generate_decoder_trt.sh"),
                str(decoder_onnx_path),
                str(decoder_engine_path),
                "--min-batch",
                str(min_batch),
                "--opt-batch",
                str(opt_batch),
                "--max-batch",
                str(max_batch),
                "--precision",
                precision,
                "--workspace",
                str(workspace_mb),
            ]
        )
        assets_volume.commit()
        return {"encoder_engine_path": str(encoder_engine_path), "decoder_engine_path": str(decoder_engine_path)}
    finally:
        if temp_key_path:
            Path(temp_key_path).unlink(missing_ok=True)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=MODAL_TIMEOUT_SECONDS,
    volumes={str(ASSETS_ROOT): assets_volume},
    secrets=[modal.Secret.from_name("model-training", environment_name="main")],
    retries=0,
)
def infer_from_bboxes(
    frame_uri: str,
    boxes: list[list[float]],
    encoder_engine_name: str = DEFAULT_ENCODER_ENGINE,
    decoder_engine_name: str = DEFAULT_DECODER_ENGINE,
    model_precision: str = "fp16",
    decoder_batch_limit: int = 50,
) -> dict[str, object]:
    """Run SAM2 TensorRT inference for a single image and normalized bbox list."""
    import cv2
    import numpy as np
    from trt_sam2 import SAM2Image

    encoder_engine_path, decoder_engine_path = _resolve_engine_pair(encoder_engine_name, decoder_engine_name)
    if not encoder_engine_path.exists() or not decoder_engine_path.exists():
        raise FileNotFoundError(
            "TensorRT engines are missing. Run prepare_engines() on an L40S worker before calling infer_from_bboxes()."
        )

    temp_key_path = _configure_gcp_credentials()
    local_image_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(frame_uri).suffix or ".jpg", delete=False) as temp_image:
            local_image_path = Path(temp_image.name)
        _materialize_input(frame_uri, local_image_path)

        image = cv2.imread(str(local_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image at {local_image_path}")
        image_height, image_width = image.shape[:2]

        normalized_bboxes: list[dict[str, object]] = []
        for index, bbox in enumerate(boxes):
            if len(bbox) != 4:
                raise ValueError(f"Each bbox must contain 4 normalized coordinates, got {bbox}")
            x1 = max(0.0, min(float(bbox[0]), 1.0)) * image_width
            y1 = max(0.0, min(float(bbox[1]), 1.0)) * image_height
            x2 = max(0.0, min(float(bbox[2]), 1.0)) * image_width
            y2 = max(0.0, min(float(bbox[3]), 1.0)) * image_height
            normalized_bboxes.append(
                {
                    "box": [x1, y1, x2, y2],
                    "prob": 1.0,
                    "id": index,
                    "label": 0,
                    "sub_name": f"box_{index}",
                }
            )

        sam2 = SAM2Image(
            encoder_path=str(encoder_engine_path),
            decoder_path=str(decoder_engine_path),
            model_precision=model_precision,
            decoder_batch_limit=decoder_batch_limit,
        )
        try:
            sam2.set_image(image)
            annotation_payload, entropy_map = sam2.sam2_from_bboxes(
                image=image,
                bboxes=normalized_bboxes,
                names=["box"],
                filename=Path(frame_uri).name,
                decoder_batch_size=decoder_batch_limit,
            )
        finally:
            sam2.destroy()

        return {
            "annotations": annotation_payload,
            "entropy_map_png_base64": _encode_png_base64(np.asarray(entropy_map)),
            "encoder_engine_path": str(encoder_engine_path),
            "decoder_engine_path": str(decoder_engine_path),
        }
    finally:
        if temp_key_path:
            Path(temp_key_path).unlink(missing_ok=True)
        if local_image_path:
            local_image_path.unlink(missing_ok=True)


@app.local_entrypoint()
def main(
    frame_uri: str,
    boxes_json: str = "[[0.1, 0.1, 0.4, 0.4]]",
    encoder_engine_name: str = DEFAULT_ENCODER_ENGINE,
    decoder_engine_name: str = DEFAULT_DECODER_ENGINE,
) -> None:
    """Run one inference call from the CLI."""
    boxes = json.loads(boxes_json)
    result = infer_from_bboxes.remote(
        frame_uri=frame_uri,
        boxes=boxes,
        encoder_engine_name=encoder_engine_name,
        decoder_engine_name=decoder_engine_name,
    )
    print(json.dumps(result, indent=2))
