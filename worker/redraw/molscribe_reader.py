"""Reads a chemical structure image with MolScribe (MIT licence) and returns the molecule with atom positions.

MolScribe's published package pins libraries that do not exist for current Python versions, so its inference is
run here directly on its model code:
- image preparation is done with OpenCV, exactly as MolScribe's own inference transform (crop white border, resize
  to the model's input size, greyscale, ImageNet normalisation);
- its Swin backbone is built from MolScribe's own class, without timm's model registry;
- the decoder's few OpenNMT-py modules are bundled in worker/third_party/onmt;
- the weights are loaded strictly: a missing or unexpected weight is an error, never silently ignored.

The model (swin_base_char_aux_1m680k.pth, 1.08 GB) is downloaded once by worker/setup_ai.py, checked against its
published SHA-256, and reduced to the reading weights (molscribe_weights.pt, 366 MB; the rest is training state).
That file is loaded weights-only, so it cannot run code. Nothing is downloaded at run time.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

SOURCE_URL = "https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth"
SOURCE_FILE = "swin_base_char_aux_1m680k.pth"
SOURCE_SHA256 = "32821fa4ba8d17f9a92e374601438c5e757947dbf1141e9ad0ef51cccfa1c501"
SOURCE_SIZE = 1_134_173_494
WEIGHTS_FILE = "molscribe_weights.pt"
BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]

_lock = threading.Lock()
_reader = None
_error: str | None = None


def models_dir() -> Path:
    base = os.environ.get("IQE_MODELS") or os.path.join(os.environ.get("LOCALAPPDATA", str(Path.home())),
                                                        "ImageQualityEnhancer", "models")
    return Path(base)


def model_path() -> Path:
    return models_dir() / "molscribe" / WEIGHTS_FILE


def verify_source(path: Path) -> bool:
    """Full SHA-256 check of the downloaded checkpoint (a few seconds); used by the setup script."""
    if not path.exists() or path.stat().st_size != SOURCE_SIZE:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest() == SOURCE_SHA256


def slim(source: Path, target: Path) -> None:
    """Keeps only what reading needs (encoder and decoder weights and the model settings)."""
    import torch
    states = torch.load(str(source), map_location="cpu", weights_only=False)  # verified by SHA-256 first
    torch.save({"encoder": states["encoder"], "decoder": states["decoder"], "args": dict(states["args"])}, str(target))


def available() -> bool:
    return _get() is not None


def unavailable_reason() -> str | None:
    _get()
    return _error


def _get():
    global _reader, _error
    if _reader is not None or _error is not None:
        return _reader
    with _lock:
        if _reader is None and _error is None:
            try:
                path = model_path()
                if not path.exists():
                    raise FileNotFoundError(f"MolScribe model not installed at {path} (run worker/setup_ai.py)")
                _reader = _Reader(path)
            except Exception as exc:  # missing packages or model: redraw is simply unavailable
                _error = f"{type(exc).__name__}: {exc}"
    return _reader


def _import_molscribe():
    third_party = str(Path(__file__).resolve().parents[1] / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    # MolScribe's Swin module was written for timm 0.4: give it the two helpers it imports and make its model
    # registration a no-op (the backbone is built directly below, not through timm's registry).
    import timm.models.helpers as helpers
    import timm.models.registry as registry
    import timm.models.vision_transformer as vit
    if not hasattr(helpers, "overlay_external_default_cfg"):
        helpers.overlay_external_default_cfg = lambda *args, **kwargs: None
    if not hasattr(vit, "_init_vit_weights"):
        vit._init_vit_weights = lambda module, name="", head_bias=0.0, jax_impl=False: None
    # MolScribe's package __init__ imports its plotting interface (matplotlib), which recognition does not need:
    # register the package without running it, so only the model modules are loaded.
    if "molscribe" not in sys.modules:
        import importlib.util
        import types
        spec = importlib.util.find_spec("molscribe")
        if spec is None or not spec.submodule_search_locations:
            raise ModuleNotFoundError("molscribe is not installed (run worker/setup_ai.py)")
        package = types.ModuleType("molscribe")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["molscribe"] = package
    original_register = registry.register_model
    registry.register_model = lambda fn: fn
    try:
        from molscribe.transformer import swin_transformer
    finally:
        registry.register_model = original_register
    import molscribe.model as model
    from molscribe.chemistry import convert_graph_to_smiles
    from molscribe.tokenizer import get_tokenizer
    return swin_transformer, model, get_tokenizer, convert_graph_to_smiles


class _Reader:
    def __init__(self, path: Path):
        import torch
        swin, model, get_tokenizer, convert = _import_molscribe()
        self._torch = torch
        self._convert = convert
        states = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        args = argparse.Namespace(encoder="swin_base", decoder="transformer", trunc_encoder=False, no_pretrained=True,
                                  use_checkpoint=False, dropout=0.5, embed_dim=256, enc_pos_emb=False,
                                  dec_num_layers=6, dec_hidden_size=256, dec_attn_heads=8, dec_num_queries=128,
                                  hidden_dropout=0.1, attn_dropout=0.1, max_relative_positions=0,
                                  continuous_coords=False, compute_confidence=False, input_size=384,
                                  vocab_file=None, coord_bins=64, sep_xy=True)
        for key, value in states["args"].items():
            setattr(args, key, value)
        args.use_checkpoint = False  # gradient checkpointing is for training only
        self.input_size = int(args.input_size)

        def create_model(name, pretrained=False, **kwargs):
            if not name.startswith("swin_base"):
                raise ValueError(f"unsupported MolScribe encoder: {name}")
            return swin.SwinTransformer(img_size=self.input_size, patch_size=4, window_size=12, embed_dim=128,
                                        depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), use_checkpoint=False)

        model.timm.create_model = create_model
        self.tokenizer = get_tokenizer(args)
        self.encoder = model.Encoder(args, pretrained=False)
        args.encoder_dim = self.encoder.n_features
        self.decoder = model.Decoder(args, self.tokenizer)
        for module, key in ((self.encoder, "encoder"), (self.decoder, "decoder")):
            weights = {k.replace("module.", ""): v for k, v in states[key].items()}
            result = module.load_state_dict(weights, strict=False)
            # Buffers rebuilt at construction (relative position indices, attention masks) may be absent from
            # the checkpoint; any other missing or unexpected weight means the model would not be MolScribe.
            missing = [k for k in result.missing_keys if not k.endswith(("relative_position_index", "attn_mask"))]
            if missing or result.unexpected_keys:
                raise RuntimeError(f"MolScribe {key} weights do not match: missing {missing[:5]}, "
                                   f"unexpected {result.unexpected_keys[:5]}")
            module.eval()
        del states
        self.decoder.compute_confidence = True
        torch.set_grad_enabled(False)

    def prepare(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """MolScribe's inference transform: crop to the drawing, resize, greyscale, normalise.
        Returns the tensor input and the crop (x0, y0, x1, y1) so predicted positions map back to the image."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = np.ascontiguousarray(img[:, :, :3])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # CropWhite: MolScribe crops to pixels that are not pure white. Scans and JPEGs have no pure white, so
        # anything lighter than 245 counts as background here.
        content = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 245)
        ys, xs = np.nonzero(content)
        if len(xs):
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        else:
            x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]
        crop = rgb[y0:y1, x0:x1]
        resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        normalised = (gray - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return np.ascontiguousarray(normalised.transpose(2, 0, 1)), (x0, y0, x1, y1)

    def read(self, img: np.ndarray) -> dict:
        """{"smiles", "molfile", "confidence", "atoms": [{"symbol", "x", "y", "confidence"}], "bonds": [...]};
        atom x/y are pixel positions in the given image."""
        torch = self._torch
        tensor, (x0, y0, x1, y1) = self.prepare(img)
        features, hiddens = self.encoder(torch.from_numpy(tensor)[None])
        prediction = self.decoder.decode(features, hiddens)[0]
        tokens = prediction["chartok_coords"]
        # The conversion rescales the coordinates in place (to the molecule's own bounding box): keep the
        # positions as predicted on the image.
        coords = [tuple(c) for c in tokens["coords"]]
        smiles_list, molblock_list, _ = self._convert([[list(c) for c in coords]], [list(tokens["symbols"])],
                                                      [prediction["edges"]], images=[img], num_workers=1)
        width, height = x1 - x0, y1 - y0
        atoms = [{"symbol": symbol, "x": x0 + float(cx) * width, "y": y0 + float(cy) * height,
                  "confidence": float(score)}
                 for symbol, (cx, cy), score in zip(tokens["symbols"], coords, tokens["atom_scores"])]
        bonds = []
        edges, scores = prediction["edges"], prediction["edge_scores"]
        for i in range(len(atoms) - 1):
            for j in range(i + 1, len(atoms)):
                if edges[i][j]:
                    bonds.append({"type": BOND_TYPES[edges[i][j]], "atoms": (i, j), "confidence": float(scores[i][j])})
        return {"smiles": smiles_list[0], "molfile": molblock_list[0], "confidence": float(prediction["overall_score"]),
                "atoms": atoms, "bonds": bonds}


def read(img: np.ndarray) -> dict:
    reader = _get()
    if reader is None:
        raise RuntimeError(_error or "MolScribe is not available")
    with _lock:  # one prediction at a time per worker process
        return reader.read(img)
