"""Train the object-crop encoder with the LeJEPA objective, then probe it.

Loss = prediction + lambda * SIGReg. Each anchor crop gives one *context* view
and several *target* views of the same object: another augmentation of the
same crop, and augmentations of the same track 3-15 recorded steps away in the
same episode -- the object seen from a slightly different pose, lighting or
occlusion, which is the invariance a policy's object embedding should have.
The predictor maps the context embedding to each target embedding (MSE), with
gradients through both sides: LeJEPA has no teacher, no EMA and no
stop-gradient, and SIGReg alone is what rules out the collapsed solution.

After training, the frozen encoder embeds every training crop to fit the
exported PCA-32 head, and a linear probe asks whether the embedding knows
*which mesh* it is looking at (the model id, which for 113_coffee-box is one to
one with `entity_identity.size_key`), fitted on training crops and scored on
validation crops -- beside the same probe on the *untrained* encoder, because
a random ViT's features are already a strong baseline on four visually
distinct meshes and a trained score means little without it.

Offline numbers only: no claim here survives to closed-loop success unless a
policy is evaluated with the embedding (see CLAUDE.md).
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import EncoderConfig, LeJEPA, PCAProjection, images_to_float
from .sigreg import DEFAULT_SLICES, effective_rank, random_directions, sigreg

MIN_GAP = 3
MAX_GAP = 15


# --------------------------------------------------------------------------- data


class CropStore:
    """One split written by scripts/build_object_crops.py, held in memory."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.directory = directory
        self.meta = json.loads((directory / "meta.json").read_text())
        self.crops = np.load(directory / "crops.npy")
        index = np.load(directory / "index.npz")
        self.index = {key: index[key] for key in index.files}
        count = len(self.crops)
        if count != self.meta["count"] or any(len(v) != count for v in self.index.values()):
            raise ValueError(f"{directory}: crops, index and meta disagree on the row count")
        if self.crops.dtype != np.uint8 or self.crops.shape[1:] != (64, 64, 3):
            raise ValueError(f"{directory}: crops are {self.crops.shape} {self.crops.dtype}")

    def __len__(self) -> int:
        return len(self.crops)

    def temporal_neighbours(
        self, min_gap: int = MIN_GAP, max_gap: int = MAX_GAP
    ) -> tuple[np.ndarray, np.ndarray]:
        """CSR (offsets, rows): for row i, rows[offsets[i]:offsets[i+1]] are the
        same (episode, track) with min_gap <= |frame difference| <= max_gap."""
        episode = self.index["episode"].astype(np.int64)
        track = self.index["track"].astype(np.int64)
        frame = self.index["frame"].astype(np.int64)
        order = np.lexsort((frame, track, episode))
        group = episode[order] * 1000 + track[order]
        boundaries = np.flatnonzero(np.diff(group)) + 1
        lists: list[np.ndarray] = [np.empty(0, np.int64)] * len(frame)
        for rows in np.split(order, boundaries):
            frames = frame[rows]
            for position, row in enumerate(rows):
                gap = np.abs(frames - frames[position])
                lists[row] = rows[(gap >= min_gap) & (gap <= max_gap)]
        lengths = np.array([len(item) for item in lists])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        flat = np.concatenate(lists) if lengths.sum() else np.empty(0, np.int64)
        return offsets, flat


def sample_targets(
    anchors: np.ndarray,
    offsets: np.ndarray,
    flat: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(count, B) temporal-neighbour rows per anchor; the anchor itself when a
    track has no neighbour (a crop near an episode end, or a skipped frame)."""
    starts, ends = offsets[anchors], offsets[anchors + 1]
    lengths = ends - starts
    if len(flat) == 0:
        return np.tile(anchors, (count, 1))
    draws = rng.integers(0, np.maximum(lengths, 1), size=(count, len(anchors)))
    picked = np.where(lengths > 0, flat[np.minimum(starts + draws, len(flat) - 1)], anchors)
    return picked


# --------------------------------------------------------------------------- augmentation


def augment(
    images: torch.Tensor,
    generator: torch.Generator,
    *,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    jitter: float = 0.4,
    jitter_p: float = 0.8,
    blur_p: float = 0.5,
    blur_sigma: tuple[float, float] = (0.1, 1.5),
) -> torch.Tensor:
    """Batched per-sample random resized crop, colour jitter and Gaussian blur.

    uint8 (B, H, W, 3) in, float (B, 3, H, W) in [0, 1] out. Written in plain
    torch so every sample draws its own parameters on the device in one
    kernel per op; torchvision's v2 functionals apply one draw to a batch.
    No flips: the boxes carry printed logos, and a mirrored logo is an
    appearance the encoder should not be taught to ignore.
    """
    x = images_to_float(images)
    batch, _, height, width = x.shape
    device = x.device

    def uniform(low, high, *shape):
        return torch.rand(*shape, generator=generator, device=device) * (high - low) + low

    area = uniform(scale[0], scale[1], batch)
    log_ratio = uniform(math.log(ratio[0]), math.log(ratio[1]), batch)
    aspect = torch.exp(log_ratio)
    w = torch.sqrt(area * aspect).clamp(max=1.0)
    h = torch.sqrt(area / aspect).clamp(max=1.0)
    cx = uniform(-1.0, 1.0, batch) * (1.0 - w)
    cy = uniform(-1.0, 1.0, batch) * (1.0 - h)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 2] = w, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, (batch, 3, height, width), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    apply = (torch.rand(batch, generator=generator, device=device) < jitter_p).view(-1, 1, 1, 1)
    brightness = uniform(1 - jitter, 1 + jitter, batch).view(-1, 1, 1, 1)
    contrast = uniform(1 - jitter, 1 + jitter, batch).view(-1, 1, 1, 1)
    saturation = uniform(1 - jitter, 1 + jitter, batch).view(-1, 1, 1, 1)
    jittered = (x * brightness).clamp(0, 1)
    gray = (0.299 * jittered[:, 0] + 0.587 * jittered[:, 1] + 0.114 * jittered[:, 2]).unsqueeze(1)
    jittered = (
        (jittered - gray.mean(dim=(2, 3), keepdim=True)) * contrast
        + gray.mean(dim=(2, 3), keepdim=True)
    ).clamp(0, 1)
    gray = (0.299 * jittered[:, 0] + 0.587 * jittered[:, 1] + 0.114 * jittered[:, 2]).unsqueeze(1)
    jittered = (gray + (jittered - gray) * saturation).clamp(0, 1)
    x = torch.where(apply, jittered, x)

    radius = 3
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=x.dtype)
    sigma = uniform(blur_sigma[0], blur_sigma[1], batch)
    kernel = torch.exp(-0.5 * (offsets[None] / sigma[:, None]).square())
    kernel = kernel / kernel.sum(dim=1, keepdim=True)  # (B, K)
    kernel = kernel.repeat_interleave(3, dim=0)  # (B*3, K)
    flat = x.reshape(1, batch * 3, height, width)
    flat = F.pad(flat, (radius, radius, radius, radius), mode="replicate")
    flat = F.conv2d(flat, kernel.view(-1, 1, 1, 2 * radius + 1), groups=batch * 3)
    flat = F.conv2d(flat, kernel.view(-1, 1, 2 * radius + 1, 1), groups=batch * 3)
    blurred = flat.view(batch, 3, height, width)
    apply = (torch.rand(batch, generator=generator, device=device) < blur_p).view(-1, 1, 1, 1)
    return torch.where(apply, blurred, x).clamp(0, 1)


# --------------------------------------------------------------------------- objective


def lejepa_loss(
    model: LeJEPA,
    context: torch.Tensor,
    targets: list[torch.Tensor],
    *,
    lambda_sigreg: float,
    step: int,
    slices: int = DEFAULT_SLICES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """(loss, parts) for float views: context (B, 3, H, W), targets V x same."""
    batch = context.shape[0]
    embeddings = model.encoder(torch.cat([context, *targets], dim=0))
    views = embeddings.float().view(len(targets) + 1, batch, -1)
    predicted = model.predictor(embeddings[:batch]).float()
    prediction = (predicted.unsqueeze(0) - views[1:]).square().mean()
    # One set of directions per step for every view, as in the reference code.
    directions = random_directions(views.shape[-1], slices, step=step, device=views.device)
    regulariser = torch.stack([sigreg(v, directions=directions) for v in views]).mean()
    loss = prediction + lambda_sigreg * regulariser
    return loss, {"prediction": prediction, "sigreg": regulariser, "context": views[0]}


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def embed(model: LeJEPA, crops: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(crops), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(crops[start : start + batch])).to(device)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            z = model.encoder(images_to_float(chunk))
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    weight_decay: float = 1e-4,
    iterations: int = 200,
) -> dict[str, float]:
    """Multinomial logistic regression (full-batch L-BFGS), train -> val.

    sklearn is not installed in the policy environment; this is the same
    model. Features are standardised on the training split only.
    """
    classes = np.unique(train_y)
    lookup = {int(c): i for i, c in enumerate(classes)}
    unseen = sorted(set(int(v) for v in np.unique(val_y)) - set(lookup))
    if unseen:
        raise ValueError(f"validation labels {unseen} never occur in training")
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    xt = torch.from_numpy(((train_x - mean) / std).astype(np.float32))
    xv = torch.from_numpy(((val_x - mean) / std).astype(np.float32))
    yt = torch.tensor([lookup[int(v)] for v in train_y])
    yv = torch.tensor([lookup[int(v)] for v in val_y])
    layer = torch.nn.Linear(xt.shape[1], len(classes))
    optimiser = torch.optim.LBFGS(
        layer.parameters(), max_iter=iterations, line_search_fn="strong_wolfe"
    )

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(layer(xt), yt) + weight_decay * layer.weight.square().sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        train_acc = (layer(xt).argmax(1) == yt).float().mean().item()
        val_acc = (layer(xv).argmax(1) == yv).float().mean().item()
    majority = float(np.mean(val_y == classes[np.bincount(yt.numpy()).argmax()]))
    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_majority_acc": majority,
        "classes": [int(c) for c in classes],
    }


def probe_report(
    model: LeJEPA, train_store: CropStore, val_store: CropStore | None, device, pca_k: int
) -> tuple[dict, PCAProjection]:
    train_z = embed(model, train_store.crops, device)
    report = {"train_effective_rank": effective_rank(torch.from_numpy(train_z))}
    pca = PCAProjection.fit(train_z, pca_k)
    report["pca_explained_fraction"] = float(
        pca.explained_variance.sum() / train_z.var(axis=0, ddof=1).sum()
    )
    if val_store is not None:
        val_z = embed(model, val_store.crops, device)
        ty, vy = train_store.index["model_id"], val_store.index["model_id"]
        report["model_id_probe_192"] = linear_probe(train_z, ty, val_z, vy)
        report["model_id_probe_pca"] = linear_probe(pca(train_z), ty, pca(val_z), vy)
    return report, pca


# --------------------------------------------------------------------------- training


def _learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train(
    crops_dir: str | Path,
    out_dir: str | Path,
    steps: int = 20000,
    batch: int = 256,
    lr: float = 5e-4,
    lambda_sigreg: float = 0.05,
    device: str = "cuda",
    seed: int = 0,
    *,
    target_views: int = 3,
    slices: int = DEFAULT_SLICES,
    weight_decay: float = 0.05,
    log_every: int = 100,
    config: EncoderConfig | None = None,
) -> dict:
    """Train on `<crops_dir>/train`, probe against `<crops_dir>/val`, write `out_dir`.

    `target_views` targets per context: one re-augmentation of the anchor crop
    and `target_views - 1` temporal neighbours.
    """
    config = config or EncoderConfig()
    if target_views < 1:
        raise ValueError("target_views must be at least 1")
    crops_dir, out_dir = Path(crops_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; refusing to overwrite a run")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_store = CropStore(crops_dir / "train")
    val_store = CropStore(crops_dir / "val") if (crops_dir / "val" / "meta.json").exists() else None
    offsets, flat = train_store.temporal_neighbours()
    without = int(np.sum(np.diff(offsets) == 0))
    crops = torch.from_numpy(train_store.crops).to(device)

    model = LeJEPA(config).to(device)
    initial = copy.deepcopy(model.state_dict())
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    warmup = max(1, min(1000, steps // 10))
    use_amp = torch.device(device).type == "cuda"

    run_config = {
        "crops_dir": str(crops_dir),
        "steps": steps,
        "batch": batch,
        "lr": lr,
        "lambda_sigreg": lambda_sigreg,
        "seed": seed,
        "target_views": target_views,
        "slices": slices,
        "weight_decay": weight_decay,
        "warmup": warmup,
        "temporal_gap": [MIN_GAP, MAX_GAP],
        "encoder": config.to_dict(),
        "train_crops": len(train_store),
        "val_crops": len(val_store) if val_store else 0,
        "train_rows_without_temporal_neighbour": without,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps({k: v for k, v in run_config.items() if k != "encoder"}))

    history = []
    start = time.time()
    log = (out_dir / "log.jsonl").open("w")
    for step in range(steps):
        model.train()
        for group in optimiser.param_groups:
            group["lr"] = _learning_rate(step, steps, lr, warmup)
        anchors = rng.integers(0, len(train_store), size=batch)
        neighbours = sample_targets(anchors, offsets, flat, target_views - 1, rng)
        anchor_crops = crops[torch.from_numpy(anchors).to(device)]
        context = augment(anchor_crops, generator)
        targets = [augment(anchor_crops, generator)]
        targets += [
            augment(crops[torch.from_numpy(rows).to(device)], generator) for rows in neighbours
        ]
        with torch.autocast(
            device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_amp
        ):
            loss, parts = lejepa_loss(
                model, context, targets, lambda_sigreg=lambda_sigreg, step=step, slices=slices
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % log_every == 0 or step == steps - 1:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "prediction": float(parts["prediction"].detach()),
                "sigreg": float(parts["sigreg"].detach()),
                "effective_rank": effective_rank(parts["context"].detach()),
                "grad_norm": float(grad_norm),
                "lr": optimiser.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - start, 1),
            }
            history.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    log.close()

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "config": config.to_dict(),
        },
        out_dir / "lejepa.pt",
    )
    report, pca = probe_report(model, train_store, val_store, device, config.visual_dim)
    pca.save(out_dir / f"pca{config.visual_dim}.npz")
    baseline = LeJEPA(config).to(device)
    baseline.load_state_dict(initial)
    baseline_report, _ = probe_report(baseline, train_store, val_store, device, config.visual_dim)
    probe = {
        "label": "model_id (1:1 with entity_identity.size_key for 113_coffee-box)",
        "trained": report,
        "random_init": baseline_report,
        "final_log": history[-1],
    }
    (out_dir / "probe.json").write_text(json.dumps(probe, indent=2))
    print(json.dumps(probe, indent=2))
    return probe


def load_encoder(path: str | Path, device="cpu") -> LeJEPA:
    """The trained LeJEPA module from a run's `lejepa.pt`."""
    state = torch.load(Path(path), map_location=device)
    model = LeJEPA(EncoderConfig(**state["config"]))
    model.encoder.load_state_dict(state["encoder"])
    model.predictor.load_state_dict(state["predictor"])
    return model.to(device).eval()
