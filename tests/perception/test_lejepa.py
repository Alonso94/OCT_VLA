import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from oct_vla.core.frames import WORKCELL_FRAME, Pose  # noqa: E402
from oct_vla.core.objects import ObjectState  # noqa: E402
from oct_vla.perception.lejepa.crops import (  # noqa: E402
    PinholeCamera,
    camera_from_placement,
    crop_object,
    default_head_camera,
    head_camera_override,
    intrinsic_from_fovy,
    square_crop_box,
)
from oct_vla.perception.lejepa.model import (  # noqa: E402
    EncoderConfig,
    LeJEPA,
    PCAProjection,
    images_to_float,
)
from oct_vla.perception.lejepa.sigreg import effective_rank, random_directions, sigreg  # noqa: E402
from oct_vla.perception.lejepa.train import (  # noqa: E402
    CropStore,
    augment,
    lejepa_loss,
    linear_probe,
    sample_targets,
)

TINY = EncoderConfig(dim=32, depth=1, heads=2, predictor_hidden=64, visual_dim=4)


# --------------------------------------------------------------------------- SIGReg


def test_sigreg_is_small_for_standard_normal_samples():
    torch.manual_seed(0)
    value = float(sigreg(torch.randn(1024, 64), step=0))
    assert value < 3.0


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.ones(512, 64),  # collapsed to a point
        torch.zeros(512, 64),
        torch.cat([torch.randn(512, 2), torch.zeros(512, 62)], dim=1),  # rank 2
        3.0 * torch.randn(512, 64),  # right shape, wrong scale
        torch.randn(512, 64) + 1.0,  # right shape, off-centre
    ],
    ids=["constant", "zeros", "low_rank", "scaled", "shifted"],
)
def test_sigreg_is_large_away_from_isotropic_gaussian(embeddings):
    torch.manual_seed(1)
    baseline = float(sigreg(torch.randn(512, 64), step=0))
    assert float(sigreg(embeddings, step=0)) > 20.0 * baseline


def test_sigreg_gradient_flows_and_points_towards_gaussian():
    torch.manual_seed(2)
    x = (0.1 * torch.randn(256, 16)).requires_grad_()
    loss = sigreg(x, step=3)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with torch.no_grad():
        stepped = x - 0.5 * x.grad / x.grad.norm() * x.norm()
    assert float(sigreg(stepped, step=3)) < float(loss.detach())


def test_random_directions_are_unit_and_seeded():
    a = random_directions(32, 8, step=5)
    assert torch.allclose(a.norm(dim=0), torch.ones(8), atol=1e-5)
    assert torch.equal(a, random_directions(32, 8, step=5))
    assert not torch.equal(a, random_directions(32, 8, step=6))


def test_effective_rank_bounds():
    torch.manual_seed(0)
    assert effective_rank(torch.ones(100, 16)) == pytest.approx(1.0)
    assert effective_rank(torch.randn(4000, 16)) > 15.0
    low = torch.randn(1000, 3) @ torch.randn(3, 16)
    assert effective_rank(low) < 3.01


# --------------------------------------------------------------------------- camera


def test_head_camera_matches_sapien():
    """Reference values read from SAPIEN 3.0.0b1's RenderCameraComponent for the
    D435 (fovy 37, 320x240) posed exactly as camera.py poses HEAD_CAMERA_OVERRIDE."""
    camera = default_head_camera()
    expected_k = [[358.64218, 0.0, 160.0], [0.0, 358.64218, 120.0], [0.0, 0.0, 1.0]]
    expected_e = [
        [1.0, 0.0, 0.0, 0.11],
        [0.0, -0.697078, -0.716995, 0.592517],
        [0.0, 0.716995, -0.697078, 1.901032],
    ]
    assert np.allclose(camera.intrinsic, expected_k, atol=1e-4)
    assert np.allclose(camera.extrinsic, expected_e, atol=1e-5)
    assert (camera.width, camera.height) == (320, 240)


def test_override_is_read_from_the_task_source():
    assert head_camera_override() == {
        "position": [-0.11, -0.95, 1.75],
        "forward": [0.0, 0.72, -0.70],
        "left": [-1.0, 0.0, 0.0],
    }


def _synthetic_camera() -> PinholeCamera:
    # At the origin, looking along +x, +y to its left, +z up.
    return camera_from_placement(
        {"position": [0, 0, 0], "forward": [1, 0, 0], "left": [0, 1, 0]},
        fovy_deg=90.0,
        width=200,
        height=100,
    )


def test_projection_of_known_points():
    camera = _synthetic_camera()
    assert np.allclose(camera.intrinsic, intrinsic_from_fovy(90.0, 200, 100))
    fy = 50.0  # (h / 2) / tan(45 deg)
    pixels, depth = camera.project(np.array([[2.0, 0.0, 0.0], [2.0, -0.5, 0.25]]))
    assert np.allclose(depth, [2.0, 2.0])
    assert np.allclose(pixels[0], [100.0, 50.0])
    # Right of the camera (negative "left") is +u; above it is -v.
    assert np.allclose(pixels[1], [100.0 + fy * 0.5 / 2.0, 50.0 - fy * 0.25 / 2.0])


def _object(position, size=(0.2, 0.2, 0.2)) -> ObjectState:
    return ObjectState("obj_0", Pose(position, (0, 0, 0, 1), WORKCELL_FRAME), size, 1.0, 1.0)


def test_crop_is_64x64_uint8_and_centred_on_the_object():
    camera = _synthetic_camera()
    image = np.zeros((100, 200, 3), np.uint8)
    image[45:56, 95:106] = 255  # inside where a 0.6 m cube 2 m ahead lands
    result = crop_object(image, camera, _object((2.0, 0.0, 0.0), size=(0.6, 0.6, 0.6)))
    assert result is not None
    crop, (left, top, side) = result
    assert crop.shape == (64, 64, 3) and crop.dtype == np.uint8
    # Near face at 1.7 m: half-width 50 * 0.3 / 1.7 = 8.8 px about (100, 50).
    assert left <= 91 and top <= 41 and left + side >= 109 and top + side >= 59
    assert crop[32, 32].tolist() == [255, 255, 255]


def test_crop_is_none_off_image_or_behind_or_tiny():
    camera = _synthetic_camera()
    image = np.zeros((100, 200, 3), np.uint8)
    assert crop_object(image, camera, _object((-2.0, 0.0, 0.0))) is None  # behind
    assert crop_object(image, camera, _object((2.0, 5.0, 0.0))) is None  # far left
    assert crop_object(image, camera, _object((60.0, 0.0, 0.0))) is None  # a few px


def test_crop_rejects_wrong_image_type():
    camera = _synthetic_camera()
    with pytest.raises(ValueError):
        crop_object(np.zeros((100, 200, 3), np.float32), camera, _object((2.0, 0.0, 0.0)))


def test_square_is_shifted_inside_not_truncated():
    left, top, side = square_crop_box((0.0, 10.0, 30.0, 40.0), 200, 100, padding=0.15)
    assert (left, top) == (0, 6) and side == 39  # round(5.5) == 6


# --------------------------------------------------------------------------- model


def test_encoder_and_predictor_shapes():
    model = LeJEPA(EncoderConfig())
    images = images_to_float(torch.zeros(2, 64, 64, 3, dtype=torch.uint8))
    z = model(images)
    assert z.shape == (2, 192)
    assert model.predictor(z).shape == (2, 192)


def test_encoder_refuses_unscaled_floats():
    model = LeJEPA(TINY)
    with pytest.raises(TypeError):
        images_to_float(torch.zeros(1, 64, 64, 3))
    with pytest.raises(ValueError):
        model(torch.full((1, 3, 64, 64), 255.0))


def test_augment_shapes_range_and_variation():
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
    a, b = augment(images, generator), augment(images, generator)
    assert a.shape == (4, 3, 64, 64) and a.dtype == torch.float32
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert not torch.allclose(a, b)


def test_one_step_reduces_loss_on_a_toy_batch():
    torch.manual_seed(0)
    model = LeJEPA(TINY)
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (32, 64, 64, 3), dtype=torch.uint8)
    context = augment(images, generator)
    targets = [augment(images, generator) for _ in range(2)]
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05)

    def loss_value():
        loss, parts = lejepa_loss(model, context, targets, lambda_sigreg=0.05, step=0, slices=32)
        return loss, parts

    before, parts = loss_value()
    assert set(parts) >= {"prediction", "sigreg"}
    optimiser.zero_grad()
    before.backward()
    optimiser.step()
    after, _ = loss_value()
    assert float(after.detach()) < float(before.detach())


def test_pca_whitens_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 8)) @ np.diag([5, 4, 3, 2, 1, 0.5, 0.2, 0.1])
    pca = PCAProjection.fit(data, 3)
    projected = pca(data)
    assert projected.shape == (500, 3)
    assert np.allclose(projected.std(axis=0, ddof=1), 1.0, atol=1e-4)
    pca.save(tmp_path / "pca.npz")
    assert np.allclose(PCAProjection.load(tmp_path / "pca.npz")(data), projected)
    with pytest.raises(ValueError):
        PCAProjection.fit(np.ones((50, 8)), 3)


def test_linear_probe_separates_separable_classes():
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 4, size=300)
    features = rng.normal(size=(300, 6)) + 4.0 * np.eye(6)[labels]
    result = linear_probe(features[:200], labels[:200], features[200:], labels[200:])
    assert result["val_acc"] > 0.95


# --------------------------------------------------------------------------- store


def _fake_store(tmp_path):
    frames = np.array([0, 3, 6, 9, 30, 0, 3], np.int32)
    tracks = np.array([0, 0, 0, 0, 0, 1, 1], np.int16)
    count = len(frames)
    np.save(tmp_path / "crops.npy", np.zeros((count, 64, 64, 3), np.uint8))
    np.savez(
        tmp_path / "index.npz",
        episode=np.zeros(count, np.int32),
        seed=np.zeros(count, np.int32),
        frame=frames,
        track=tracks,
        model_id=np.ones(count, np.int16),
        size_xyz=np.ones((count, 3), np.float32),
        square=np.zeros((count, 3), np.int16),
        eef_distance=np.ones(count, np.float32),
    )
    (tmp_path / "meta.json").write_text(json.dumps({"count": count}))
    return CropStore(tmp_path)


def test_temporal_neighbours_stay_within_track_and_gap(tmp_path):
    store = _fake_store(tmp_path)
    offsets, flat = store.temporal_neighbours(3, 15)
    neighbours = [sorted(flat[offsets[i] : offsets[i + 1]].tolist()) for i in range(len(store))]
    assert neighbours == [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], [], [6], [5]]
    picked = sample_targets(np.arange(len(store)), offsets, flat, 4, np.random.default_rng(0))
    assert picked.shape == (4, len(store))
    assert (picked[:, 4] == 4).all()  # no neighbour: the anchor itself
    for row in range(len(store)):
        if neighbours[row]:
            assert set(picked[:, row].tolist()) <= set(neighbours[row])
