import gzip

from oct_vla.core.action import Action, ArmAction
from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.episode import Episode, Sample
from oct_vla.data.store import read_episode, write_episode

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _pose(x: float = 0.0) -> Pose:
    return Pose((x, 0.1, 0.2), IDENTITY, WORKCELL_FRAME)


def _frame(seed: int) -> RGBFrame:
    # Distinct, non-repeating bytes per frame so a byte-slice-offset bug
    # in read_episode would corrupt the comparison rather than pass by luck.
    data = bytes((seed + i) % 256 for i in range(2 * 2 * 3))
    return RGBFrame(2, 2, data)


def _observation(x: float, seed: int) -> RobotObservation:
    return RobotObservation(
        round(x, 6),
        EEFState(ArmState(_pose(x), 0.2), ArmState(_pose(-x), 0.8)),
        _frame(seed),
        _frame(seed + 1),
        _frame(seed + 2),
    )


def _scene(track_id: str = "box_0") -> ObjectScene:
    return ObjectScene(
        0.0,
        (
            ObjectState(
                track_id,
                _pose(),
                (0.05, 0.06, 0.07),
                0.9,
                0.8,
                support_surface="upper_shelf",
                mask=b"\x01\x02\x03",
                embedding=(0.1, 0.2, 0.3),
            ),
        ),
    )


def _action(x: float) -> Action:
    return Action(
        ArmAction((x, 0.0, 0.0), (0.0, 0.0, x), 0.3),
        ArmAction((-x, 0.0, 0.0), (0.0, 0.0, 0.0), 0.7),
    )


def _context(target: str = "box_0") -> TaskContext:
    return TaskContext("stock the shelf", target, previous_neighbor_track_id=None, phase="lift")


def _episode() -> Episode:
    samples = tuple(
        Sample(
            timestamp=round(index * 0.1, 6),
            observation=_observation(index * 0.01, index * 10),
            scene=_scene(),
            context=_context(),
            action=_action(index * 0.001),
            phase=f"phase_{index}",
        )
        for index in range(4)
    )
    return Episode(
        seed=42,
        instruction="stock the shelf",
        samples=samples,
        success=True,
        metadata={"entrypoint": "shelf_restock", "commit": "abc123"},
    )


def test_write_then_read_round_trips_an_episode_exactly(tmp_path):
    episode = _episode()

    write_episode(episode, tmp_path / "episode_0")
    restored = read_episode(tmp_path / "episode_0")

    assert restored.seed == episode.seed
    assert restored.instruction == episode.instruction
    assert restored.success == episode.success
    assert dict(restored.metadata) == dict(episode.metadata)
    assert len(restored.samples) == len(episode.samples)

    for original, back in zip(episode.samples, restored.samples, strict=True):
        assert back.timestamp == original.timestamp
        assert back.phase == original.phase
        assert back.action.to_vector() == original.action.to_vector()
        assert back.context == original.context
        assert back.scene == original.scene
        assert back.observation.eef == original.observation.eef
        assert back.observation.head_rgb == original.observation.head_rgb
        assert back.observation.left_wrist_rgb == original.observation.left_wrist_rgb
        assert back.observation.right_wrist_rgb == original.observation.right_wrist_rgb


def test_write_episode_creates_gzip_compressed_camera_files(tmp_path):
    episode = _episode()

    directory = write_episode(episode, tmp_path / "episode_0")

    for name in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
        path = directory / f"{name}.rgb.gz"
        assert path.exists()
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
        # 4 samples x 2x2x3 bytes per frame.
        assert len(raw) == 4 * 2 * 2 * 3


def test_read_episode_reconstructs_frame_i_from_a_known_byte_slice(tmp_path):
    episode = _episode()
    write_episode(episode, tmp_path / "episode_0")

    restored = read_episode(tmp_path / "episode_0")

    for index, sample in enumerate(restored.samples):
        assert sample.observation.head_rgb == episode.samples[index].observation.head_rgb
