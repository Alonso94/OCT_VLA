"""Record a RoboTwin built-in task through our canonical pipeline.

Our own task is expensive to add: a 1150-line oracle, a success checker, a
manager, a spec. A RoboTwin built-in supplies all four -- 51 of its 52 tasks
ship both `play_once` and `check_success` -- so the only thing missing is a way
to get its demonstration into our `Episode` format.

The obstacle is that `play_once` does not go through our port. It drives the
scene with `Base_Task`'s own helpers, which call `self.scene.step()` directly,
while `EpisodeRecorder` captures as a side effect of `NativePort.tick()`. Point
the recorder at a built-in and it captures nothing at all.

But the same control loop already has a recording hook: `take_dense_action`
calls `self._take_picture()` every `save_freq` steps, which is how RoboTwin
writes its own datasets. Overriding that method captures exactly the frames
RoboTwin would have saved, at a cadence we choose, while its oracle runs
untouched.

Two hazards this module exists to handle, both silent:

* **`move()` returns early once planning has failed.** `Base_Task.move` opens
  with `if self.plan_success is False: return False`, so after one planning
  failure every subsequent motion is skipped and `play_once` still returns
  normally. Our own port raises instead. A loop that does not check
  `plan_success` afterwards records a stationary robot as a successful
  demonstration.
* **A built-in reports no motion records**, so `label_spans` -- which rebuilds
  phase boundaries from per-motion tick counts -- has nothing to work from, and
  every captured frame would be dropped for falling outside every span.
  `single_span` labels the whole demonstration as one phase instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.data.entity_tokens import EntitySupport


@dataclass(frozen=True)
class ObjectBinding:
    """Where one canonical object comes from on a RoboTwin task instance.

    RoboTwin tasks expose their objects as ad-hoc attributes (`self.container`,
    `self.plate`); our pipeline wants a `track_id -> TrackedObject` mapping. The
    binding is per task and declared rather than discovered, because guessing
    which attributes are manipulable objects would be wrong quietly.
    """

    #: Attribute on the task instance holding the actor.
    attribute: str
    #: Stable id for this object in every recorded scene.
    track_id: str
    #: Attributes holding the asset name and variant, for size lookup. Either
    #: may be a literal instead, for a task that hardcodes one.
    modelname: str
    model_id: int | str = 0

    def resolve(self, task: Any) -> tuple[Any, str, int]:
        actor = getattr(task, self.attribute, None)
        if actor is None:
            raise AttributeError(
                f"task {type(task).__name__} has no attribute {self.attribute!r}; "
                "the binding names an object this task does not spawn"
            )
        name = getattr(task, self.modelname, self.modelname)
        variant = self.model_id
        if isinstance(variant, str):
            variant = getattr(task, variant)
        return actor, str(name), int(variant)


@dataclass(frozen=True)
class BuiltinTaskSpec:
    """Everything needed to record one RoboTwin built-in task."""

    #: The RoboTwin task module/class name, e.g. "place_container_plate".
    task_name: str
    #: Language instruction stored on every sample.
    instruction: str
    #: Which object the task is about; must be one of `objects`' track_ids.
    target_track_id: str
    objects: tuple[ObjectBinding, ...]
    #: Height of the work surface, for the support entity. RoboTwin's tabletop
    #: tasks all stand on one table rather than our two shelf decks.
    table_top_z: float = 0.74
    table_size_xyz: tuple[float, float, float] = (1.2, 0.8, 0.02)
    #: One label for the whole demonstration; a built-in reports no phases.
    phase: str = "demonstration"

    def __post_init__(self) -> None:
        ids = [binding.track_id for binding in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate track_id in {self.task_name} bindings: {ids}")
        if self.target_track_id not in ids:
            raise ValueError(
                f"target_track_id {self.target_track_id!r} is not among {ids}; "
                "the recorded context would name an object no scene contains, "
                "and validate_episode would reject every sample"
            )

    def supports(self) -> tuple[EntitySupport, ...]:
        """The work surface, as an entity in the same set as the objects."""
        return (
            EntitySupport(
                Pose((0.0, 0.0, self.table_top_z), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME),
                self.table_size_xyz,
            ),
        )


#: `place_container_plate`: move a bowl or a cup onto a static plate.
#:
#: Chosen as the easy half of an easy/hard pair. It is pick-and-place, so it is
#: structurally comparable to shelf restock; it sits at RoboTwin's 400-step easy
#: floor; and its container is drawn from **two categories** -- `002_bowl` (4
#: meshes) or `021_cup` (7) -- where our own corpus has one asset whose variants
#: are distinguished only by size. That makes this the first corpus where a
#: semantic channel can carry something geometry does not.
PLACE_CONTAINER_PLATE = BuiltinTaskSpec(
    task_name="place_container_plate",
    instruction="Place the container on the plate.",
    target_track_id="container",
    objects=(
        ObjectBinding("container", "container", modelname="actor_name", model_id="container_id"),
        ObjectBinding("plate", "plate", modelname="003_plate", model_id="plate_id"),
    ),
)

SPECS: dict[str, BuiltinTaskSpec] = {PLACE_CONTAINER_PLATE.task_name: PLACE_CONTAINER_PLATE}


def spec_for(task_name: str) -> BuiltinTaskSpec:
    try:
        return SPECS[task_name]
    except KeyError:
        raise ValueError(
            f"no binding for RoboTwin task {task_name!r}; known: {sorted(SPECS)}"
        ) from None


def tracked_objects(task: Any, spec: BuiltinTaskSpec, assets_root: Any) -> dict[str, Any]:
    """Build the `track_id -> TrackedObject` map our collection loop expects.

    Sizes come from the asset metadata rather than the live actor, for the same
    reason `shelf_restock` does it: `create_actor` overwrites its own `scale`
    argument with the value inside `model_data<N>.json`, so the mesh is the only
    authority on how big the object actually is.
    """
    from oct_vla.robots.robotwin.assets import UPRIGHT_ROTATION, upright_center_offset, upright_size
    from oct_vla.tasks.shelf_restock.robotwin_env import TrackedObject

    tracked: dict[str, Any] = {}
    for binding in spec.objects:
        actor, name, variant = binding.resolve(task)
        tracked[binding.track_id] = TrackedObject(
            actor,
            upright_size(name, variant, assets_root),
            UPRIGHT_ROTATION,
            upright_center_offset(name, variant, assets_root),
            category=name,
        )
    return tracked


def recording_task_class(task_name: str, on_frame: Callable[[], None]) -> type:
    """A subclass of a RoboTwin built-in that calls `on_frame` per captured step.

    `_take_picture` is RoboTwin's own per-frame hook, reached from
    `take_dense_action` every `save_freq` control steps. Overriding it -- rather
    than reimplementing `play_once` -- means the stock oracle drives the robot
    and we only observe, so nothing about the demonstration changes.
    """
    from oct_vla.robots.robotwin.native import _load_task_class

    base = _load_task_class(task_name)

    class RecordingTask(base):  # type: ignore[misc, valid-type]
        def _take_picture(self) -> None:  # noqa: D102 - RoboTwin's own hook
            on_frame()

    RecordingTask.__name__ = f"Recording{base.__name__}"
    RecordingTask.__qualname__ = RecordingTask.__name__
    return RecordingTask
