from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.data.object_tokens import ObjectTokenSpec, object_tokens


def _object(track_id, x):
    return ObjectState(
        track_id=track_id,
        pose=Pose((x, 0.0, 0.8), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME),
        size_xyz=(0.05, 0.04, 0.1),
        visibility=0.8,
        confidence=0.9,
    )


def test_tokens_are_padded_and_roles_are_derived_from_context():
    scene = ObjectScene(
        0.0,
        (_object("other", 3.0), _object("previous", 2.0), _object("target", 1.0)),
    )
    context = TaskContext("restock", "target", "previous")
    tokens, mask = object_tokens(scene, context, spec=ObjectTokenSpec(max_objects=4))

    assert mask == (True, True, True, False)
    assert tokens[0][0] == 1.0 and tokens[0][-3:] == (1.0, 0.0, 0.0)
    assert tokens[1][0] == 2.0 and tokens[1][-3:] == (0.0, 1.0, 0.0)
    assert tokens[2][0] == 3.0 and tokens[2][-3:] == (0.0, 0.0, 1.0)
    assert tokens[3] == (0.0,) * 15
