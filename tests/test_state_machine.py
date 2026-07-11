from ev_vision.state_machine import Event, State, VisionStateMachine


def advance_to_track(machine: VisionStateMachine) -> None:
    for event in (Event.BOOT_COMPLETE, Event.START, Event.BOARD_FOUND, Event.BOARD_STABLE, Event.LASER_CALIBRATED, Event.CENTERED, Event.TRACK_REQUESTED):
        machine.handle(event)


def test_nominal_track_path() -> None:
    machine = VisionStateMachine()
    advance_to_track(machine)
    assert machine.state is State.TRACK
    assert machine.policy.motion_allowed
    assert machine.policy.laser_allowed


def test_circle_requires_valid_homography_and_sync() -> None:
    machine = VisionStateMachine()
    advance_to_track(machine)
    machine.handle(Event.CIRCLE_REQUESTED, homography_valid=False, circle_sync_valid=True)
    assert machine.state is State.TRACK
    machine.handle(Event.CIRCLE_REQUESTED, homography_valid=True, circle_sync_valid=True)
    assert machine.state is State.CIRCLE


def test_short_loss_enters_recover_and_reacquires() -> None:
    machine = VisionStateMachine()
    advance_to_track(machine)
    machine.handle(Event.TARGET_LOST)
    assert machine.state is State.RECOVER
    assert not machine.policy.laser_allowed
    machine.handle(Event.BOARD_FOUND)
    assert machine.state is State.ACQUIRE


def test_fault_is_latched_and_fail_closed_until_reset() -> None:
    machine = VisionStateMachine()
    advance_to_track(machine)
    machine.handle(Event.FAULT)
    assert machine.state is State.FAULT
    assert not machine.policy.motion_allowed
    assert not machine.policy.laser_allowed
    machine.handle(Event.BOARD_FOUND)
    assert machine.state is State.FAULT
    machine.handle(Event.RESET)
    assert machine.state is State.SAFE
