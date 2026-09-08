import pytest
from src.task import Task, TaskState
from src.scheduler import TaskScheduler

def test_cancel_cascades_to_children():
    scheduler = TaskScheduler()
    t1 = Task("t1", lambda: 1)
    t2 = Task("t2", lambda: 2)
    t3 = Task("t3", lambda: 3)

    scheduler.add_task(t1)
    scheduler.add_task(t2)
    scheduler.add_task(t3)

    scheduler.add_dependency("t1", "t2")
    scheduler.add_dependency("t2", "t3")

    # FAILS ON UNPATCHED CODE
    scheduler.cancel_task("t1")
    assert t1.state == TaskState.CANCELLED
    assert t2.state == TaskState.CANCELLED
    assert t3.state == TaskState.CANCELLED
