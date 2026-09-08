class TaskState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class Task:
    def __init__(self, task_id: str, action):
        self.task_id = task_id
        self.action = action
        self.state = TaskState.PENDING
        self.dependents = []

    def cancel(self):
        self.state = TaskState.CANCELLED
