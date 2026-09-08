from src.task import Task, TaskState

class TaskScheduler:
    def __init__(self):
        self.tasks = {}

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def add_dependency(self, parent_id: str, child_id: str):
        self.tasks[parent_id].dependents.append(self.tasks[child_id])

    def cancel_task(self, task_id: str):
        if task_id not in self.tasks:
            return
        target = self.tasks[task_id]
        target.cancel()
        # BUG: does not cascade cancellation to dependents
