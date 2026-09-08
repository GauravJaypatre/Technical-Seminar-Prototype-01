class DispatchAggregateError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"{len(errors)} listener(s) raised an exception.")

class EventDispatcher:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_name: str, listener):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(listener)

    def dispatch(self, event_name: str, data=None):
        if event_name not in self._listeners:
            return
        # BUG: stops iteration immediately on first exception
        for listener in self._listeners[event_name]:
            listener(data)
