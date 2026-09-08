from collections.abc import MutableMapping

class CaseInsensitiveDict(MutableMapping):
    """A dictionary with case-insensitive string keys."""
    def __init__(self, data=None):
        self._store = {}
        if data:
            self.update(data)

    def __setitem__(self, key: str, value):
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key: str):
        return self._store[key.lower()][1]

    def __delitem__(self, key: str):
        del self._store[key.lower()]

    def __iter__(self):
        return (orig_key for orig_key, _ in self._store.values())

    def __len__(self):
        return len(self._store)

    # BUG: get and __contains__ do not normalize key to lowercase
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._store

    def get(self, key: str, default=None):
        if key in self._store:
            return self._store[key][1]
        return default
