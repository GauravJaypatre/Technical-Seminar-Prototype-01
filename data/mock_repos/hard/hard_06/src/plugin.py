class PluginContext:
    # BUG: mutable default dictionary shared across instances!
    def __init__(self, plugin_name: str, metadata: dict = {}):
        self.plugin_name = plugin_name
        self.metadata = metadata

    def set_config(self, key: str, val):
        self.metadata[key] = val
