from src.plugin import PluginContext

class PluginRegistry:
    def __init__(self):
        self.contexts = {}

    def register(self, plugin_name: str) -> PluginContext:
        ctx = PluginContext(plugin_name)
        self.contexts[plugin_name] = ctx
        return ctx
