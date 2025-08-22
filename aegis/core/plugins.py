# aegis/core/plugins.py

import importlib
import pkgutil
import inspect
from typing import Type, List

class AegisPlugin:
    """A base class that all plugins must inherit from."""
    pass

class PluginManager:
    """
    Discovers and loads all available Aegis plugins from specified packages.
    """
    def __init__(self, plugin_packages: List[str]):
        self.plugins: List[Type[AegisPlugin]] = []
        self.plugin_packages = plugin_packages
        self._discover_plugins()

    def _discover_plugins(self):
        """
        Dynamically finds and imports all classes that inherit from AegisPlugin
        within the given packages.
        """
        print("🔍 Discovering plugins...")
        discovered_plugins = set() # Use a set to avoid duplicates

        for package_name in self.plugin_packages:
            package = importlib.import_module(package_name)
            for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
                try:
                    module = importlib.import_module(module_name)
                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)
                        if (
                            isinstance(attribute, type) and
                            issubclass(attribute, AegisPlugin) and
                            attribute is not AegisPlugin and
                            not inspect.isabstract(attribute) # FIX: Ignore abstract base classes
                        ):
                            if attribute not in discovered_plugins:
                                discovered_plugins.add(attribute)
                                print(f"  -> Found plugin: {attribute.__name__}")
                except Exception as e:
                    print(f"Could not import module {module_name}: {e}")
        
        self.plugins = list(discovered_plugins)

    def get_plugins(self, plugin_type: Type[AegisPlugin]) -> List[AegisPlugin]:
        """
        Gets all instantiated plugins of a specific type.
        """
        return [plugin() for plugin in self.plugins if issubclass(plugin, plugin_type)]
