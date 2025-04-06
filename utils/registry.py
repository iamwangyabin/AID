import difflib
from typing import Any, Callable, Dict, List, Optional, Type, Union

from rich.console import Console
from rich.table import Table


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: True if it is a sequence of expected type.
    """
    if seq_type is None:
        exp_seq_type = (list, tuple)
    else:
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    @property
    def name(self):
        return self._name

    def has(self, name: str) -> bool:
        """Check if a name is in the registry."""
        return name in self._module_dict

    @property
    def module_dict(self):
        return self._module_dict

    def _suggest_correction(self, input_string: str) -> Optional[str]:
        """Suggest the most similar string from the registered modules."""
        suggestions = difflib.get_close_matches(input_string, self._module_dict.keys(), n=1, cutoff=0.6)
        if suggestions:
            return suggestions[0]
        return None

    def get(self, name):
        if name in self._module_dict:
            return self._module_dict[name]
        suggestion = self._suggest_correction(name)
        if suggestion:
            raise KeyError(f'"{name}" is not registered in {self.name}. Did you mean "{suggestion}"?')
        else:
            raise KeyError(f'"{name}" is not registered in {self.name} and no similar names were found.')

    def _register_module(
        self,
        module: Type,
        module_name: Optional[Union[str, List[str]]] = None,
        force: bool = False
    ) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered.
            module_name (str or list of str, optional): The module name(s).
            force (bool): Whether to override existing. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(
        self,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        module: Optional[Type] = None
    ) -> Union[type, Callable]:
        """Register a module.

        Can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name(s).
            force (bool): Whether to override existing. Defaults to False.
            module (type, optional): Module to register.

        Returns:
            The registered module or a decorator.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, a str, or a sequence of str, '
                f'but got {type(name)}')

        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def build(self, name: str, *args, **kwargs) -> Any:
        """Build an instance by name."""
        return self.get(name)(*args, **kwargs)


# Global registries
MODELS = Registry('MODELS')
DATASETS = Registry('DATASETS')
POSTFUNCS = Registry('POSTFUNCS')