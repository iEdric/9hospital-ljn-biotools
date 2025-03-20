"""
Utilities to work with a OmegaConf structured config object

From Dolma Toolkit: https://github.com/allenai/dolma/blob/64886d9db15bd99acea9e28740ae20a510875dfb/python/dolma/cli/__init__.py

Author: Luca Soldaini (@soldni)
"""  # noqa: E501

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import Field
from dataclasses import field as dataclass_field
from dataclasses import is_dataclass
from logging import warning
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import smart_open
from necessary import necessary
from omegaconf import MISSING, DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from rich.console import Console
from rich.syntax import Syntax
from yaml import safe_load  # type: ignore

from .errors import DolmaRefineError

__all__ = ["field", "namespace_to_nested_omegaconf", "print_config", "make_cli", "read_config", "to_native_types"]


T = TypeVar("T", bound=Any)
D = TypeVar("D", bound="DataClass")
A = TypeVar("A", bound="ArgumentParser")


def _field_nargs(default: Any) -> Union[Literal["?"], Literal["*"]]:
    # return '+' if _default is iterable but not string/bytes, else 1
    if isinstance(default, (str, bytes)):
        return "?"

    if isinstance(default, Iterable):
        return "*"

    return "?"


def field(default: T = MISSING, help: Optional[str] = None, **extra: Any) -> T:
    metadata = {"help": help, "type": type(default), "default": default, "nargs": _field_nargs(default), **extra}
    return dataclass_field(default_factory=lambda: deepcopy(default), metadata=metadata)


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Field]


def read_config(path: Union[None, str]) -> Dict[str, Any]:
    """Read a configuration file if it exists"""
    if path is None:
        return {}

    try:
        with smart_open.open(path, mode="rt") as f:
            return dict(safe_load(f))
    except FileNotFoundError as ex:
        raise DolmaRefineError(f"Config file not found: {path}") from ex
    except Exception as ex:
        raise DolmaRefineError(f"Error while reading config file: {path}") from ex


def save_config(config: Union[dict, DictConfig, list, ListConfig, DataClass], path: str) -> None:
    """Save a configuration to a file"""
    if isinstance(config, (list, dict)):
        config = om.create(config)
    elif is_dataclass(config):
        config = om.structured(config)

    with smart_open.open(path, mode="wt") as f:
        f.write(om.to_yaml(config))


def _make_parser(parser: A, config: Type[DataClass], prefix: Optional[str] = None) -> A:
    for field_name, dt_field in config.__dataclass_fields__.items():
        # get type from annotations or metadata
        typ_ = config.__annotations__.get(field_name, dt_field.metadata.get("type", MISSING))

        if typ_ is MISSING:
            warning(f"No type annotation for field {field_name} in {config.__name__}")
            continue

        # join prefix and field name
        field_name = f"{prefix}.{field_name}" if prefix else field_name

        # This section here is to handle Optional[T] types; we only care for cases where T is a dataclass
        # So we first check if type is Union since Optional[T] is just a shorthand for Union[T, None]
        # and that the union contains only one non-None type
        if get_origin(typ_) == Union:
            # get all non-None types
            args = [a for a in get_args(typ_) if a is not type(None)]  # noqa: E721

            if len(args) == 1:
                # simple Optional[T] type
                typ_ = args[0]

        # here's where we check if T is a dataclass
        if is_dataclass(typ_):
            # recursively add subparsers
            _make_parser(parser, typ_, prefix=field_name)  # type: ignore
            continue

        if typ_ is bool:
            # for boolean values, we add two arguments: --field_name and --no-field_name
            parser.add_argument(
                f"--{field_name}",
                help=dt_field.metadata.get("help"),
                dest=field_name,
                action="store_true",
                default=MISSING,
            )
            parser.add_argument(
                f"--no-{field_name}",
                help=f"Disable {field_name}",
                dest=field_name,
                action="store_false",
                default=MISSING,
            )
        else:
            # else it's just a normal argument
            parser.add_argument(
                f"--{field_name}",
                help=dt_field.metadata.get("help"),
                nargs=dt_field.metadata.get("nargs", "?"),
                default=MISSING,
            )

    return parser


def make_nested_dict(key: str, value: Any, d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    d = d or {}

    if "." in key:
        key, rest = key.split(".", 1)
        value = make_nested_dict(rest, value, d.get(key))

    # the value was provided (is not MISSING constant) and is not an empty dict or list
    if value != MISSING and (not isinstance(value, (dict, list)) or len(value) > 0):
        d[key] = value

    return d


def to_native_types(obj: Any, resolve: bool = True, throw_on_missing: bool = True, enum_to_str: bool = True) -> Any:
    """Converts an OmegaConf object to native types (dicts, lists, etc.)"""

    # convert dataclass to structured config
    if hasattr(obj, "to_dict"):
        # huggingface objects have a to_dict method, we prefer that
        obj = obj.to_dict()
    elif is_dataclass(obj):
        # we go through structured config instead and hope for the best
        obj = om.to_container(obj)

    if isinstance(obj, DictConfig) or isinstance(obj, ListConfig):
        obj = om.to_container(obj, resolve=resolve, throw_on_missing=throw_on_missing, enum_to_str=enum_to_str)

    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native_types(v) for v in obj]
    else:
        return obj


def namespace_to_nested_omegaconf(args: Namespace, structured: Type[T], config: Optional[dict] = None) -> T:
    nested_config_dict: Dict[str, Any] = {}
    for key, value in vars(args).items():
        nested_config_dict = make_nested_dict(key, value, nested_config_dict)

    untyped_config: DictConfig = om.merge(
        om.create(config or {}), om.create(nested_config_dict)
    )  # pyright: ignore (pylance is confused because om.create might return a DictConfig or a ListConfig)

    # resolve any interpolations in the config
    om.resolve(untyped_config)

    # create structured config from cli dataclass
    base_structured_config: DictConfig = om.structured(structured)

    # merge with options parsed from config file and
    merged_config = om.merge(base_structured_config, untyped_config)

    # check for type
    if not isinstance(merged_config, DictConfig):
        raise DolmaRefineError(f"Expected a DictConfig, got {type(merged_config).__name__}")

    # try resolving all cross references in the config, raise a DolmaConfigError if it fails
    try:
        om.resolve(merged_config)
    except OmegaConfBaseException as ex:
        raise DolmaRefineError(f"Invalid error while parsing key `{ex.full_key}`: {type(ex).__name__}") from ex

    return merged_config  # pyright: ignore


def print_config(config: Any, console: Optional[Console] = None) -> None:
    if not isinstance(config, (DictConfig, ListConfig)):
        config = om.create(config)

    # print the config as yaml using a rich syntax highlighter
    console = console or Console()
    yaml_config = om.to_yaml(config, sort_keys=True).strip()
    highlighted = Syntax(code=yaml_config, lexer="yaml", theme="ansi_dark")
    console.print(highlighted)


def _patch_old_omegaconf():
    """Monkey patch omegaconf below version 2.3.0 to support custom resolver returning
    lists or dicts. Applies patch https://github.com/omry/omegaconf/pull/1093"""

    if necessary(("omegaconf", "2.4.0"), soft=True):
        # no need to patch
        return

    if getattr(_patch_old_omegaconf, "__patched__", False):
        # already patched
        return

    from omegaconf import _impl  # pylint: disable=import-outside-toplevel
    from omegaconf import (  # pylint: disable=import-outside-toplevel
        Container,
        Node,
        ValueNode,
    )
    from omegaconf._utils import (  # noqa: F401  # pylint: disable=import-outside-toplevel
        _ensure_container,
        _get_value,
        is_primitive_container,
        is_structured_config,
    )
    from omegaconf.errors import (  # pylint: disable=import-outside-toplevel
        InterpolationToMissingValueError,
    )
    from omegaconf.nodes import (  # pylint: disable=import-outside-toplevel
        InterpolationResultNode,
    )

    def _resolve_container_value(cfg: Container, key: Any) -> None:
        node = cfg._get_child(key)  # pylint: disable=protected-access
        assert isinstance(node, Node)
        if node._is_interpolation():  # pylint: disable=protected-access
            try:
                resolved = node._dereference_node()  # pylint: disable=protected-access
            except InterpolationToMissingValueError:
                node._set_value(MISSING)  # pylint: disable=protected-access
            else:
                if isinstance(resolved, Container):
                    _impl._resolve(resolved)  # pylint: disable=protected-access
                if isinstance(resolved, InterpolationResultNode):
                    resolved_value = _get_value(resolved)
                    if is_primitive_container(resolved_value) or is_structured_config(resolved_value):
                        resolved = _ensure_container(resolved_value)
                if isinstance(resolved, Container) and isinstance(node, ValueNode):
                    cfg[key] = resolved
                else:
                    node._set_value(_get_value(resolved))  # pylint: disable=protected-access
        else:
            _impl._resolve(node)  # pylint: disable=protected-access

    # set new function and mark as patched
    setattr(_impl, "_resolve_container_value", _resolve_container_value)
    setattr(_patch_old_omegaconf, "__patched__", True)


# actually executes the patch
_patch_old_omegaconf()


def make_cli(config_cls: Type[D], _config_flag: str = "config", _dryrun_flag: str = "dryrun") -> D:
    """Create a CLI parser for a dataclass and parse the arguments into a structured config object."""

    if hasattr(config_cls, _config_flag):
        raise DolmaRefineError(f"`{_config_flag}` is a reserved attribute; remove it from `{config_cls.__name__}`")

    if hasattr(config_cls, _dryrun_flag):
        raise DolmaRefineError(f"`{_dryrun_flag}` is a reserved attribute; remove it from `{config_cls.__name__}`")

    parser = ArgumentParser()
    parser.add_argument(f"-{_config_flag[0]}", f"--{_config_flag}", help="Path to config file", default=None, type=str)
    parser.add_argument(
        f"-{_dryrun_flag[0]}",
        f"--{_dryrun_flag}",
        help="Dry run mode: print config and exit",
        action="store_true",
        default=False,
    )

    parser = _make_parser(parser, config_cls)
    args = parser.parse_args()

    parsed_config: Dict[str, Any] = {}
    if (config_path := getattr(args, _config_flag)) is not None:
        parsed_config = read_config(config_path)
    delattr(args, _config_flag)

    only_dryrun = getattr(args, _dryrun_flag, False)
    delattr(args, _dryrun_flag)

    full_config = namespace_to_nested_omegaconf(args, config_cls, parsed_config)

    print_config(full_config)

    if only_dryrun:
        exit(0)

    return full_config
