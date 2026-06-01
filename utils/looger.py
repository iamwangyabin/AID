import os
from collections.abc import Mapping

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - omegaconf is a project dependency.
    OmegaConf = None


DEFAULT_PROJECT = "DeepfakeDetection"
DEFAULT_JOB_TYPE = "train"


class ExperimentLogger:
    def __init__(self, backend, run=None):
        self.backend = backend
        self.run = run

    def log(self, metrics, step=None):
        if self.run is None:
            return
        if step is None:
            self.run.log(metrics)
        else:
            self.run.log(metrics, step=step)

    def finish(self):
        if self.run is None:
            return
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            finish()


def build_experiment_logger(conf, run_name):
    logger_conf = _logger_config(conf)
    backend = _normalize_backend(
        os.getenv("AID_LOGGER_BACKEND", _cfg_get(logger_conf, "backend", "auto"))
    )

    if backend == "disabled":
        return ExperimentLogger("disabled")
    if backend == "wandb":
        return _build_wandb_logger(conf, logger_conf, run_name)
    if backend == "swanlab":
        return _build_swanlab_logger(conf, logger_conf, run_name)
    if backend == "auto":
        return _build_auto_logger(conf, logger_conf, run_name)

    raise ValueError(
        "Unsupported logger backend "
        f'"{backend}". Choose from auto, wandb, swanlab, or disabled.'
    )


def _build_auto_logger(conf, logger_conf, run_name):
    try:
        import swanlab

        wandb_mode = _cfg_get(logger_conf, "wandb_mode", _cfg_get(logger_conf, "mode", "offline"))
        os.environ["WANDB_MODE"] = str(wandb_mode)
        swanlab.sync_wandb(
            mode=_cfg_get(logger_conf, "swanlab_mode", "cloud"),
            workspace=_cfg_get(logger_conf, "workspace", None),
            logdir=_cfg_get(logger_conf, "logdir", None),
        )
        return _build_wandb_logger(conf, logger_conf, run_name, backend="swanlab_wandb")
    except ImportError:
        return _build_wandb_logger(conf, logger_conf, run_name)


def _build_wandb_logger(conf, logger_conf, run_name, backend="wandb"):
    import wandb

    init_kwargs = {
        "name": _cfg_get(logger_conf, "name", run_name),
        "project": _cfg_get(logger_conf, "project", DEFAULT_PROJECT),
        "job_type": _cfg_get(logger_conf, "job_type", DEFAULT_JOB_TYPE),
        "group": _cfg_get(logger_conf, "group", _cfg_get(conf, "name", None)),
    }
    for src_key, dst_key in (
        ("entity", "entity"),
        ("dir", "dir"),
        ("mode", "mode"),
        ("notes", "notes"),
        ("tags", "tags"),
        ("id", "id"),
        ("resume", "resume"),
    ):
        value = _cfg_get(logger_conf, src_key, None)
        if value is not None:
            init_kwargs[dst_key] = value

    config = _resolved_logger_config(conf, logger_conf)
    if config is not None:
        init_kwargs["config"] = config

    return ExperimentLogger(backend, wandb.init(**init_kwargs))


def _build_swanlab_logger(conf, logger_conf, run_name):
    import swanlab

    init_kwargs = {
        "experiment_name": _cfg_get(logger_conf, "name", run_name),
        "project": _cfg_get(logger_conf, "project", DEFAULT_PROJECT),
        "job_type": _cfg_get(logger_conf, "job_type", DEFAULT_JOB_TYPE),
        "group": _cfg_get(logger_conf, "group", _cfg_get(conf, "name", None)),
    }
    for key in (
        "workspace",
        "description",
        "tags",
        "logdir",
        "mode",
        "public",
        "id",
        "resume",
        "reinit",
    ):
        value = _cfg_get(logger_conf, key, None)
        if value is not None:
            init_kwargs[key] = value

    config = _resolved_logger_config(conf, logger_conf)
    if config is not None:
        init_kwargs["config"] = config

    return ExperimentLogger("swanlab", swanlab.init(**init_kwargs))


def _logger_config(conf):
    logger_conf = _cfg_get(conf, "logger", None)
    if logger_conf is None:
        logger_conf = _cfg_get(conf, "log", None)
    if logger_conf is not None:
        return logger_conf

    train_conf = _cfg_get(conf, "train", None)
    logger_conf = _cfg_get(train_conf, "logger", None)
    if logger_conf is None:
        logger_conf = _cfg_get(train_conf, "log", {})
    return logger_conf


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    get = getattr(config, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _normalize_backend(value):
    backend = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "wb": "wandb",
        "sl": "swanlab",
        "none": "disabled",
        "off": "disabled",
        "false": "disabled",
        "no": "disabled",
        "swanlab_wandb": "auto",
        "sync_wandb": "auto",
        "wandb_swanlab": "auto",
    }
    return aliases.get(backend, backend)


def _resolved_logger_config(conf, logger_conf):
    log_config = _cfg_get(logger_conf, "config", None)
    if log_config is True:
        return _to_container(conf)
    if log_config in (None, False):
        return None
    return _to_container(log_config)


def _to_container(value):
    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value
