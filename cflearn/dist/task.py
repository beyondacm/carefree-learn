import os
import sys
import platform

import numpy as np

from typing import *
from cftool.misc import Saving, shallow_copy_dict

from ..types import data_type

IS_LINUX = platform.system() == "Linux"


class Task:
    def __init__(self, idx: int, model: str, identifier: str, temp_folder: str):
        self.idx = idx
        self.model = model
        self.identifier = identifier
        self.temp_folder = temp_folder
        self.config: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"Task({self.identifier}_{self.idx})"

    __repr__ = __str__

    @property
    def data_folder(self) -> str:
        if self.model == "data":
            folder = self.temp_folder
        else:
            folder = os.path.join(self.temp_folder, "__data__")
        folder = os.path.join(folder, self.identifier, str(self.idx))
        folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def config_folder(self) -> str:
        folder = os.path.abspath(
            os.path.join(
                self.temp_folder,
                "__config__",
                self.identifier,
                str(self.idx),
            )
        )
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def saving_folder(self) -> str:
        folder = os.path.join(self.temp_folder, self.identifier, str(self.idx))
        folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def run_command(self) -> str:
        python = sys.executable
        return f"{python} -m {'.'.join(['cflearn', 'dist', 'run'])}"

    def prepare(
        self,
        x: data_type = None,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        external: bool,
        data_task: Optional["Task"] = None,
        mlflow_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Task":
        kwargs = shallow_copy_dict(kwargs)
        kwargs["model"] = self.model
        kwargs["data_folder"] = self.data_folder
        kwargs["logging_folder"] = self.saving_folder
        if mlflow_config is not None:
            kwargs["mlflow_config"] = mlflow_config
        if external:
            kwargs["trigger_logging"] = True
            if data_task is None:
                if not isinstance(x, np.ndarray):
                    kwargs["x"], kwargs["y"] = x, y
                    kwargs["x_cv"], kwargs["y_cv"] = x_cv, y_cv
                else:
                    self.dump_data(x, y)
                    self.dump_data(x_cv, y_cv, "_cv")
        self.config = kwargs
        return self

    # external run

    def dump_data(self, x: data_type, y: data_type = None, postfix: str = "") -> None:
        for key, value in zip([f"x{postfix}", f"y{postfix}"], [x, y]):
            if value is None:
                continue
            np.save(os.path.join(self.data_folder, f"{key}.npy"), value)

    def fetch_data(self, postfix: str = "") -> Tuple[data_type, data_type]:
        data = []
        for key in [f"x{postfix}", f"y{postfix}"]:
            file = os.path.join(self.data_folder, f"{key}.npy")
            data.append(None if not os.path.isfile(file) else np.load(file))
        return data[0], data[1]

    def dump_config(self, config: Dict[str, Any]) -> "Task":
        Saving.save_dict(config, "config", self.config_folder)
        return self

    def run_external(self, cuda: Optional[int] = None) -> "Task":
        config = shallow_copy_dict(self.config)
        config["cuda"] = cuda
        self.dump_config(config)
        os.system(f"{self.run_command} --config_folder {self.config_folder}")
        return self

    # internal fit

    def fit(
        self,
        make: Callable,
        save: Callable,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        prepare: bool = True,
        cuda: Optional[int] = None,
        sample_weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "Task":
        if prepare:
            kwargs = shallow_copy_dict(kwargs)
            self.prepare(x, y, x_cv, y_cv, external=False, **shallow_copy_dict(kwargs))
        assert self.config is not None
        m = make(cuda=cuda, **shallow_copy_dict(self.config))
        m.fit(x, y, x_cv, y_cv, sample_weights=sample_weights)
        save(m, saving_folder=self.saving_folder)
        return self

    # save & load

    def save(self, saving_folder: str) -> "Task":
        os.makedirs(saving_folder, exist_ok=True)
        Saving.save_dict(
            {
                "idx": self.idx,
                "model": self.model,
                "identifier": self.identifier,
                "temp_folder": self.temp_folder,
                "config": self.config,
            },
            "kwargs",
            saving_folder,
        )
        return self

    @classmethod
    def load(cls, saving_folder: str) -> "Task":
        kwargs = Saving.load_dict("kwargs", saving_folder)
        config = kwargs.pop("config")
        task = cls(**shallow_copy_dict(kwargs))
        task.config = config
        return task

    # special

    @classmethod
    def data_task(cls, i: int, identifier: str, experiments: Any) -> "Task":
        data_folder = os.path.join(experiments.temp_folder, "__data__")
        return cls(i, "data", identifier, data_folder)


__all__ = ["Task"]
