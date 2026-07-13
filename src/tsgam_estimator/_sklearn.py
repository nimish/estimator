# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Small scikit-learn parameter adapter for nested config dataclasses."""

from __future__ import annotations

from dataclasses import Field
from typing import ClassVar, Self, cast


class SklearnConfigMixin:
    """Expose dataclass fields through scikit-learn's parameter protocol."""

    __dataclass_fields__: ClassVar[dict[str, Field[object]]]

    def get_params(self, deep: bool = True) -> dict[str, object]:
        params = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.init
        }
        if not deep:
            return params

        nested_params: dict[str, object] = {}
        for name, value in params.items():
            self._collect_nested_params(name, value, nested_params)
        params.update(nested_params)
        return params

    @classmethod
    def _collect_nested_params(
        cls,
        prefix: str,
        value: object,
        params: dict[str, object],
    ) -> None:
        if isinstance(value, SklearnConfigMixin):
            for name, nested_value in value.get_params(deep=True).items():
                params[f"{prefix}__{name}"] = nested_value
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, SklearnConfigMixin):
                    item_prefix = f"{prefix}__{index}"
                    params[item_prefix] = item
                    cls._collect_nested_params(item_prefix, item, params)

    def set_params(self, **params: object) -> Self:
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        for name, value in params.items():
            if name not in valid_params:
                valid_names = sorted(valid_params)
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {valid_names!r}."
                )
            path = name.split("__")
            self._set_nested_value(self, path, value, full_name=name)
        return self

    @classmethod
    def _set_nested_value(
        cls,
        target: object,
        path: list[str],
        value: object,
        *,
        full_name: str,
    ) -> None:
        component = path[0]
        if isinstance(target, list):
            mutable_target = cast(list[object], target)
            try:
                index = int(component)
            except ValueError as error:
                raise ValueError(
                    f"Invalid list index {component!r} in parameter {full_name!r}."
                ) from error
            if len(path) == 1:
                mutable_target[index] = value
            else:
                cls._set_nested_value(
                    mutable_target[index],
                    path[1:],
                    value,
                    full_name=full_name,
                )
            return

        if len(path) == 1:
            setattr(target, component, value)
            return
        cls._set_nested_value(
            getattr(target, component),
            path[1:],
            value,
            full_name=full_name,
        )
