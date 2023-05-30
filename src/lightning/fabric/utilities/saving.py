# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Any, Callable, Mapping, Union, Dict
import torch.nn as nn

_FILTER_FUNCTION = Callable[[str], bool]
_FILTER_REGEX = str
_FILTER_MAP = Mapping[str, Union[_FILTER_REGEX, _FILTER_FUNCTION]]


def _canonicalize_param_filters(
    state: Dict[str, Any],
    filter: Union[_FILTER_REGEX, _FILTER_FUNCTION, _FILTER_MAP],
) -> Dict[str, _FILTER_FUNCTION]:
    filter = {} if filter is None else filter
    module_map = {name: module for name, module in state.items() if isinstance(module, nn.Module)}
    if not module_map:
        return {}

    if not isinstance(filter, dict):
        # the same filter applies to all models
        filter_map = {name: filter for name in module_map.keys()}
    else:
        filter_map = filter

    unknown_names = set(filter_map.keys()) - set(module_map.keys())
    if unknown_names:
        raise ValueError(f"The filter map contains names that are not present in the state: {unknown_names!r}")

    # add the no-op filter to all models for which user didn't provide a filter
    # filter_map.update({name: lambda _: True for name in module_map.keys() if name not in filter_map})

    # standardize all regex strings to function-style filters
    filter_map = {name: re.compile(filter).match if isinstance(filter, str) else filter for name in filter_map}
    return filter_map


def _filter_state_dict(
    state_dict: Mapping[str, Any],
    filter: Dict[str, _FILTER_FUNCTION],
) -> Mapping[str, Any]:
    return {k: v for k, v in state_dict.items() if filter(k)}
