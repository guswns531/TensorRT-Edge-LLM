# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    'container_matrix',
    Path(__file__).parents[2] /
    'benchmarks/phase_serving/run_container_http_matrix.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('failure', [False, True])
def test_reset_retry_and_exact_container_cleanup(tmp_path, monkeypatch,
                                                 failure):
    container = 'a' * 64
    output = Mock(side_effect=[container, 'true'])
    run = Mock()
    if failure:
        run.side_effect = [
            subprocess.CalledProcessError(1, ['client']), None, None, None
        ]
    health = Mock(side_effect=[ConnectionResetError(), MagicMock()])
    monkeypatch.setattr(MODULE.subprocess, 'check_output', output)
    monkeypatch.setattr(MODULE.subprocess, 'run', run)
    monkeypatch.setattr(MODULE.urllib.request, 'urlopen', health)
    monkeypatch.setattr(MODULE.time, 'sleep', lambda _: None)
    case = {
        'server': ['docker', 'run', '--detach', 'image'],
        'client': ['client', '{output}/client'],
        'health_url': 'http://localhost/health'
    }
    destination = tmp_path / 'case'
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            MODULE.run_case(case, destination, 10)
    else:
        MODULE.run_case(case, destination, 10)
    commands = [call.args[0] for call in run.call_args_list]
    assert commands == [['client', str(destination / 'client')],
                        ['docker', 'stop', '--time', '10', container],
                        ['docker', 'logs', container],
                        ['docker', 'rm', container]]
    assert health.call_count == 2
    assert (destination / 'commands.json').is_file()
    assert (destination / 'server.log').is_file()


def test_refuses_existing_results(tmp_path, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(MODULE.subprocess, 'check_output', launch)
    with pytest.raises(FileExistsError):
        MODULE.run_case({}, tmp_path, 1)
    launch.assert_not_called()
