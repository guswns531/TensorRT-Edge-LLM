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
"""Run frozen HTTP clients with one fresh detached Docker server per case."""

import argparse
import http.client
import json
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.request


def run_case(case, destination, ready_timeout):
    """Retain commands and logs, retiring only the newly created container."""
    destination.mkdir(parents=True, exist_ok=False)
    command = [x.replace('{output}', str(destination)) for x in case['client']]
    (destination / 'commands.json').write_text(
        json.dumps(
            {
                'server': case['server'],
                'client': command,
                'health_url': case['health_url']
            },
            indent=2))
    container = subprocess.check_output(case['server'], text=True).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise RuntimeError('Server command must return one detached Docker ID')
    (destination / 'container-id.txt').write_text(container + '\n')
    try:
        deadline = time.monotonic() + ready_timeout
        while True:
            try:
                with urllib.request.urlopen(case['health_url'], timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException):
                pass
            if subprocess.check_output(
                ['docker', 'inspect', '-f', '{{.State.Running}}', container],
                    text=True).strip() != 'true':
                raise RuntimeError('Server exited before becoming healthy')
            if time.monotonic() >= deadline:
                raise TimeoutError('Server health deadline exceeded')
            time.sleep(1)
        with (destination / 'client.log').open('w') as log:
            subprocess.run(command,
                           stdout=log,
                           stderr=subprocess.STDOUT,
                           check=True)
    finally:
        subprocess.run(['docker', 'stop', '--time', '10', container],
                       check=False)
        with (destination / 'server.log').open('w') as log:
            subprocess.run(['docker', 'logs', container],
                           stdout=log,
                           stderr=subprocess.STDOUT,
                           check=False)
        subprocess.run(['docker', 'rm', container], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commands', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--ready-timeout', type=float, default=600)
    args = parser.parse_args()
    for case in json.loads(args.commands.read_text()):
        if not re.fullmatch('[a-zA-Z0-9_-]+', case['name']):
            raise ValueError('Invalid case directory name')
        if case['server'][:2] != ['docker', 'run'] or '--rm' in case['server']:
            raise ValueError(
                'Server must use docker run without --rm to retain exit logs')
        run_case(case, args.output_dir / case['name'], args.ready_timeout)
        print('Completed ' + case['name'], flush=True)


if __name__ == '__main__':
    main()
