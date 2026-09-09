# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Render a shared-time-axis scientific figure from measured CUDA intervals."""

import argparse
import collections
import csv
import json
import pathlib
import xml.etree.ElementTree as etree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    datasets = []
    for version in ['v0100', 'v0101']:
        folder = args.root / version
        summary = json.loads(
            (folder / 'analysis/activity-summary.json').read_text())
        segments = list(
            csv.DictReader((folder / 'analysis/measured-segments.csv').open()))
        counts = collections.Counter()
        encoder_batches = []
        measured = False
        for line in (folder / 'run-001/activity-events.jsonl').open():
            tag, _, raw = line.partition('\t')
            if tag == 'PHASE_EPOCH':
                measured = True
                continue
            if not measured:
                continue
            record = json.loads(raw)
            if tag == 'PHASE_SCHEDULER_EVENT':
                counts[record['event_kind'] + ':' + record['action_kind']] += 1
            elif tag == 'PHASE_ENCODER_METRIC':
                encoder_batches.append(record['batch_size'])
        datasets.append(
            dict(version=version,
                 summary=summary,
                 segments=segments,
                 serving=json.loads(
                     (folder / 'run-001/client/aggregate.json').read_text()),
                 actions=dict(counts),
                 encoder_batches=encoder_batches))
    maximum = max(item['summary']['active_span_ms'] for item in datasets)
    svg = etree.Element('svg',
                        xmlns='http://www.w3.org/2000/svg',
                        viewBox='0 0 1000 520',
                        role='img')
    etree.SubElement(
        svg, 'title'
    ).text = 'Same-engine mixed HTTP: measured E/P/D intervals, one run per version'
    etree.SubElement(svg, 'rect', width='1000', height='520', fill='#ffffff')

    def label(x, y, text, size=14):
        element = etree.SubElement(svg,
                                   'text',
                                   x=str(x),
                                   y=str(y),
                                   fill='#18202b',
                                   attrib={
                                       'font-size': str(size),
                                       'font-family': 'sans-serif'
                                   })
        element.text = text

    label(15, 24,
          'Same-engine mixed HTTP | V1 Scalar | E=0001 P=0010 D=0100 C=1000',
          18)
    label(
        15, 47,
        'CUDA event-bounded work spans, not SM utilization. Copy: no recorded intervals.'
    )
    colors = {
        0: '#eeeeee',
        1: '#277da1',
        2: '#e9a124',
        3: '#58a68c',
        4: '#8254a3',
        5: '#407d80',
        6: '#c4587b',
        7: '#455366'
    }
    for index, item in enumerate(datasets):
        top = 86 + index * 200
        label(
            15, top,
            item['version'] + '  |  idle %.2f%%  |  E/P/D overlap %.2f%%' %
            (item['summary']['all_idle_ratio'] * 100,
             item['summary']['epd_overlap_ms'] /
             item['summary']['active_span_ms'] * 100), 16)
        epoch = float(item['segments'][0]['start_ms'])
        for row, (name, bit) in enumerate([('E', 1), ('P', 2), ('D', 4),
                                           ('C', 8), ('mask', 0)]):
            y = top + 13 + row * 25
            label(24, y + 16, name)
            for segment in item['segments']:
                mask = int(segment['mask'])
                if bit and not (mask & bit):
                    continue
                x = 85 + (float(segment['start_ms']) - epoch) / maximum * 880
                width = float(segment['duration_ms']) / maximum * 880
                etree.SubElement(svg,
                                 'rect',
                                 x=str(x),
                                 y=str(y),
                                 width=str(width),
                                 height='18',
                                 fill=colors.get(bit or mask, '#777777'))
            if name == 'C':
                label(85, y + 15, 'not observed (coverage unavailable)')
        for tick in range(0, int(maximum) + 1, 500):
            label(85 + tick / maximum * 880, top + 159, str(tick))
    label(380, 464, 'Elapsed ms from first recorded work in each run')
    label(
        15, 490,
        'Mask: 0000 idle | 0001 E | 0010 P | 0011 E+P | 0100 D | 0101 E+D | 0110 P+D'
    )
    label(
        15, 510,
        'Independent epochs aligned at first work; not identical scheduler snapshots or learned posteriors.'
    )
    etree.ElementTree(svg).write(args.root / 'version-timeline.svg',
                                 encoding='unicode')
    for item in datasets:
        del item['segments']
    (args.root /
     'comparison.json').write_text(json.dumps(datasets, indent=2) + '\n')
    print(
        json.dumps([{
            key: value
            for key, value in item.items()
            if key in ['version', 'actions', 'encoder_batches']
        } for item in datasets],
                   indent=2))


if __name__ == '__main__':
    main()
