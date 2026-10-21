"""Bounded streaming readers for the local large nuScenes JSON files."""
import json
from pathlib import Path


class JSONStream:
    def __init__(self, path):
        self.handle = open(path)
        self.buffer = ''
        self.pos = 0
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.pos:] + self.handle.read(1024 * 1024)
        self.pos = 0

    def char(self):
        while True:
            if self.pos >= len(self.buffer):
                self.fill()
                if not self.buffer:
                    raise ValueError('Unexpected end of JSON')
            if self.buffer[self.pos].isspace():
                self.pos += 1
            else:
                return self.buffer[self.pos]

    def expect(self, char):
        if self.char() != char:
            raise ValueError('Expected ' + char)
        self.pos += 1

    def value(self):
        self.char()
        while True:
            try:
                value, self.pos = self.decoder.raw_decode(self.buffer, self.pos)
                return value
            except json.JSONDecodeError:
                old = len(self.buffer) - self.pos
                self.fill()
                if len(self.buffer) == old:
                    raise ValueError('Invalid or truncated JSON')


def iter_json_array(path):
    stream = JSONStream(path)
    try:
        stream.expect('[')
        while stream.char() != ']':
            yield stream.value()
            if stream.char() == ',': stream.expect(',')
            elif stream.char() != ']': raise ValueError('Malformed JSON array')
    finally:
        stream.handle.close()


def read_prediction_subset(path, sample_ids):
    stream = JSONStream(path)
    found = {}
    try:
        stream.expect('{')
        while stream.char() != '}':
            key = stream.value(); stream.expect(':')
            if key == 'results':
                stream.expect('{')
                while stream.char() != '}':
                    sample = stream.value(); stream.expect(':')
                    values = stream.value()
                    if sample in sample_ids: found[sample] = values
                    if stream.char() == ',': stream.expect(',')
                    elif stream.char() != '}': raise ValueError('Malformed predictions')
                stream.expect('}')
            else:
                stream.value()
            if stream.char() == ',': stream.expect(',')
            elif stream.char() != '}': raise ValueError('Malformed predictions')
    finally:
        stream.handle.close()
    return found


def write_json(path, value):
    from worldstream.m1.state_types import json_value
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')
