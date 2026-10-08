"""Content-addressed caches: no credentials, no reuse across models/prompts/corpora."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class JsonCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def get(self, key):
        path = self.directory / f'{digest(key)}.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def put(self, key, value):
        write_json(self.directory / f'{digest(key)}.json', value)


class CachedEmbeddings:
    def __init__(self, client, directory, identity):
        self.client, self.identity = client, identity
        self.cache = JsonCache(directory)

    async def aembed_documents(self, texts):
        keys = [{'model': self.identity, 'text': text, 'kind': 'document'} for text in texts]
        vectors = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(vectors) if v is None]
        for start in range(0, len(missing), 64):
            batch = missing[start:start + 64]
            values = await self.client.aembed_documents([texts[i] for i in batch])
            if len(values) != len(batch):
                raise ValueError('Embedding response count does not match request')
            for i, value in zip(batch, values):
                vectors[i] = value
                self.cache.put(keys[i], value)
        return vectors

    async def aembed_query(self, text):
        key = {'model': self.identity, 'text': text, 'kind': 'query'}
        value = self.cache.get(key)
        if value is None:
            value = await self.client.aembed_query(text)
            self.cache.put(key, value)
        return value
