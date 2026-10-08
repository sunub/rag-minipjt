"""Structured calls with a persistent, prompt/schema/model-specific cache."""
from .cache import JsonCache


class StructuredLLM:
    def __init__(self, client, cache_dir, identity):
        self.client, self.identity = client, identity
        self.cache = JsonCache(cache_dir)

    async def invoke(self, schema, system, payload):
        import json
        key = {'model': self.identity, 'schema': schema.model_json_schema(),
               'system': system, 'payload': payload}
        cached = self.cache.get(key)
        if cached is not None:
            return schema.model_validate(cached)
        chain = self.client.with_structured_output(schema, method='function_calling')
        result = await chain.ainvoke([('system', system),
                                     ('human', json.dumps(payload, ensure_ascii=False))])
        if not isinstance(result, schema):
            result = schema.model_validate(result)
        self.cache.put(key, result.model_dump())
        return result
