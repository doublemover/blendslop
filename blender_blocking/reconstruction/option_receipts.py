"""Record actual configuration reads separately from dispatched settings."""
from dataclasses import replace


class ConsumedOptions(dict):
    def __init__(self, values=(), *, reads=None):
        super().__init__(values)
        self.reads = set() if reads is None else reads

    def __getitem__(self, key):
        self.reads.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.add(key)
        return super().get(key, default)

    def copy(self):
        return type(self)(self, reads=self.reads)


def copy_options(values):
    return values.copy() if isinstance(values, ConsumedOptions) else dict(values)


def reconstruct_with_receipt(backend, request):
    options = ConsumedOptions(request.config)
    request = replace(request, config=options)
    result = backend.reconstruct(request)
    receipt = {'consumer': request.backend_name, 'consumed_keys': sorted(options.reads & options.keys()),
               'unconsumed_keys': sorted(options.keys()-options.reads),
               'meaning': 'keys read by this backend invocation; reading a switch does not prove admission'}
    extras = {**result.metric_result.extras, 'option_consumption': receipt}
    return replace(result, metric_result=replace(result.metric_result, extras=extras))
