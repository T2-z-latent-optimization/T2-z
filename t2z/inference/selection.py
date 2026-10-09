def parse_indices(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(',') if item.strip()]
    return [int(item) for item in value]


def group_samples(rows, support_indices, query_indices, environment_key):
    supports = {}
    for index in support_indices:
        key = str(rows[index][environment_key])
        supports.setdefault(key, []).append(index)
    groups = {}
    for index in query_indices:
        key = str(rows[index][environment_key])
        group = groups.setdefault(key, {'environment_id': key, 'support_indices': supports.get(key, []), 'query_indices': []})
        group['query_indices'].append(index)
    return list(groups.values())
