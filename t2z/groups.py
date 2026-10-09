def adapt_environment_indices(rows):
    for row in rows:
        if 'friction_mu' in row:
            continue
        value = row.get('environment_index')
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Dataset requires a non-negative integer environment_index')
        row['friction_mu'] = float(value)
    return rows
