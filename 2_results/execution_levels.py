"""Execution-source vocabulary used for the approach and reference bases."""

BASIS_ORDER = ['Real vehicle', 'Real hardware', 'Real dataset', 'Simulator', 'Simulated dataset', 'None']
BASIS_COLORS = dict(zip(BASIS_ORDER, ['#009E73', '#56B4E9', '#E6B800', '#0072B2', '#CC79A7', '#dcdcdc']))


def basis_tags(value):
    tags = str(value).split(' + ')
    if any(tag not in BASIS_ORDER for tag in tags) or len(tags) != len(set(tags)):
        raise ValueError(f'Invalid execution basis: {value!r}')
    if 'None' in tags and len(tags) != 1:
        raise ValueError('None cannot be combined with an execution source')
    return tags
