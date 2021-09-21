def marital_status_pattern_1(text):
    return text + " is this person married? <mask>"


def apply_pattern(pattern, verbalizer, dataset, args):
    dataset['text'] = dataset['social_assessment'].apply(pattern)
    dataset['target'] = dataset.apply(verbalizer, axis=1)


def get_pattern_from_string(pattern_string, args):
    if pattern_string == 'marital_status_pattern_1':
        return marital_status_pattern_1
    return None


def get_patterns_from_args(args):
    if args.patterns is None:
        return None
    patterns = []
    for pattern_string in args.patterns:
        pattern = get_pattern_from_string(pattern_string, args)
        if pattern is None:
            return None
        else:
            patterns.append(pattern)
    return patterns