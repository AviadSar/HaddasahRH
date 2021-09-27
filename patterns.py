def sex_pattern_0(text):
    return text + " is this person a man? <mask>"


def sex_pattern_1(text):
    return text + " is this person a male? <mask>"


def sex_pattern_2(text):
    return text + " this person's sex is <mask>: "


def sex_pattern_3(text):
    return text + " this person's gender is <mask>: "


def marital_status_pattern_0(text):
    return text + " is this person married? <mask>"


def marital_status_pattern_1(text):
    return text + " this person is <mask>"


def marital_status_pattern_2(text):
    return " this person is <mask>: " + text


def apply_pattern(pattern, verbalizer, dataset, args):
    dataset['text'] = dataset['social_assessment'].apply(pattern)
    dataset['target'] = dataset.apply(verbalizer, axis=1)


def get_pattern_from_string(pattern_string, args):
    if pattern_string == 'marital_status_pattern_0':
        return marital_status_pattern_0
    if pattern_string == 'marital_status_pattern_1':
        return marital_status_pattern_1
    if pattern_string == 'marital_status_pattern_2':
        return marital_status_pattern_2
    if pattern_string == 'sex_pattern_0':
        return sex_pattern_0
    if pattern_string == 'sex_pattern_1':
        return sex_pattern_1
    if pattern_string == 'sex_pattern_2':
        return sex_pattern_2
    if pattern_string == 'sex_pattern_3':
        return sex_pattern_3
    return None


def get_patterns_from_args(args):
    if args.patterns is None:
        raise ValueError('no patterns found in args file.')
    patterns = []
    for pattern_string in args.patterns:
        pattern = get_pattern_from_string(pattern_string, args)
        if pattern is None:
            raise ValueError('no such pattern "{}" in patterns.py'.format(pattern_string))
        else:
            patterns.append(pattern)
    return patterns