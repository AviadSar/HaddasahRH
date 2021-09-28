def recommended_residence_pattern_0(text):
    return text + " should this person live in his own home? <mask>"


def recommended_residence_pattern_1(text):
    return text + " should this person live in a nursing home? <mask>"


def recommended_residence_pattern_2(text):
    return text + " this person should live in a <mask> home"


def residence_pattern_0(text):
    return text + " does this person live in his own home? <mask>"


def residence_pattern_1(text):
    return text + " does this person live in a nursing home? <mask>"


def residence_pattern_2(text):
    return text + " this person lives in a <mask> home"


def is_dementic_pattern_0(text):
    return text + " does this person suffer from dementia? <mask>"


def is_dementic_pattern_1(text):
    return text + " is this person mentally ill? <mask>"


def is_dementic_pattern_2(text):
    return text + " does this person have alzheimer's? <mask>"


def is_confused_pattern_0(text):
    return text + " is this person confused? <mask>"


def is_confused_pattern_1(text):
    return text + " is this person disoriented? <mask>"


def is_confused_pattern_2(text):
    return text + " is this person forgetful? <mask>"


def has_extreme_nursing_pattern_0(text):
    return text + " does this person have nursing? <mask>"


def has_extreme_nursing_pattern_1(text):
    return text + " this person <mask> nursing"


def needs_extreme_nursing_pattern_0(text):
    return text + " does this person need nursing? <mask>"


def needs_extreme_nursing_pattern_1(text):
    return text + " this person should <mask> nursing"


def is_exhausted_pattern_0(text):
    return text + " is this person exhausted? <mask>"


def is_exhausted_pattern_1(text):
    return text + " is this person fatigued? <mask>"


def is_exhausted_pattern_2(text):
    return text + " is this person debilitated? <mask>"


def seeking_help_at_home_pattern_0(text):
    return text + " this person <mask> more help hours than others"


def seeking_help_at_home_pattern_1(text):
    return text + " this person <mask> more help hours than others"


def seeking_help_at_home_pattern_2(text):
    return text + " this person <mask> more help hours than others"


def help_at_home_hours_pattern_0(text):
    return text + " this person gets <mask> hours of help at home"


def help_at_home_hours_pattern_1(text):
    return text + " <mask> hours each day, a nurse helps this person"


def help_at_home_hours_pattern_2(text):
    return text + " this person gets <mask> help at home"


def closest_supporting_relative_pattern_0(text):
    return text + " this person's relatives support him from <mask>"


def closest_supporting_relative_pattern_1(text):
    return text + " this person is aided by his relatives from <mask>"


def closest_relative_pattern_0(text):
    return text + " this person's relatives live <mask>"


def closest_relative_pattern_1(text):
    return text + " this person lives <mask>"


def children_pattern_0(text):
    return text + " does this person have children? <mask>"


def children_pattern_1(text):
    return text + " does this person have kids? <mask>"


def marital_status_pattern_0(text):
    return text + " is this person married? <mask>"


def marital_status_pattern_1(text):
    return text + " this person is <mask>"


def marital_status_pattern_2(text):
    return " this person is <mask>: " + text


def immigrant_pattern_0(text):
    return text + " is this person an immigrant? <mask>"


def immigrant_pattern_1(text):
    return text + " this person was born <mask> israel"


def immigrant_pattern_2(text):
    return text + " this person was born <mask>"


def immigrant_pattern_3(text):
    return text + " this person is from <mask>"


def sex_pattern_0(text):
    return text + " is this person a man? <mask>"


def sex_pattern_1(text):
    return text + " is this person a male? <mask>"


def sex_pattern_2(text):
    return text + " this person's sex is <mask>: "


def sex_pattern_3(text):
    return text + " this person's gender is <mask>: "


def apply_pattern(pattern, verbalizer, dataset, args):
    dataset['text'] = dataset['social_assessment'].apply(pattern)
    dataset['target'] = dataset.apply(verbalizer, axis=1)


def get_pattern_from_string(pattern_string, args):
    if pattern_string == 'recommended_residence_pattern_0':
        return recommended_residence_pattern_0
    elif pattern_string == 'recommended_residence_pattern_1':
        return recommended_residence_pattern_1
    elif pattern_string == 'recommended_residence_pattern_2':
        return recommended_residence_pattern_2
    elif pattern_string == 'residence_pattern_0':
        return residence_pattern_0
    elif pattern_string == 'residence_pattern_1':
        return residence_pattern_1
    elif pattern_string == 'residence_pattern_2':
        return residence_pattern_2
    elif pattern_string == 'is_dementic_pattern_0':
        return is_dementic_pattern_0
    elif pattern_string == 'is_dementic_pattern_1':
        return is_dementic_pattern_1
    elif pattern_string == 'is_dementic_pattern_2':
        return is_dementic_pattern_2
    elif pattern_string == 'is_confused_pattern_0':
        return is_confused_pattern_0
    elif pattern_string == 'is_confused_pattern_1':
        return is_confused_pattern_1
    elif pattern_string == 'is_confused_pattern_2':
        return is_confused_pattern_2
    elif pattern_string == 'has_extreme_nursing_pattern_0':
        return has_extreme_nursing_pattern_0
    elif pattern_string == 'has_extreme_nursing_pattern_1':
        return has_extreme_nursing_pattern_1
    elif pattern_string == 'needs_extreme_nursing_pattern_0':
        return needs_extreme_nursing_pattern_0
    elif pattern_string == 'needs_extreme_nursing_pattern_1':
        return needs_extreme_nursing_pattern_1
    elif pattern_string == 'is_exhausted_pattern_0':
        return is_exhausted_pattern_0
    elif pattern_string == 'is_exhausted_pattern_1':
        return is_exhausted_pattern_1
    elif pattern_string == 'is_exhausted_pattern_2':
        return is_exhausted_pattern_2
    elif pattern_string == 'seeking_help_at_home_pattern_0':
        return seeking_help_at_home_pattern_0
    elif pattern_string == 'seeking_help_at_home_pattern_1':
        return seeking_help_at_home_pattern_1
    elif pattern_string == 'seeking_help_at_home_pattern_2':
        return seeking_help_at_home_pattern_2
    elif pattern_string == 'help_at_home_hours_pattern_0':
        return help_at_home_hours_pattern_0
    elif pattern_string == 'help_at_home_hours_pattern_1':
        return help_at_home_hours_pattern_1
    elif pattern_string == 'help_at_home_hours_pattern_2':
        return help_at_home_hours_pattern_2
    elif pattern_string == 'closest_supporting_relative_pattern_0':
        return closest_supporting_relative_pattern_0
    elif pattern_string == 'closest_supporting_relative_pattern_1':
        return closest_supporting_relative_pattern_1
    elif pattern_string == 'closest_relative_pattern_0':
        return closest_relative_pattern_0
    elif pattern_string == 'closest_relative_pattern_1':
        return closest_relative_pattern_1
    elif pattern_string == 'children_pattern_0':
        return children_pattern_0
    elif pattern_string == 'children_pattern_1':
        return children_pattern_1
    elif pattern_string == 'marital_status_pattern_0':
        return marital_status_pattern_0
    elif pattern_string == 'marital_status_pattern_1':
        return marital_status_pattern_1
    elif pattern_string == 'marital_status_pattern_2':
        return marital_status_pattern_2
    elif pattern_string == 'immigrant_pattern_0':
        return immigrant_pattern_0
    elif pattern_string == 'immigrant_pattern_1':
        return immigrant_pattern_1
    elif pattern_string == 'immigrant_pattern_2':
        return immigrant_pattern_2
    elif pattern_string == 'immigrant_pattern_3':
        return immigrant_pattern_3
    elif pattern_string == 'sex_pattern_0':
        return sex_pattern_0
    elif pattern_string == 'sex_pattern_1':
        return sex_pattern_1
    elif pattern_string == 'sex_pattern_2':
        return sex_pattern_2
    elif pattern_string == 'sex_pattern_3':
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