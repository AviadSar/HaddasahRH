class verbalizer(object):
    def __init__(self, args, label_tokens):
        self.args = args
        self.target_column = args.target_column
        # args.labels[0] == "m", args.labels[1] == "f", args.labels[2] == "unknown"
        self.label_tokens = label_tokens
        self.classes = {}
        for idx, label in enumerate(args.labels):
            self.classes[label] = ' ' + label_tokens[idx]

    def __call__(self, row, *args, **kwargs):
        target = row['text']
        for idx, label in enumerate(self.args.labels):
            if row[self.target_column] == label:
                return target.replace('<mask>', self.label_tokens[idx])
        raise ValueError('no such label "{}"'.format(row[self.target_column]))


class recommended_residence_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class recommended_residence_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['no', 'yes'])


class recommended_residence_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['private', 'rest'])


class residence_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class residence_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['no', 'yes'])


class residence_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['private', 'rest'])


class is_dementic_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_dementic_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_dementic_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_confused_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_confused_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_confused_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class has_extreme_nursing_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class has_extreme_nursing_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['has', 'need'])


class needs_extreme_nursing_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class needs_extreme_nursing_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['seek', 'avoid'])


class is_exhausted_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_exhausted_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class is_exhausted_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class seeking_help_at_home_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['needs', 'has'])


class seeking_help_at_home_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['want', 'has'])


class seeking_help_at_home_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['asks', 'has'])


class help_at_home_hours_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['no', 'few', 'many', 'all'])


class help_at_home_hours_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['zero', 'few', 'many', 'all'])


class help_at_home_hours_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['no', 'little', 'great', 'special'])


class closest_supporting_relative_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['home', 'close', 'far'])


class closest_supporting_relative_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['home', 'close', 'far'])


class closest_relative_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['with', 'close', 'far'])


class closest_relative_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['with', 'close', 'far'])


class children_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class children_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class marital_status_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class marital_status_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['married', 'single'])


class marital_status_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class immigrant_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no'])


class immigrant_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['outside', 'inside'])


class immigrant_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['far', 'here'])


class immigrant_verbalizer_3(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['far', 'here'])


class sex_verbalizer_0(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no', 'maybe'])


class sex_verbalizer_1(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['yes', 'no', 'maybe'])


class sex_verbalizer_2(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['male', 'female', 'unknown'])


class sex_verbalizer_3(verbalizer):
    def __init__(self, args):
        super().__init__(args, ['man', 'woman', 'unknown'])


def get_verbalizer_from_string(verbalizer_string, args):
    if verbalizer_string == 'recommended_residence_verbalizer_0':
        return recommended_residence_verbalizer_0(args)
    elif verbalizer_string == 'recommended_residence_verbalizer_1':
        return recommended_residence_verbalizer_1(args)
    elif verbalizer_string == 'recommended_residence_verbalizer_2':
        return recommended_residence_verbalizer_2(args)
    elif verbalizer_string == 'residence_verbalizer_0':
        return residence_verbalizer_0(args)
    elif verbalizer_string == 'residence_verbalizer_1':
        return residence_verbalizer_1(args)
    elif verbalizer_string == 'residence_verbalizer_2':
        return residence_verbalizer_2(args)
    elif verbalizer_string == 'is_dementic_verbalizer_0':
        return is_dementic_verbalizer_0(args)
    elif verbalizer_string == 'is_dementic_verbalizer_1':
        return is_dementic_verbalizer_1(args)
    elif verbalizer_string == 'is_dementic_verbalizer_2':
        return is_dementic_verbalizer_2(args)
    elif verbalizer_string == 'is_confused_verbalizer_0':
        return is_confused_verbalizer_0(args)
    elif verbalizer_string == 'is_confused_verbalizer_1':
        return is_confused_verbalizer_1(args)
    elif verbalizer_string == 'is_confused_verbalizer_2':
        return is_confused_verbalizer_2(args)
    elif verbalizer_string == 'has_extreme_nursing_verbalizer_0':
        return has_extreme_nursing_verbalizer_0(args)
    elif verbalizer_string == 'has_extreme_nursing_verbalizer_1':
        return has_extreme_nursing_verbalizer_1(args)
    elif verbalizer_string == 'needs_extreme_nursing_verbalizer_0':
        return needs_extreme_nursing_verbalizer_0(args)
    elif verbalizer_string == 'needs_extreme_nursing_verbalizer_1':
        return needs_extreme_nursing_verbalizer_1(args)
    elif verbalizer_string == 'is_exhausted_verbalizer_0':
        return is_exhausted_verbalizer_0(args)
    elif verbalizer_string == 'is_exhausted_verbalizer_1':
        return is_exhausted_verbalizer_1(args)
    elif verbalizer_string == 'is_exhausted_verbalizer_2':
        return is_exhausted_verbalizer_2(args)
    elif verbalizer_string == 'seeking_help_at_home_verbalizer_0':
        return seeking_help_at_home_verbalizer_0(args)
    elif verbalizer_string == 'seeking_help_at_home_verbalizer_1':
        return seeking_help_at_home_verbalizer_1(args)
    elif verbalizer_string == 'seeking_help_at_home_verbalizer_2':
        return seeking_help_at_home_verbalizer_2(args)
    elif verbalizer_string == 'help_at_home_hours_verbalizer_0':
        return help_at_home_hours_verbalizer_0(args)
    elif verbalizer_string == 'help_at_home_hours_verbalizer_1':
        return help_at_home_hours_verbalizer_1(args)
    elif verbalizer_string == 'help_at_home_hours_verbalizer_2':
        return help_at_home_hours_verbalizer_2(args)
    elif verbalizer_string == 'closest_supporting_relative_verbalizer_0':
        return closest_supporting_relative_verbalizer_0(args)
    elif verbalizer_string == 'closest_supporting_relative_verbalizer_1':
        return closest_supporting_relative_verbalizer_1(args)
    elif verbalizer_string == 'closest_relative_verbalizer_0':
        return closest_relative_verbalizer_0(args)
    elif verbalizer_string == 'closest_relative_verbalizer_1':
        return closest_relative_verbalizer_1(args)
    elif verbalizer_string == 'children_verbalizer_0':
        return children_verbalizer_0(args)
    elif verbalizer_string == 'children_verbalizer_1':
        return children_verbalizer_1(args)
    elif verbalizer_string == 'marital_status_verbalizer_0':
        return marital_status_verbalizer_0(args)
    elif verbalizer_string == 'marital_status_verbalizer_1':
        return marital_status_verbalizer_1(args)
    elif verbalizer_string == 'marital_status_verbalizer_2':
        return marital_status_verbalizer_2(args)
    elif verbalizer_string == 'immigrant_verbalizer_0':
        return immigrant_verbalizer_0(args)
    elif verbalizer_string == 'immigrant_verbalizer_1':
        return immigrant_verbalizer_1(args)
    elif verbalizer_string == 'immigrant_verbalizer_2':
        return immigrant_verbalizer_2(args)
    elif verbalizer_string == 'immigrant_verbalizer_3':
        return immigrant_verbalizer_3(args)
    elif verbalizer_string == 'sex_verbalizer_0':
        return sex_verbalizer_0(args)
    elif verbalizer_string == 'sex_verbalizer_1':
        return sex_verbalizer_1(args)
    elif verbalizer_string == 'sex_verbalizer_2':
        return sex_verbalizer_2(args)
    elif verbalizer_string == 'sex_verbalizer_3':
        return sex_verbalizer_3(args)
    return None


def get_verbalizers_from_args(args):
    if args.verbalizers is None:
        raise ValueError('no verbalizers found in args file.')
    verbalizers = []
    for verbalizer_string in args.verbalizers:
        verbalizer = get_verbalizer_from_string(verbalizer_string, args)
        if verbalizer is None:
            raise ValueError('no such verbalizer "{}" in verbalizers.py'.format(verbalizer_string))
        else:
            verbalizers.append(verbalizer)
    return verbalizers
