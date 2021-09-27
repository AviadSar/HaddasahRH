class sex_verbalizer_0(object):
    def __init__(self, args):
        self.target_column = args.target_column
        # args.labels[0] == "m", args.labels[1] == "f", args.labels[2] == "unknown"
        self.classes = {args.labels[0]: ' yes', args.labels[1]: ' no', args.labels[1]: ' maybe'}

    def __call__(self, row, *args, **kwargs):
        target = row['text']
        if row[self.target_column] == 'm':
            return target.replace('<mask>', 'yes')
        elif row[self.target_column] == 'f':
            return target.replace('<mask>', 'no')
        else:
            return target.replace('<mask>', 'maybe')


class sex_verbalizer_1(sex_verbalizer_0):
    def __init__(self, args):
        super().__init__(args)


class sex_verbalizer_2(object):
    def __init__(self, args):
        self.target_column = args.target_column
        # args.labels[0] == "m", args.labels[1] == "f", args.labels[2] == "unknown"
        self.classes = {args.labels[0]: ' male', args.labels[1]: ' female', args.labels[1]: ' unknown'}

    def __call__(self, row, *args, **kwargs):
        target = row['text']
        if row[self.target_column] == 'm':
            return target.replace('<mask>', 'male')
        elif row[self.target_column] == 'f':
            return target.replace('<mask>', 'female')
        else:
            return target.replace('<mask>', 'unknown')


class sex_verbalizer_3(object):
    def __init__(self, args):
        self.target_column = args.target_column
        # args.labels[0] == "m", args.labels[1] == "f", args.labels[2] == "unknown"
        self.classes = {args.labels[0]: ' man', args.labels[1]: ' woman', args.labels[1]: ' unknown'}

    def __call__(self, row, *args, **kwargs):
        target = row['text']
        if row[self.target_column] == 'm':
            return target.replace('<mask>', 'man')
        elif row[self.target_column] == 'f':
            return target.replace('<mask>', 'woman')
        else:
            return target.replace('<mask>', 'unknown')


class marital_status_verbalizer_0(object):
    def __init__(self, args):
        self.target_column = args.target_column
        # args.labels[0] == "married", args.labels[1] == "not_married"
        self.classes = {args.labels[0]: ' yes', args.labels[1]: ' no'}

    def __call__(self, row, *args, **kwargs):
        target = row['text']
        if row[self.target_column] == 'married':
            return target.replace('<mask>', 'yes')
        else:
            return target.replace('<mask>', 'no')


class marital_status_verbalizer_1(object):
    def __init__(self, args):
        self.target_column = args.target_column
        # args.labels[0] == "married", args.labels[1] == "not_married"
        self.classes = {args.labels[0]: ' married', args.labels[1]: ' single'}

    def __call__(self, row, *args, **kwargs):
        target = row['text']
        if row[self.target_column] == 'married':
            return target.replace('<mask>', 'married')
        else:
            return target.replace('<mask>', 'single')


class marital_status_verbalizer_2(marital_status_verbalizer_1):
    def __init__(self, args):
        super().__init__(args)


def get_verbalizer_from_string(verbalizer_string, args):
    if verbalizer_string == 'marital_status_verbalizer_0':
        return marital_status_verbalizer_0(args)
    elif verbalizer_string == 'marital_status_verbalizer_1':
        return marital_status_verbalizer_1(args)
    elif verbalizer_string == 'marital_status_verbalizer_2':
        return marital_status_verbalizer_2(args)
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
