class marital_status_verbalizer_1(object):
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


class marital_status_verbalizer_2(object):
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


def get_verbalizer_from_string(verbalizer_string, args):
    if verbalizer_string == 'marital_status_verbalizer_1':
        return marital_status_verbalizer_1(args)
    elif verbalizer_string == 'marital_status_verbalizer_2':
        return marital_status_verbalizer_2(args)
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
