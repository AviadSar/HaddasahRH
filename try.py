import pandas as pd

with open("data/social_assements_30_annotations.tsv", 'r') as data_file:
    df = pd.read_csv(data_file, sep='\t')
    events = df['event_identifier']
    uniq_events = set(events)
    print(len(events) - len(uniq_events))
