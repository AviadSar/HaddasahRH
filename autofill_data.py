import pandas as pd
import numpy as np


def fill_empty_values(social_assessments):
    fill_to_row = 90
    social_assessments['sex'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['age'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['immigrant'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['year_of_immigration'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['marital_status'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['children'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['closest_relative'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['closest_supporting_relative'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['help_at_home_hours'][:fill_to_row].fillna(value=0, inplace=True)
    social_assessments['seeking_help_at_home'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['is_holocaust_survivor'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['is_fatigued_(tashush)'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['needs_extreme_nursing_(siudi)'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['has_extreme_nursing'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['is_confused'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['is_dementic'][:fill_to_row].fillna(value='no', inplace=True)
    social_assessments['residence'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['recommeded_residence'][:fill_to_row].fillna(value='unknown', inplace=True)
    social_assessments['is_owner'][:fill_to_row].fillna(value='unknown', inplace=True)


def autofill_data():
    social_assessments = pd.read_csv("data/social_assesments_100_annotations_clean_en.tsv", sep="\t")
    fill_empty_values(social_assessments)
    social_assessments.to_csv("data/social_assesments_100_annotations_clean_filled_en.tsv", sep="\t")


def load_data(data_file):
    social_assessments = pd.read_csv(data_file, sep="\t")
    return social_assessments


autofill_data()