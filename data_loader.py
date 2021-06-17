import pandas as pd
import numpy as np


def fill_empty_values(social_assessments):
    social_assessments['sex'].fillna(value='unknown', inplace=True)
    social_assessments['age'].fillna(value='unknown', inplace=True)
    social_assessments['immigrant'].fillna(value='no', inplace=True)
    social_assessments['year_of_immigration'].fillna(value='unknown', inplace=True)
    social_assessments['marital_status'].fillna(value='unknown', inplace=True)
    social_assessments['children'].fillna(value='unknown', inplace=True)
    social_assessments['closest_relative'].fillna(value='unknown', inplace=True)
    social_assessments['closest_supporting_relative'].fillna(value='unknown', inplace=True)
    social_assessments['help_at_home_hours'].fillna(value=0, inplace=True)
    social_assessments['seeking_help_at_home'].fillna(value='no', inplace=True)
    social_assessments['is_holocaust_survivor'].fillna(value='no', inplace=True)
    social_assessments['is_fatigued_(tashush)'].fillna(value='no', inplace=True)
    social_assessments['needs_extreme_nursing_(siudi)'].fillna(value='no', inplace=True)
    social_assessments['has_extreme_nursing'].fillna(value='no', inplace=True)
    social_assessments['is_confused'].fillna(value='no', inplace=True)
    social_assessments['is_dementic'].fillna(value='no', inplace=True)
    social_assessments['residence'].fillna(value='unknown', inplace=True)
    social_assessments['recommeded_residence'].fillna(value='unknown', inplace=True)
    social_assessments['is_owner'].fillna(value='unknown', inplace=True)


def autofill_data():
    social_assessments = pd.read_csv("/home/aviad/Documents/HaddasahRH/social_assessments_only_first_100_en.tsv", sep="\t")
    fill_empty_values(social_assessments)
    social_assessments.to_csv("/home/aviad/Documents/HaddasahRH/social_assessments_only_first_100_en_filled.tsv", sep="\t")


def load_data(data_file):
    social_assessments = pd.read_csv(data_file, sep="\t")
    return social_assessments


social_assessments = load_data("/home/aviad/Documents/HaddasahRH/social_assessments_only_first_100_en_filled.tsv")
