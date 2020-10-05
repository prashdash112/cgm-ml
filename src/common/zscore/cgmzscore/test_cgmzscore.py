import pytest
from cgmzscore import Calculator
import pandas as pd

df = pd.read_csv('test_data/test_data.csv')

# 1 for male
# 2 for female


def test_zScore_wfa():
    for i in range(len(df)):
        if(pd.isnull(df.loc[i, 'WEIGHT']) or pd.isnull(df.loc[i, '_agedays']) or pd.isnull(df.loc[i, '_ZWEI'])):
            assert True
            continue
        if(df.loc[i, '_agedays'] == 0):
            assert True
            continue

        if df['GENDER'][i] == 1:
            sex = 'M'
        else:
            sex = 'F'
        g = float("{0:.9f}". format(df['WEIGHT'][i]))
        v = Calculator().zScore_wfa(weight=str(g), height=str(
            df['HEIGHT'][i]), sex=sex, age_in_days=str((df['_agedays'][i])))
        ans = float("{0:.2f}". format(abs(v-df['_ZWEI'][i])))
        assert ans <= 0.01


def test_zScore_lhfa():
    for i in range(len(df)):
        if(pd.isnull(df.loc[i, 'HEIGHT']) or pd.isnull(df.loc[i, '_agedays']) or pd.isnull(df.loc[i, '_ZLEN'] or df.loc[i, '_agedays'] == 0)):
            assert True
            continue

        if df['GENDER'][i] == 1:
            sex = 'M'
        else:
            sex = 'F'

        g = float("{0:.9f}". format(df['HEIGHT'][i]))

        v = Calculator().zScore_lhfa(height=str(g), sex=sex,
                                     age_in_days=str((df['_agedays'][i])))

        print(g)

        ans = float("{0:.2f}". format(abs(v-df['_ZLEN'][i])))
        assert ans <= 0.01


def test_zScore_wfh():
    for i in range(len(df)):
        if(pd.isnull(df.loc[i, 'HEIGHT']) or pd.isnull(df.loc[i, 'WEIGHT']) or pd.isnull(df.loc[i, '_agedays']) or pd.isnull(df.loc[i, '_ZWFL'] or df.loc[i, '_agedays'] == 0)):
            assert True
            continue

        if df['GENDER'][i] == 1:
            sex = 'M'
        else:
            sex = 'F'

        g = float("{0:.5f}". format(df['HEIGHT'][i]))
        t = float("{0:.9f}". format(df['WEIGHT'][i]))

        v = Calculator().zScore_wfh(height=str(g), weight=str(
            t), sex=sex, age_in_days=str((df['_agedays'][i])))

        ans = float("{0:.2f}". format(abs(v-df['_ZWFL'][i])))
        assert ans <= 0.01

def test_zScore_wfl():
    for i in range(len(df)):
        if(pd.isnull(df.loc[i, 'HEIGHT']) or pd.isnull(df.loc[i, 'WEIGHT']) or pd.isnull(df.loc[i, '_agedays']) or pd.isnull(df.loc[i, '_ZWFL'] or df.loc[i, '_agedays'] == 0)):
            assert True
            continue

        if df['GENDER'][i] == 1:
            sex = 'M'
        else:
            sex = 'F'

        g = float("{0:.5f}". format(df['HEIGHT'][i]))
        t = float("{0:.9f}". format(df['WEIGHT'][i]))

        v = Calculator().zScore_wfl(height=str(g), weight=str(
            t), sex=sex, age_in_days=str((df['_agedays'][i])))

        ans = float("{0:.2f}". format(abs(v-df['_ZWFL'][i])))
        assert ans <= 0.01