# Z score

z-scores of anthropometric measurements of children below 5 years  based on WHO

## Installation

pip install zscore

## Usage

from cgmzscore import Calculator
calculator=Calculator()


v=Calculator().zScore_wfa(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to calculate z score for weight vs age
v=Calculator().zScore_wfl(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to calculate z score for weight vs length/height
v=Calculator().zScore_wfh(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to calculate z score for weight vs length/height
both wfl and wfh works same
v=Calculator().zScore_lhfa(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to calculate z score for length vs age

v=Calculator().zScore(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to calculate all three z score
v=Calculator().zScore_withclass(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to calculate all three z score along with class
v=Calculator().SAM_MAM(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')-->Use to find child is SAM/MAM/Healthy



