import sklearn
from ourStatistics import interpolate_by_mean as ibm
import pandas as pd

cortex_data = pd.read_excel('Data_Cortex_Nuclear.xls')

ibm(cortex_data)
