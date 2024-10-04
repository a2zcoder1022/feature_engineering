import numpy as np
from scipy import stats
class Ttest:
    def __init__(self, data, x, y):
        # seperating the groups by class
        self.group_1 = data[data[y] == 0] 
        self.group_2 = data[data[y] == 1]

        #groups mean
        self.group_1_mean = self.group_1[x].mean()
        self.group_2_mean = self.group_2[x].mean()

        #groups variance
        self.group_1_variance = self.group_1[x].var()
        self.group_2_variance = self.group_2[x].var()

        #groups sample size
        self.group_1_sample_size = len(self.group_1)
        self.group_2_sample_size = len(self.group_2)
        self.df = (self.group_1_sample_size + self.group_2_sample_size) - 2
    
    def calculate(self):
        mean_diff = self.group_1_mean - self.group_2_mean #difference in mean
        var_1_samp = self.group_1_variance / self.group_1_sample_size #variance group_1
        var_2_samp = self.group_2_variance / self.group_2_sample_size #variance group_2
        total_var = var_1_samp + var_2_samp #total_variance
        sqrt_var = np.sqrt(total_var)
        return mean_diff / sqrt_var #t_score

    def get_p_value(self):
        t_stat = self.calculate()
        p_value = stats.t.sf(abs(t_stat), self.df) * 2 # two tailed test
        return p_value
        