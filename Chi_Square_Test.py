from scipy import stats
import pandas as pd
from dataclasses import dataclass
import pandas as pd

@dataclass
class Chi_Square_Test:
    x: str
    y: str
    data: pd.DataFrame()
    
    def __post_init__(self):
        self.data = data[[x, y]].copy()  # Avoid SettingWithCopyWarning
        self.data["count"] = 1
        self.feature = x
        self.label = y
        self.p_data = self.expected_contingency_table()
        self.expected_value_matrix = self.calculate_expected_value_matrix()
        self.calculate_degrees_of_freedom()

    def calculate_degrees_of_freedom(self) -> int:
        df = (len(self.p_data) - 1) * (len(self.p_data.columns) - 1)
        return df

    def expected_contingency_table(self) -> pd.DataFrame:
        grouped_data = self.data.groupby([self.feature, self.label]).agg({'count': 'sum'}).reset_index()
        pivoted_data = grouped_data.pivot_table(columns=[self.feature], index=[self.label], values="count", fill_value=0)
        return pivoted_data

    def calculate_expected_value_matrix(self) -> pd.DataFrame:
        total = self.p_data.sum().sum()
        expected_values = (self.p_data.sum(axis=0).values * self.p_data.sum(axis=1).values[:, None]) / total
        return pd.DataFrame(expected_values, columns=self.p_data.columns, index=self.p_data.index)

    def calculate_chi_square_matrix(self) -> float:
        chi_square = (self.p_data - self.expected_value_matrix)**2 / (self.expected_value_matrix + 1e-10)  # Avoid division by zero
        return chi_square.sum().sum()

    def calculate_p_value(self, degrees_of_freedom: int, chi_square_stat: float) -> float:
        p_value = stats.chi2.sf(chi_square_stat, df=degrees_of_freedom)
        return p_value

    def calculate(self):
        chi_square_stat = self.calculate_chi_square_matrix()
        df = self.calculate_degrees_of_freedom()
        p_value = self.calculate_p_value(df, chi_square_stat)
        return chi_square_stat, p_value
