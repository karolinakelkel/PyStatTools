from math import sqrt
import scipy.stats as stats


class DataSet(list):
    def __init__(self, data: list):
        """
        Construct a DataSet object from a list of data.
        """

        super().__init__(data)
        self.size = len(data)

    def __pow__(self, power: int):
        """
        Raise each element of the dataset to the specified power.
        """

        return DataSet([value ** power for value in self])

    def __mul__(self, other: 'DataSet'):
        """
         Compute the element-wise product of two datasets.
        """

        return DataSet([first_set_value * second_set_value for first_set_value, second_set_value in zip(self, other)])


def z_value(confidence_level=0.95):
    return round(abs(stats.norm.ppf((1 - confidence_level) / 2)), 2)


class DataSetWithStatistics(DataSet):
    def __init__(self, data_set: list):
        super().__init__(data=data_set)

        self.__sum_of_observations = None
        self.__is_num_of_items_even = None

    def sum_of_observations(self) -> [int, float]:
        if self.__sum_of_observations is None:
            self.__sum_of_observations = sum(self)

        return self.__sum_of_observations

    def is_num_of_items_even(self) -> bool:
        if self.__is_num_of_items_even is None:
            self.__is_num_of_items_even = self.size % 2 == 0

        return self.__is_num_of_items_even

    def arithmetic_mean(self, bessel_correction=False) -> float:
        """
        Compute the arithmetic mean of a sorted dataset. If the 'bessel_correction' parameter is set to True,
        the method applies Bessel's correction to the sample mean formula.
        """

        sum_of_observations = self.sum_of_observations

        if bessel_correction:
            return sum_of_observations() / (self.size - 1)
        else:
            return sum_of_observations() / self.size

    def median(self) -> [int, float]:
        """
        Compute the arithmetic mean of a sorted dataset.
        """

        if self.is_num_of_items_even() is False:
            return self[self.size // 2]

        idx_of_second_mid_value = self.size // 2
        idx_of_first_mid_value = idx_of_second_mid_value - 1

        return (self[idx_of_first_mid_value] + self[idx_of_second_mid_value]) / 2

    def mode(self) -> [int, float]:
        freq_dictionary = {}
        for value in self:
            freq_dictionary[value] = freq_dictionary.get(value, 0) + 1

        modes_list = []
        max_freq = 0
        for value, value_freq in freq_dictionary.items():
            if value_freq > max_freq:
                modes_list = [value]
                max_freq = value_freq
            elif value_freq == max_freq:
                modes_list.append(value)

        num_of_modes = len(modes_list)
        if max_freq > 1 and num_of_modes > 1:
            return modes_list
        elif max_freq > 1 and num_of_modes == 1:
            return modes_list[0]
        else:
            raise ValueError('No mode found')

    def range(self) -> [int, float]:
        return max(self) - min(self)

    def diff_between_value_and_mean(self, squared=True) -> 'DataSetWithStatistics':
        data_set_mean = self.arithmetic_mean()
        diff = DataSetWithStatistics([value - data_set_mean for value in self])

        return DataSetWithStatistics(diff ** 2 if squared else diff)

    def variance(self, bessel_correction=True):
        squared_differences = self.diff_between_value_and_mean()

        return squared_differences.arithmetic_mean(bessel_correction)

    def standard_deviation(self, bessel_correction=True):
        return sqrt(self.variance(bessel_correction))

    def covariance(self, other, bessel_correction=True, total_covariance_only=False):
        set_1_diff = self.diff_between_value_and_mean(squared=False)
        set_2_diff = other.diff_between_value_and_mean(squared=False)
        products_of_sets = set_1_diff * set_2_diff

        return sum(products_of_sets) if total_covariance_only \
            else DataSetWithStatistics(products_of_sets).arithmetic_mean(bessel_correction)

    def pearson_correlation_coefficient(self, other):
        total_covariance = self.covariance(other, total_covariance_only=True)
        set_1_sum_of_squared_diffs = sum(self.diff_between_value_and_mean(squared=True))
        set_2_sum_of_squared_diffs = sum(other.diff_between_value_and_mean(squared=True))
        product_of_roots = sqrt(set_1_sum_of_squared_diffs) * sqrt(set_2_sum_of_squared_diffs)

        return total_covariance / product_of_roots

    def quantiles(self) -> list:
        sorted_data_set = DataSetWithStatistics(sorted(self))

        idx_of_mid_value = sorted_data_set.size // 2
        first_quartile = DataSetWithStatistics(sorted_data_set[:idx_of_mid_value]).median()

        idx_of_mid_value_for_third_quartile = idx_of_mid_value if self.is_num_of_items_even() else idx_of_mid_value + 1
        third_quartile = DataSetWithStatistics(sorted_data_set[idx_of_mid_value_for_third_quartile:]).median()

        return [first_quartile, self.median(), third_quartile]

    def interquartile_range(self):
        quantiles = self.quantiles()

        return quantiles[2] - quantiles[0]

    def remove_outliers_iqr(self, outlier_step=1.5, log=False):
        quantiles = self.quantiles()
        first_quantile = quantiles[0]
        third_quantile = quantiles[2]
        interquartile_range = third_quantile - first_quantile
        lower_limit = first_quantile - outlier_step * interquartile_range
        upper_limit = third_quantile + outlier_step * interquartile_range

        for value in reversed(self):
            if value < lower_limit or value > upper_limit:
                self.remove(value)
                if log:
                    print(f'The outlier {value} has been removed')