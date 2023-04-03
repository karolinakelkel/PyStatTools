from math import sqrt


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

        :returns: A new DataSet object with each element raised to the specified power.
        """

        return DataSet([value ** power for value in self])

    def __mul__(self, other: 'DataSet'):
        """
         Compute the element-wise product of two datasets.

         :returns: A new DataSet object with the element-wise product of the two datasets.
        """

        return DataSet([first_set_value * second_set_value for first_set_value, second_set_value in zip(self, other)])


class DataSetWithStatistics(DataSet):
    def __init__(self, data_set: list):
        super().__init__(data=data_set)

        self.__sum_of_observations = None
        self.__is_num_of_items_even = None

    def sum_of_observations(self) -> [int, float]:
        """
        Computes the sum of the observations in the dataset.

        :returns: The sum of the observations in the dataset.
        """

        if self.__sum_of_observations is None:
            self.__sum_of_observations = sum(self)

        return self.__sum_of_observations

    def is_num_of_items_even(self) -> bool:
        """
        Determines whether the number of items in the dataset is even.

        :returns: True if the number of items in the dataset is even, False otherwise.
        """
        if self.__is_num_of_items_even is None:
            self.__is_num_of_items_even = self.size % 2 == 0

        return self.__is_num_of_items_even

    def arithmetic_mean(self, bessel_correction=False) -> float:
        """
        Compute the arithmetic mean of a sorted dataset. If the 'bessel_correction' parameter is set to True,
        the method applies Bessel's correction to the sample mean formula.

        :returns: The arithmetic mean of the dataset.
        """

        sum_of_observations = self.sum_of_observations

        if bessel_correction:
            return sum_of_observations() / (self.size - 1)
        else:
            return sum_of_observations() / self.size

    def median(self) -> [int, float]:
        """
        Compute the arithmetic mean of a sorted dataset.

        :returns: The median of the dataset.
        """

        if self.is_num_of_items_even() is False:
            return self[self.size // 2]

        idx_of_second_mid_value = self.size // 2
        idx_of_first_mid_value = idx_of_second_mid_value - 1

        return (self[idx_of_first_mid_value] + self[idx_of_second_mid_value]) / 2

    def mode(self) -> [int, float]:
        """
        Compute the mode of the dataset.

        :returns: The mode of the dataset.
        """

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
        """
        Compute the range of the dataset.

        :returns: The range of the dataset.
        """

        return max(self) - min(self)

    def diff_between_values_and_mean(self, squared=True) -> 'DataSetWithStatistics':
        """
        Compute the difference between each value in the dataset and the arithmetic mean. If the 'squared' parameter is
        True, the squared differences are returned.

        :returns: A new DataSetWithStatistics object containing the computed differences between each value in the
        dataset and the arithmetic mean.
        """

        data_set_mean = self.arithmetic_mean()
        diff = DataSetWithStatistics([value - data_set_mean for value in self])

        return DataSetWithStatistics(diff ** 2 if squared else diff)

    def variance(self, bessel_correction=True):
        """
        Compute the variance of the dataset. If the 'bessel_correction' parameter is True, Bessel's correction is
        applied to the sample variance formula.

        :returns: The computed variance as a float.
        """

        squared_differences = self.diff_between_values_and_mean()

        return squared_differences.arithmetic_mean(bessel_correction)

    def standard_deviation(self, bessel_correction=True):
        """
        Compute the standard deviation of the dataset. If the 'bessel_correction' parameter is True, Bessel's
        correction is applied to the formula.

        :returns: The computed standard deviation as a float.
        """

        return sqrt(self.variance(bessel_correction))

    def covariance(self, other, bessel_correction=True, raw_covariance=False):
        """
        Compute the covariance between two datasets. If the 'bessel_correction' parameter is True,
        Bessel's correction is applied to the covariance formula. If the 'raw_covariance' parameter is True,
        the raw covariance is returned.

        :returns: The computed covariance between the two datasets as a float, or the raw covariance if
        the 'raw_covariance' parameter is True.
        """

        set_1_diff = self.diff_between_values_and_mean(squared=False)
        set_2_diff = other.diff_between_values_and_mean(squared=False)
        products_of_sets = set_1_diff * set_2_diff

        return sum(products_of_sets) if raw_covariance \
            else DataSetWithStatistics(products_of_sets).arithmetic_mean(bessel_correction)

    def pearson_correlation_coefficient(self, other):
        """
        Computes the Pearson correlation coefficient between two datasets.

        :returns: The computed Pearson correlation coefficient between two datasets as a float.
        """

        raw_covariance = self.covariance(other, raw_covariance=True)
        set_1_sum_of_squared_diffs = sum(self.diff_between_values_and_mean(squared=True))
        set_2_sum_of_squared_diffs = sum(other.diff_between_values_and_mean(squared=True))
        product_of_roots = sqrt(set_1_sum_of_squared_diffs) * sqrt(set_2_sum_of_squared_diffs)

        return raw_covariance / product_of_roots

    def quantiles(self) -> list:
        """
        Computes the first, second (median), and third quartiles of a dataset.

        :returns: A list of the first, second (median), and third quartiles of the dataset.
        """

        sorted_data_set = DataSetWithStatistics(sorted(self))

        idx_of_mid_value = sorted_data_set.size // 2
        first_quartile = DataSetWithStatistics(sorted_data_set[:idx_of_mid_value]).median()

        idx_of_mid_value_for_third_quartile = idx_of_mid_value if self.is_num_of_items_even() else idx_of_mid_value + 1
        third_quartile = DataSetWithStatistics(sorted_data_set[idx_of_mid_value_for_third_quartile:]).median()

        return [first_quartile, self.median(), third_quartile]

    def interquartile_range(self):
        """
        Computes the interquartile range (IQR) of a dataset

        :returns: The computed IQR of the dataset.
        """

        quantiles = self.quantiles()
        return quantiles[2] - quantiles[0]

    def remove_outliers_iqr(self, outlier_step=1.5, log=False):
        """
        Removes outliers from a dataset using the interquartile range (IQR) method.

        :returns: None. The method modifies the dataset in place.
        """

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
