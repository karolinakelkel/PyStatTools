from math import sqrt
from scipy import stats
import Statistics


class DataSetHypothesisTester(Statistics.DataSetWithStatistics):
    def __init__(self, data_set=None):
        super().__init__(data_set)

    def one_sample_z_test(self, hypothetical_population_mean: [int, float], population_standard_deviation=None,
                          significance_level=0.05, direction_of_hypothesis='two-tailed', log=False):
        """
        Performs a one-sample z-test on the sample data.

        :returns: A list containing the z-value, p-value, and a boolean indicating whether to reject the null hypothesis.
        """

        if direction_of_hypothesis not in ['right-tailed', 'left-tailed', 'two-tailed']:
            raise ValueError('Invalid alternative hypothesis direction. Please, choose "two-tailed", "left-tailed" or '
                             '"right-tailed".')

        sample_mean = self.arithmetic_mean()

        if not population_standard_deviation:
            population_standard_deviation = self.standard_deviation()

        test_statistic = (sample_mean - hypothetical_population_mean) / (
                    population_standard_deviation / sqrt(self.size))

        if direction_of_hypothesis == 'two-tailed':
            critical_value = round(stats.norm.ppf(1 - significance_level / 2), 3)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        elif direction_of_hypothesis == 'right-tailed':
            critical_value = round(stats.norm.ppf(1 - significance_level), 3)
            p_value = 1 - stats.norm.cdf(test_statistic)
        else:
            critical_value = round(stats.norm.ppf(significance_level), 3)
            p_value = stats.norm.cdf(test_statistic)

        null_hypothesis_rejected = p_value < significance_level

        result = [('z-value', test_statistic),
                  ('p-value', p_value),
                  ('null hypothesis rejected', null_hypothesis_rejected)]

        if log:
            direction_sign = {'left-tailed': ['≥', '<'],
                              'right-tailed': ['≤', '>'],
                              'two-tailed': ['=', '≠']}.get(direction_of_hypothesis)

            print('-----------------\n'
                  'Hypothesis:\n'
                  f'H₀: µ {direction_sign[0]} {hypothetical_population_mean}\n'
                  f'H₁: µ {direction_sign[1]} {hypothetical_population_mean}\n'
                  '-----------------\n'
                  'Test type:\n'
                  f'{direction_of_hypothesis.capitalize()} One-sample Z-test\n'
                  '-----------------\n'
                  'Test statistic (Z-value):\n'
                  f'{test_statistic}\n'
                  '-----------------\n'
                  'P-value:\n'
                  f'{p_value}\n'
                  '-----------------\n'
                  'Critical value:\n'
                  f'{critical_value}\n'
                  '-----------------\n'
                  'Conclusion:')

            reject_or_fail_to_reject = 'reject' if null_hypothesis_rejected else 'fail to reject'
            if direction_of_hypothesis == 'two-tailed':
                print(f'Since the calculated z-value ({round(test_statistic, 4)}) falls '
                      f'{"outside" if null_hypothesis_rejected else "within"} '
                      f'the critical region (from {-critical_value} to {critical_value}) and the p-value '
                      f'({round(p_value, 4)}) is {"less" if null_hypothesis_rejected else "grater"} than the '
                      f'significance level ({significance_level}), we {reject_or_fail_to_reject} the null hypothesis.')
            elif direction_of_hypothesis == 'right-tailed':
                print(round(p_value, 4))
                print(f'Since the calculated z-value ({round(test_statistic, 4)}) is '
                      f'{"grater" if null_hypothesis_rejected else "less"} than the critical value '
                      f'({critical_value}) and the p-value ({round(p_value, 4)}) is '
                      f'{"less" if null_hypothesis_rejected else "grater"} than the significance level '
                      f'({significance_level}), we {reject_or_fail_to_reject} the null hypothesis.')
            else:
                print(f'Since the calculated z-value ({round(test_statistic, 4)}) is '
                      f'{"less" if null_hypothesis_rejected else "grater"} than the critical value '
                      f'({critical_value}) and the p-value ({round(p_value, 4)}) is '
                      f'{"less" if null_hypothesis_rejected else "grater"} than the significance level '
                      f'({significance_level}), we {reject_or_fail_to_reject} the null hypothesis.')
            print('-----------------')

        return result
