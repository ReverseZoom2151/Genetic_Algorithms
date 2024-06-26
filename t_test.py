# This code implements a simple statistical test - the "t-test" - for checking whether two data sets are significantly different

import math         # uses "sqrt"
import statistics   # uses "mean", "stdev"

# To use the t_test function, it can be called with two lists of data:

    # t, df = t_test.t_test(data1, data2)

# "t" is the t-test statistic itself - a higher value means more difference between the two datasets
# "df" is the "degrees of freedom" of the data - this is needed in order to judge whether "t" is high enough to be "significant"

# Runs a t-test on two samples - to tell whether they are statistically significantly different from each other
#   (adapted from: https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/)
#   data1 and data2 are each a list of numbers
#   t-test expects each set of numbers to be normally distributed with similar std deviation

def t_test(data1, data2):

    # calculates mean, standard deviation (stdev) and standard error (sterr) of each sample
    mean1, mean2   = statistics.mean(data1), statistics.mean(data2)
    stdev1, stdev2 = statistics.stdev(data1), statistics.stdev(data2)
    sterr1, sterr2 = stdev1 / math.sqrt(len(data1)), stdev2 / math.sqrt(len(data2))

    # calculates the standard error for the difference between the sample means (sterrdm)
    sterrdm = math.sqrt(sterr1 * sterr1 + sterr2 * sterr2)

    # calculates the "t" statistic
    t = (mean1 - mean2) / sterrdm

    # calculates the "degrees of freedom" ("df") = (n1 - 1) + (n2 - 1)
    df = len(data1) + len(data2) - 2

    return t, df


# Note on the t test:
#   The t test is used to tell if two samples are likely to be drawn from two different Normal distributions
#   The "t" statistic is calculated as: (mean1 - mean2) / sterrdm where mean1 and mean2 are the means of the two samples
#                                   and sterrdm is the standard error of the differences between the two means
#   t captures the difference between the sample means relative to the spread of values in the two samples
#   If t is large enough we say that the difference between the samples is *significant*
#   To check for significance, scipy or some other python package can be used:
#     or the values for "t" and "df" can be given to an online calculator:
#     e.g., https://www.statology.org/t-score-p-value-calculator/
