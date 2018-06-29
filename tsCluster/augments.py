from copy import deepcopy
import matplotlib
matplotlib.use('Qt5Agg')


def mixer(t1, t2, model, item, freq=2):

    """Function that will take two time series (vectors), STL decompose and swap elements in decomposition.

        Arguments:
            t1 -- The first time series
            t2 -- The second time series

        Keyword arguments:
            freq: The frequency of the periodic phenomenon in the time series, must be an integer.
            model: The STL model in use. Allowed arguments are 'additive' and 'multiplicative'
            item: The item in the decomposition to swap between time-series. Allowed values are 'seasonal', 'trend',
                and 'residue'

        Returns:
            rt1, rt2:  A tuple of new vectors

    """

    from statsmodels.tsa.seasonal import seasonal_decompose as s_dec

    r1 = s_dec(t1, freq=freq, model=model)
    r2 = s_dec(t2, freq=freq, model=model)

    if (item, model) == ('residue', 'additive'):
        return r1.seasonal + r1.trend + 1.1*r2.resid, r2.seasonal + r2.trend + 1.1*r1.resid
    elif (item, model) == ('residue', 'multiplicative'):
        return r1.seasonal * r1.trend * r2.resid, r2.seasonal * r2.trend * r1.resid
    elif (item, model) == ('trend', 'multiplicative'):
        return r1.seasonal * r2.trend * r1.resid, r2.seasonal * r1.trend * r2.resid
    elif (item, model) == ('trend', 'additive'):
        return r1.seasonal + r2.trend + r1.resid, r2.seasonal + r1.trend + r2.resid
    elif (item, model) == ('seasonal', 'additive'):
        return r2.seasonal + r1.trend + r1.resid, r1.seasonal + r2.trend + r2.resid
    elif (item, model) == ('seasonal', 'multiplicative'):
        return r2.seasonal * r1.trend * r1.resid, r1.seasonal * r2.trend * r2.resid
    else:
        raise ValueError


def augment_sub(samples, model='additive', item='residue'):

    """ Function to augment data given samples belonging to a single subject. """

    from itertools import combinations
    augmented = list()

    def multi_roi(arr1, arr2):

        """ Helper function that will swap decomposition element for each ROI in multi-dimensional time-series. """

        to_return1, to_return2 = list(), list()
        import numpy as np

        if arr1.shape != arr2.shape:
            raise ValueError
        else:
            for i in range(arr1.shape[0]):
                tn1, tn2 = mixer(arr1[i], arr2[i], model=model, item=item, freq=3)
                tn1[0] = tn1[1]     # Remove nan values that arise from decomposition, set to element next to it
                tn1[-1] = tn1[-2]
                tn2[0] = tn2[1]
                tn2[-1] = tn2[-2]
                to_return1.append(tn1)
                to_return2.append(tn2)
        return np.asarray(to_return1), np.asarray(to_return2)

    # Make sure samples belong to a single subject
    if len(set(sample['Name'] for sample in samples)) > 1:
            raise AttributeError

    for combination in combinations(samples,2):
        temp1 = deepcopy(combination[0])
        temp2 = deepcopy(combination[1])
        temp1['Name'] = temp1['Name'].replace('sub', 'aug')
        temp2['Name'] = temp2['Name'].replace('sub', 'aug')
        temp1['TimeSeries'], temp2['TimeSeries'] = multi_roi(combination[0]['TimeSeries'], combination[1]['TimeSeries'])
        augmented.append(temp1)
        augmented.append(temp2)

    return augmented


def augment_group(samples, num=None, model='additive', item='residue'):

    """ Function that augments a group in the data-set.

        Arguments:
            samples: samples belonging to a single group
            num: The number of subjects in the list whose data can be used for augmentation
            model: The decomposition model
            item: Component of the decomposition to hot-swap
    """
    import random
    all_names = set([sample['Name'] for sample in samples])
    augmented = list()

    # Check only a single group
    if len(set([sample['Group'] for sample in samples])) > 1:
        raise ValueError

    if num:
        names = random.sample(all_names, num)
    else:
        names = all_names

    for name in names:
        by_name = [sample for sample in samples if sample['Name'] == name]
        augmented.extend(augment_sub(by_name, model=model, item=item))

    return augmented
