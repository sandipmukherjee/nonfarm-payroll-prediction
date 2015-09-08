# Predict nonfarm payroll with sterling data.
import pandas as pd
from sklearn import linear_model
import numpy as np
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pprint import pprint


class NFPPredictor(object):

    def __init__(self, monthly_data, nfp_full, number_of_prev_nfp_features):
        self.num_of_month = len(monthly_data)
        self.num_previous_month = len(monthly_data) - 1
        self.nfp_full_data = nfp_full
        self.predictor = linear_model.Lasso(alpha=0.1, max_iter=1000000)
        self._generate_feature_dict(monthly_data)
        self._generate_nfp_dict(nfp_full)
        self.df = pd.DataFrame(columns=self.feature_names)
        self.df = pd.DataFrame.from_dict(self.feature_dict, orient="index")
        self.df.columns = self.feature_names
        self.df = self.df.sort()
        self.df_norm = (self.df - self.df.mean()) / \
            (self.df.max() - self.df.min())

        self._add_nfp_to_df()
        if number_of_prev_nfp_features != 0:
            self._include_nfp_prev_as_features(number_of_prev_nfp_features)

    def _generate_feature_dict(self, monthly_data):
        self.feature_dict = dict()
        for one_month_feature in monthly_data:
            self.feature_names = list(one_month_feature.keys())
            month = one_month_feature['month']
            self.feature_names.remove('month')
            one_month_feature.pop("month", None)
            self.feature_dict[month] = one_month_feature

    def _generate_nfp_dict(self, nfp_full):
        # NFP is expected to be a monthly sorted list of dictionaries
        # {"month": month, "NFP": NFP_value}
        self.NFP_dict = {}
        for one_month_NFP in nfp_full:
            self.NFP_dict[one_month_NFP["month"]] = one_month_NFP["NFP"]

    def _add_nfp_to_df(self):
        nonfarm = []
        for month in self.df.index.values:
            # next_month = (datetime.strptime(month, "%Y-%m") + relativedelta(months=1)).strftime("%Y-%m")
            next_month = (datetime.strptime(month, "%Y-%m")).strftime("%Y-%m")
            nonfarm.append(self.NFP_dict[next_month])
        self.df_norm['nonfarm'] = nonfarm

    def _include_nfp_prev_as_features(self, num_prev_month):
        for i in range(1, num_prev_month+1):
            prev_i = [self.NFP_dict[(datetime.strptime(self.df_norm.index[[j]].values[0], "%Y-%m") -
                                     relativedelta(months=i)).strftime("%Y-%m")] for j in range(len(self.df_norm))]
            self.df_norm.loc[:, ('NFP_prev' + str(i))] = prev_i
            self.feature_names.append('NFP_prev' + str(i))

    def _fit_and_test(self, train, test):
        x, y = train[self.feature_names].values, train['nonfarm'].values
        self.predictor.fit(x, y)
        prediction = self.predictor.predict(test[self.feature_names].values)
        mean_error = np.mean(abs(test['nonfarm'].values - prediction) / test['nonfarm'].values)
        return mean_error

    def cal_error(self):
        # print(self.num_previous_month)
        for n in range(-6, -1):
            train = self.df_norm[:n]
            test = self.df_norm.ix[[n]]
            yield self._fit_and_test(train, test)

    def fit_and_predict(self, train, predict_month):
        x, y = train[self.feature_names].values, train['nonfarm'].values
        self.predictor.fit(x, y)
        prediction = self.predictor.predict(predict_month[self.feature_names].values)
        mean_error = np.mean(list(self.cal_error()))
        print("Mean Error:", mean_error)
        max_prediction = prediction[0] * (1 + mean_error)
        min_prediction = prediction[0] * (1 - mean_error)
        # return {'prediction': prediction[0]}
        next_month_prediction = {'month': predict_month.index.values[0],
                                 'prediction': prediction[0],
                                 'max_prediction': max_prediction,
                                 'min_prediction': min_prediction}
        return next_month_prediction

    def get_last_months_predictions(self, number_of_predict_months):
        result_list = []
        for x in range(number_of_predict_months):
            n = x + 1
            train = self.df_norm[:-n]
            predict_month = self.df_norm.ix[[-n]]
            result_list.append(self.fit_and_predict(train, predict_month))
        return result_list


def get_nfp():
    nonfarm_payroll = [161.0, 44.0, 332.0, 249.0, 307.0, 74.0, 32.0, 132.0, 162.0, 346.0, 65.0, 129.0,
                       134.0, 239.0, 134.0, 363.0, 175.0, 245.0, 373.0, 196.0, 67.0, 84.0, 337.0, 159.0,
                       277.0, 315.0, 280.0, 182.0, 23.0, 77.0, 207.0, 184.0, 157.0, 2.0, 210.0, 171.0,
                       238.0, 88.0, 188.0, 78.0, 144.0, 71.0, -33.0, -16.0, 85.0, 82.0, 118.0, 97.0,
                       15.0, -86.0, -80.0, -214.0, -182.0, -172.0, -210.0, -259.0, -452.0, -474.0, -765.0, -697.0,
                       -798.0, -701.0, -826.0, -684.0, -354.0, -467.0, -327.0, -216.0, -227.0, -198.0, -6.0, -283.0,
                       18.0, -50.0, 156.0, 251.0, 516.0, -122.0, -61.0, -42.0, -57.0, 241.0, 137.0, 71.0,
                       70.0, 168.0, 212.0, 322.0, 102.0, 217.0, 106.0, 122.0, 221.0, 183.0, 164.0, 196.0,
                       360.0, 226.0, 243.0, 96.0, 110.0, 88.0, 160.0, 150.0, 161.0, 225.0, 203.0, 214.0,
                       197.0, 280.0, 141.0, 203.0, 199.0, 201.0, 149.0, 202.0, 164.0, 237.0, 274.0, 84.0,
                       144.0, 222.0, 203.0, 304.0, 229.0, 267.0, 243.0, 203.0, 271, 261, 423, 329,
                       201, 266, 119, 187, 254, 223]
    dt = datetime(2004, 1, 15)
    end = datetime(2015, 7, 15)
    step = relativedelta(months=1)
    nfp_list = []
    i = 0
    while dt < end:
        nfp_list.append({"month": dt.strftime('%Y-%m'), "NFP": nonfarm_payroll[i]})
        dt += step
        i += 1
    pprint(nfp_list)
    return nfp_list


if __name__ == '__main__':
    with open('monthly_features_new.pickle', 'rb') as f:
        monthly_features = pickle.load(f)
        monthly_features = monthly_features[:-1]  # truncate the last month as it is not full month
        pprint(monthly_features[-1:])
        nfp = get_nfp()
        predictor = NFPPredictor(monthly_features, nfp, number_of_prev_nfp_features=9)
        print(predictor.df_norm)
        result_past_months = predictor.get_last_months_predictions(3)
        print(result_past_months)
