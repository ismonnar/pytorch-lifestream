from functools import reduce
from operator import iadd
from typing import List, Union, Callable, Dict

import pandas as pd
import dask.dataframe as dd
from pymonad.either import Either
from pymonad.maybe import Maybe
from sklearn.base import BaseEstimator, TransformerMixin

from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.base.transformation.col_event_time_transformer import DatetimeToTimestamp
from ptls.preprocessing.base.transformation.col_identity_transformer import ColIdentityEncoder
from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer
from ptls.preprocessing.base.transformation.user_group_transformer import UserGroupTransformer
from ptls.preprocessing.pandas.pandas_transformation.pandas_freq_transformer import FrequencyEncoder
from ptls.preprocessing.pandas.pandas_transformation.category_identity_encoder import CategoryIdentityEncoder


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 col_id: str,
                 col_event_time: Union[str, ColTransformer],
                 cols_category: List[Union[str, ColCategoryTransformer]] = None,
                 cols_numerical: List[str] = None,
                 cols_identity: List[str] = None,
                 t_user_group: ColTransformer = None,
                 ):
        self.cl_id = col_id
        self.ct_event_time = col_event_time
        self.cts_category = cols_category
        self.cts_numerical = cols_numerical
        self.cols_identity = cols_identity
        self.t_user_group = t_user_group
        self._init_transform_function()

        self._all_col_transformers = [
            [self.ct_event_time],
            self.cts_category,
            self.cts_numerical,
            [self.t_user_group],
        ]
        self.unitary_func, self.aggregate_func = {}, {}
        self._all_col_transformers = reduce(iadd, self._all_col_transformers, [])

    def _init_transform_function(self):
        self.cts_numerical = [ColIdentityEncoder(col_name_original=col) for col in self.cts_numerical]
        self.t_user_group = UserGroupTransformer(col_name_original=self.cl_id, cols_first_item=self.cols_first_item,
                                                 return_records=self.return_records)
        if isinstance(self.ct_event_time, str):  # use as is
            self.ct_event_time = Either(value=self.ct_event_time,
                                        monoid=['event_time',
                                                self.event_time_transformation == 'dt_to_timestamp']). \
                either(left_function=lambda x: ColIdentityEncoder(col_name_original=self.ct_event_time,
                                                                  col_name_target=x,
                                                                  is_drop_original_col=False),
                       right_function=lambda x: DatetimeToTimestamp(col_name_original=x))
        else:
            self.ct_event_time = self.ct_event_time

        if isinstance(self.cts_category[0], str):
            self.cts_category = Either(value=self.ct_event_time,
                                       monoid=['event_time',
                                               self.category_transformation == 'frequency']). \
                either(
                left_function=lambda x: [FrequencyEncoder(col_name_original=col) for col in self.cts_category],
                right_function=lambda x: [CategoryIdentityEncoder(col_name_original=col) for col in
                                          self.cts_category])

    def _chunk_data(self, dataset: Union[pd.DataFrame, dd.DataFrame], func_to_transform: List[Callable]):
        col_dict, self.func_dict = {}, {}
        for func_name in func_to_transform:
            if func_name.__repr__() == 'Unitary transformation':
                key = func_name if isinstance(func_name, str) else func_name.col_name_original
                value = dataset[func_name] if isinstance(func_name, str) else dataset[func_name.col_name_original]
                col_dict.update({key: value})
                self.unitary_func.update({func_name.col_name_original: func_name})
            else:
                self.aggregate_func.update({func_name.col_name_original: func_name})
        return col_dict

    def _apply_aggregation(self, individuals: List[Dict], input_data):
        result_dict = reduce(lambda a, b: {**a, **b}, individuals)
        transformed_df = pd.concat(result_dict, axis=1)
        for agg_col, agg_fun in self.aggregate_func.items():
            transformed_df[agg_col] = input_data[agg_col]
            transformed_df = agg_fun.fit_transform(transformed_df)
        return transformed_df

    def fit(self, x):
        for i, ct in enumerate(self._all_col_transformers):
            if i == len(self._all_col_transformers):
                ct.fit(x)
            else:
                x = ct.fit_transform(x)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        transformed_cols = Maybe.insert(self._chunk_data(dataset=X, func_to_transform=self._all_col_transformers)) \
            .maybe(default_value=None,
                   extraction_function=lambda chunked_data: [func_impl.fit_transform(chunked_data[func_name]) for func_name, func_impl in self.unitary_func.items()])
        transformed_features = self._apply_aggregation(individuals=transformed_cols, input_data=X)
        return transformed_features

    def transform(self, X):
        self.fit_transform(X)

    def get_category_dictionary_sizes(self):
        """Gets a dict of mapping to integers lengths for categories
        """
        return {ct.col_name_target: ct.dictionary_size for ct in self.cts_category}

    def to_yaml(self):
        raise NotImplementedError()

    def from_yaml(self):
        raise NotImplementedError()
