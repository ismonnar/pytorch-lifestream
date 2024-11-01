import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Callable
from functools import reduce
from operator import iadd

import dask
import dask.dataframe as dd
from pymonad.either import Either
from pymonad.maybe import Maybe

from ptls.preprocessing.base import DataPreprocessor
from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.base.transformation.col_identity_transformer import ColIdentityEncoder
from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer
from ptls.preprocessing.dask.dask_transformation.category_identity_encoder import CategoryIdentityEncoder
from ptls.preprocessing.dask.dask_transformation.event_time import DatetimeToTimestamp
from ptls.preprocessing.dask.dask_transformation.frequency_encoder import FrequencyEncoder
from ptls.preprocessing.dask.dask_transformation.user_group_transformer import UserGroupTransformer
from ptls.preprocessing.multithread_dispatcher import DaskDispatcher
from ptls.preprocessing.util import determine_n_jobs

logger = logging.getLogger(__name__)


class DaskDataPreprocessor(DataPreprocessor):
    """Data preprocessor based on dask.dataframe

    During preprocessing it
        * transforms `cols_event_time` column with date and time
        * encodes category columns `cols_category` into ints;
        * apply logarithm transformation to `cols_log_norm' columns;
        * (Optional) select the last `max_trx_count` transactions for each `col_id`;
        * groups flat data by `col_id`;
        * arranges data into list of dicts with features

    Parameters
    ----------
    col_id : str
        name of column with ids
    cols_event_time : str,
        name of column with time and date
    cols_category : list[str],s
        list of category columns
    cols_log_norm : list[str],
        list of columns to be logarithmed
    cols_identity : list[str],
        list of columns to be passed as is without any transformation
    cols_target: List[str],
        list of columns with target
    time_transformation: str. Default: 'default'.
        type of transformation to be applied to time column
    remove_long_trx: bool. Default: False.
        If True, select the last `max_trx_count` transactions for each `col_id`.
    max_trx_count: int. Default: 5000.
        used when `remove_long_trx`=True
    print_dataset_info : bool. Default: False.
        If True, print dataset stats during preprocessor fitting and data transformation
    """

    def __init__(self,
                 col_id: str,
                 col_event_time: Union[str, ColTransformer],
                 event_time_transformation: str = 'dt_to_timestamp',
                 cols_category: List[Union[str, ColCategoryTransformer]] = None,
                 category_transformation: str = 'frequency',
                 cols_numerical: List[str] = None,
                 cols_identity: List[str] = None,
                 cols_first_item: List[str] = None,
                 return_records: bool = True,
                 t_user_group: ColTransformer = None,
                 ):
        self.category_transformation = category_transformation
        self.event_time_transformation = event_time_transformation
        self.cols_first_item = cols_first_item
        self.return_records = return_records
        self.n_jobs = determine_n_jobs(-1)
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
        self.multithread_dispatcher = DaskDispatcher(self.n_jobs)
        
        self.dask_load_func = {'parquet': dd.read_parquet,
                               'csv': dd.read_csv,
                               'json': dd.read_json,
                               'pd.DataFrame': dd.from_pandas}
        
    def _init_transform_function(self):
        self.cts_numerical = [ColIdentityEncoder(col_name_original=col) for col in self.cts_numerical]
        self.t_user_group = UserGroupTransformer(col_name_original=self.cl_id, cols_first_item=self.cols_first_item,
                                                 return_records=self.return_records, n_jobs=self.n_jobs)
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

    def _create_dask_dataset(self, path_or_data):
        for dataset in list(self.dask_load_func.keys()):
            if path_or_data.__contains__(dataset):
                df = self.dask_load_func[dataset](path_or_data)
            elif dataset == 'pd.DataFrame':
                df = self.dask_load_func[dataset](path_or_data, npartitions=self.n_jobs*5)
        return df

    def categorize(self):
        self.dask_df = self.dask_df.categorize(columns=self.dask_df.select_dtypes(include="category").columns.tolist())

    def create_dask_dataset(self):
        self.dask_df = self.dask_df.persist()

    def _apply_aggregation(self, individuals: List[Dict], input_data):
        result_dict = reduce(lambda a, b: {**a, **b}, individuals)
        transformed_df = dd.concat([result_dict], axis=1)
        for agg_col, agg_fun in self.aggregate_func.items():
            transformed_df[agg_col] = input_data[agg_col]
            transformed_df = self.multithread_dispatcher.evaluate(individuals=transformed_df, objective_func=agg_fun)
        return transformed_df
    
    def fit_transform(self, X, y=None, **fit_params):
        self.dask_df = X if isinstance(X, dd.DataFrame) else self._create_dask_dataset(X)
        transformed_cols = Maybe.insert(self._chunk_data(dataset=self.dask_df, func_to_transform=self._all_col_transformers)) \
            .maybe(default_value=None,
                   extraction_function=lambda chunked_data: self.multithread_dispatcher.evaluate(
                       individuals=chunked_data, objective_func=self.unitary_func))
        transformed_features = self._apply_aggregation(individuals=transformed_cols, input_data=self.dask_df)
        self.multithread_dispatcher.shutdown()
        return transformed_features