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
                 path: str,
                 col_id: str,
                 col_event_time: Union[str, ColTransformer],
                 event_time_transformation: str = 'dt_to_timestamp',
                 cols_category: List[Union[str, ColCategoryTransformer]] = None,
                 category_transformation: str = 'frequency',
                 cols_numerical: List[str] = None,
                 cols_identity: List[str] = None,
                 cols_last_item: List[str] = None,
                 max_trx_count: int = None,
                 max_cat_num: Union[Dict[str, int], int] = 10000,
                 cols_first_item: List[str] = None,
                 return_records: bool = True,
                 n_jobs: int = -1,
                 t_user_group: ColTransformer = None,
                 ):

        self.dask_load_func = {'parquet': dd.read_parquet,
                               'csv': dd.read_csv,
                               'json': dd.read_json}

        self.dask_df = self._create_dask_dataset(path)
        self.multithread_dispatcher = DaskDispatcher()
        
        self.category_transformation = category_transformation
        self.event_time_transformation = event_time_transformation
        self.cols_first_item = cols_first_item
        self.return_records = return_records
        self.n_jobs = n_jobs
        super().__init__(col_id=col_id,
                         col_event_time=col_event_time,
                         cols_category=cols_category,
                         cols_identity=cols_identity,
                         cols_numerical=cols_numerical
                         )
        
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

    def _create_dask_dataset(self, path):
        for dataset in ['csv', 'json', 'parquet']:
            if path.__contains__(dataset):
                break
            else:
                raise AttributeError
        df = self.dask_load_func[dataset](path)
        return df

    def categorize(self):
        self.dask_df = self.dask_df.categorize(columns=self.dask_df.select_dtypes(include="category").columns.tolist())

    def create_dask_dataset(self):
        self.dask_df = self.dask_df.persist()

    # @staticmethod
    # def _td_default(df, cols_event_time):
    #     w = Window().orderBy(cols_event_time)
    #     tmp_df = df.select(cols_event_time).distinct()
    #     tmp_df = tmp_df.withColumn('event_time', F.row_number().over(w) - 1)
    #     df = df.join(tmp_df, on=cols_event_time)
    #     return df

    # @staticmethod
    # def _td_float(df, col_event_time):
    #     logger.info('To-float time transformation begins...')
    #     df = df.withColumn('event_time', F.col(col_event_time).astype('float'))
    #     logger.info('To-float time transformation ends')
    #     return df

    # @staticmethod
    # def _td_gender(df, col_event_time):
    #     """Gender-dataset-like transformation
    #     'd hh:mm:ss' -> float where integer part is day number and fractional part is seconds from day begin
    #     '1 00:00:00' -> 1.0
    #     '1 12:00:00' -> 1.5
    #     '1 01:00:00' -> 1 + 1 / 24
    #     '2 23:59:59' -> 1.99
    #     '432 12:00:00' -> 432.5   '000432 12:00:00'
    #     :param df:
    #     :param col_event_time:
    #     :return:
    #     """

    #     logger.info('Gender-dataset-like time transformation begins...')
    #     df = df.withColumn('_et_day', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 1, 6).cast('float'))

    #     df = df.withColumn('_et_time', F.substring(F.lpad(F.col(col_event_time), 15, '0'), 8, 8))
    #     df = df.withColumn('_et_time', F.regexp_replace('_et_time', r'\:60$', ':59'))
    #     df = df.withColumn('_et_time', F.unix_timestamp('_et_time', 'HH:mm:ss') / (24 * 60 * 60))

    #     df = df.withColumn('event_time', F.col('_et_day') + F.col('_et_time'))
    #     df = df.drop('_et_day', '_et_time')
    #     logger.info('Gender-dataset-like time transformation ends')
    #     return df

    # def _td_hours(self, df, col_event_time):
    #     logger.info('To hours time transformation begins...')
    #     df = df.withColumn('_dt', (F.col(col_event_time)).cast(dataType=T.TimestampType()))
    #     df = df.withColumn('event_time', ((F.col('_dt')).cast('float') - self.time_min) / 3600)
    #     df = df.drop('_dt')
    #     logger.info('To hours time transformation ends')
    #     return df

    def _reset(self):
        """Reset internal data-dependent state of the preprocessor, if necessary.
        __init__ parameters are not touched.
        """
        self.time_min = None
        self.remove_long_trx = False
        self.max_trx_count = 5000
        super()._reset()

    def pd_hist(self, df, name, bins=10):
        # logger.info('pd_hist begin')
        # logger.info(f'sf = {self.config.sample_fraction}')
        data = df.select(name)
        if self.config.sample_fraction is not None:
            data = data.sample(fraction=self.config.sample_fraction)
        data = data.toPandas()[name]

        if data.dtype.kind == 'f':
            round_len = 1 if data.max() > bins + 1 else 2
            bins = np.linspace(data.min(), data.max(), bins + 1).round(round_len)
        elif np.percentile(data, 99) - data.min() > bins - 1:
            bins = np.linspace(data.min(), np.percentile(data, 99), bins).astype(int).tolist() + [int(data.max() + 1)]
        else:
            bins = np.arange(data.min(), data.max() + 2, 1).astype(int)
        df = pd.cut(data, bins, right=False).rename(name)
        df = df.to_frame().assign(cnt=1).groupby(name)[['cnt']].sum()
        df['% of total'] = df['cnt'] / df['cnt'].sum()
        return df
    
    def _apply_aggregation(self, individuals: List[Dict], input_data):
        result_dict = reduce(lambda a, b: {**a, **b}, individuals)
        transformed_df = pd.concat(result_dict, axis=1)
        for agg_col, agg_fun in self.aggregate_func.items():
            transformed_df[agg_col] = input_data[agg_col]
            transformed_df = self.multithread_dispatcher.evaluate(individuals=transformed_df, objective_func=agg_fun)
        return transformed_df
    
    def fit_transform(self, X, y=None, **fit_params):
        X = dd.from_pandas(X, npartitions=12).persist()
        transformed_cols = Maybe.insert(self._chunk_data(dataset=X.compute(), func_to_transform=self._all_col_transformers)) \
            .maybe(default_value=None,
                   extraction_function=lambda chunked_data: self.multithread_dispatcher.evaluate(
                       individuals=chunked_data, objective_func=self.unitary_func))
        transformed_features = self._apply_aggregation(individuals=transformed_cols, input_data=X.compute())
        self.multithread_dispatcher.shutdown()
        return transformed_features