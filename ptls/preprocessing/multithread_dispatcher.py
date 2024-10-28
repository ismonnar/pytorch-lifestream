from typing import Union, List, Dict, Callable

import pandas as pd
from joblib import wrap_non_picklable_objects, parallel_backend, Parallel, delayed
from pymonad.maybe import Maybe
from ptls.preprocessing.dask.dask_client import DaskServer


class DaskDispatcher:
    def __init__(self, n_jobs: int):
        self.dask_client = DaskServer(n_jobs).client
        print(f'Link Dask Server - {self.dask_client.dashboard_link}')
        self.transformation_func = 'fit_transform'
        self.n_jobs = n_jobs

    def shutdown(self):
        self.dask_client.close()
        del self.dask_client

    def _multithread_eval(self, individuals_to_evaluate: Union[Dict, pd.DataFrame],
                          objective_func: Union[Callable, Dict]):

        if isinstance(objective_func, dict):
            with parallel_backend(backend='dask',
                                  n_jobs=self.n_jobs,
                                  scatter=list(individuals_to_evaluate.values())
                                  ):
                evaluation_results = [self.evaluate_single(self,
                                                           data=individuals_to_evaluate[func_name],
                                                           eval_func=func_impl) for func_name, func_impl in
                                      objective_func.items()]
        else:
            evaluation_results = self.evaluate_single(self,
                                                      data=individuals_to_evaluate,
                                                      eval_func=objective_func)
        return evaluation_results

    def evaluate(self, individuals: Union[List, Dict], objective_func: Union[Callable, List[Callable]]):
        individuals_evaluated = Maybe.insert(individuals). \
            maybe(default_value=None,
                  extraction_function=lambda generation: self._multithread_eval(generation, objective_func))
        return individuals_evaluated

    @wrap_non_picklable_objects
    def evaluate_single(self,
                        data: Union[pd.DataFrame, pd.Series],
                        eval_func: Callable):
        if self.transformation_func == 'fit_transform':
            eval_res = eval_func.fit_transform(data)
        else:
            eval_res = eval_func.transform(data)
        return eval_res
