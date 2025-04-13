"""
This script exists to reduce code duplication across metrics.
"""

import numpy as np
import pandas as pd
import pandas.api.types
from typing import Union

class ParticipantVisibleError(Exception):
    pass

class HostVisibleError(Exception):
    pass

def treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    """
    Many metrics can raise errors that may leak competition data.
    This function attempts to judge if the error message is safe to show to participants.
    """
    # If solution is a DataFrame
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pd.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pd.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    # If solution is a numpy array
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pd.api.types.is_numeric_dtype(solution)
        solution_has_bools = pd.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if 'true' in error_message.lower() or 'false' in error_message.lower():
            return False
    return True

def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    """
    Calls the metric function. If the metric function raises an error that is safe to show to
    participants (because it is unlikely to leak sensitive data), then raise a ParticipantVisibleError.
    Otherwise, raise the error normally.
    """
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == 'ParticipantVisibleError':
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == 'HostVisibleError':
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    # Convert result to float (prevents issues with numpy scalar types)
    return float(score_result)

def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """
    Verify that the dataframe contains valid probabilities.
    """
    if not pd.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f'All target values in {df_name} must be numeric')

    if df.min().min() < 0:
        raise ParticipantVisibleError(f'All target values in {df_name} must be at least zero')

    if df.max().max() > 1:
        raise ParticipantVisibleError(f'All target values in {df_name} must be no greater than one')

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(f'Target values in {df_name} do not add to one within all rows')