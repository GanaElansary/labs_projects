import pandas as pd
from scipy.stats import skew as skew_func

def skew_calc(college):
    """
    Diagnoses skewness for every numeric column in a DataFrame and recommends a transformation based on the column's skewness and
    minimum value. Binary, encoded, and ID columns are excluded, since skewness isn't a meaningful for them.
    It returns a DataFrame with the following columns:
    Feature, Skewness, Degree, Direction, Recommended Transformation
    """
    rows = []

    for col in college.select_dtypes(include='number').columns:
        if college[col].nunique() <= 2:
            continue

        col_skew = skew_func(college[col], bias=False)

        if col_skew < -1:
            degree, direction = 'High', 'Left'
        elif col_skew < -0.5:
            degree, direction = 'Moderate', 'Left'
        elif col_skew <= 0.5:
            degree, direction = 'Normal', 'Symmetrical'
        elif col_skew <= 1:
            degree, direction = 'Moderate', 'Right'
        else:
            degree, direction = 'High', 'Right'

        if degree == 'Normal':
            recommendation = 'No Transformation Needed'
        elif college[col].min() < 0:
            recommendation = 'Yeo-Johnson'
        elif college[col].min() == 0:
            recommendation = 'Log Plus One'
        else:
            recommendation = 'Box-Cox'

        rows.append({
            'Feature': col,
            'Skewness': round(col_skew, 4),
            'Degree': degree,
            'Direction': direction,
            'Recommended Transformation': recommendation
        })

    return pd.DataFrame(rows)
