def racog_fit_resample(X, y):
    """Try R-based RACOG via rpy2; fall back to identity if unavailable."""
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.packages as rpackages
        import pandas as pd
        base = rpackages.importr('base')
        imbalance = rpackages.importr('imbalance')
        # Convert to R data.frame
        import numpy as np
        if not isinstance(X, (pd.DataFrame,)):
            import pandas as pd
            X = pd.DataFrame(X)
        df = X.copy()
        df['label'] = y
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        r_df = pandas2ri.py2rpy(df)
        res = imbalance.RACOG(r_df, "label")
        py = pandas2ri.rpy2py(res)
        y_new = py['label'].astype(int).values
        X_new = py.drop(columns=['label']).values
        return X_new, y_new
    except Exception as e:
        # fallback
        return X, y
