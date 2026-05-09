estimate_price.py
from datetime import datetime
import numpy as np


# Example predefined variables (replace with actual values in your use case)
epoch = datetime(2000, 1, 1).date()
period = 365.25
coef = np.array([1000, 0.5, 50, -20])


def estimate_price(date_input):
    if isinstance(date_input, str):
        for fmt in ('%Y-%m-%d','%m/%d/%y','%m/%d/%Y','%d-%m-%Y'):
            try:
                dt = datetime.strptime(date_input, fmt).date()
                break
            except Exception:
                dt = None
        if dt is None:
            raise ValueError('Unrecognized date format. Use YYYY-MM-DD or MM/DD/YY.')
    elif isinstance(date_input, datetime):
        dt = date_input.date()
    else:
        dt = date_input
    
    days = (dt - epoch).days
    x_pred = np.array([1, days, np.sin(2*np.pi*days/period), np.cos(2*np.pi*days/period)])
    est = float(x_pred.dot(coef))
    return est