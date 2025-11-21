import datetime
import calendar

def get_first_last_days():
    # Get the current date
    current_date = datetime.date.today()

    # Calculate 5 months ago (for start date: 5 months ago, day 1)
    start_year, start_month = current_date.year, current_date.month
    for _ in range(5):  # Go back 5 months
        if start_month == 1:
            start_month = 12
            start_year -= 1
        else:
            start_month -= 1
    
    # Start date: first day of 5 months ago
    start_date = datetime.date(start_year, start_month, 1)

    # Calculate last month (for end date: last month, last day)
    if current_date.month == 1:
        last_month = 12
        last_year = current_date.year - 1
    else:
        last_month = current_date.month - 1
        last_year = current_date.year
    
    # End date: last day of last month
    last_day_of_last_month = calendar.monthrange(last_year, last_month)[1]
    end_date = datetime.date(last_year, last_month, last_day_of_last_month)

    # Format the dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Generate month labels: 4 months starting from 4 months ago (skip the 5 months ago start month)
    year_month_labels = []
    year, month = start_year, start_month
    # Skip first month (5 months ago), start from 4 months ago
    month += 1
    if month > 12:
        month = 1
        year += 1
    
    for _ in range(4):  # 4 consecutive months: 4,3,2,1 months ago
        year_month_labels.append(f"{year}-{str(month).zfill(2)}")
        month += 1
        if month > 12:
            month = 1
            year += 1

    return start_date_str, end_date_str, year_month_labels
