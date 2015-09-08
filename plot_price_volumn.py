import sys
import numpy
import dateutil.parser as parser
import csv
import pickle

import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, WeekdayLocator, DateFormatter
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU


def n_day_average(dates, vals, n):
    "average the value over n (5 or 7) days period to smooth the val fluctuation due to week"
    half_interval = n // 2
    n_day_average_vals = []
    indices = list(range(len(dates)))
    for date, val, index in zip(dates, vals, indices):
        if index < half_interval or index >= len(dates) - half_interval:
            n_day_average_vals.append(0)
        else:
            new_val = sum(vals[index - half_interval:index + half_interval + 1]) / n
            n_day_average_vals.append(new_val)
    n_day_average_vals[:half_interval] = [n_day_average_vals[half_interval]] * half_interval
    n_day_average_vals[-half_interval:] = [n_day_average_vals[-half_interval - 1]] * half_interval
    return n_day_average_vals


def week_agerage(dates, vals):
    week_sum = 0
    week_days_count = 0
    week_agerage_dates = []
    week_agerage_val = []
    for date, val in zip(dates, vals):
        year = date.year
        week_number = date.isocalendar()[1]
        weekyday = date.isocalendar()[2]
        if weekyday == 1:
            week_agerage_dates.append(date)
            week_agerage_val.append(week_sum / week_days_count)
            week_sum = 0
            week_days_count = 0
        week_sum += val
        week_days_count += 1
    return week_agerage_dates, week_agerage_val

def seansonally_adjust(dates, vals):
    val_sum = {}
    date_sum = {}
    for date, val in zip(dates, vals):
        key = str(date.month) + '-' + str(date.day)
        val_sum[key] = val_sum.get(key, 0) + val
        date_sum[key] = date_sum.get(key, 0) + 1
    adjusted_vals = []
    for date, val in zip(dates, vals):
        key = str(date.month) + '-' + str(date.day)
        average = val_sum[key] / date_sum[key]
        adjusted_vals.append(val / average)
    return adjusted_vals


def changes(series):
    changes = []
    for index in range(len(series)):
        if index > 0:
            changes.append((series[index] - series[index - 1]) / series[index - 1])
        else:
            changes.append(0)
    return changes


dates_sterling = []
counts = []
fname = sys.argv[1]

with open(fname, 'rb') as f:
    daily_applications = pickle.load(f)

for key, value in daily_applications.items():
    dt = parser.parse(key)
    dates_sterling.append(dt.date())
    counts.append(value)

dates_sterling, counts = (list(t) for t in zip(*sorted(zip(dates_sterling, counts))))

weekly_dates, weekly_counts = week_agerage(dates_sterling, counts)

n_day_average_counts = n_day_average(dates_sterling, counts, 7)

seansonally_adjusted = seansonally_adjust(dates_sterling, n_day_average_counts)

# fname = sys.argv[1]

# dates = []
# prices = []
# volumns = []

# with open(fname) as csvfile:
#     data = csv.reader(csvfile, delimiter=' ')
#     for row in data:
#         dt = parser.parse(row[0])
#         dates.append(dt.date())
#         prices.append(float(row[4]))
#         volumns.append(float(row[5]))

fig, ax = plt.subplots()
# ax.plot_date(dates, prices, '-')
# ax.plot_date(dates_sterling, counts, '-')
# ax.plot_date(weekly_dates, weekly_counts, '-')
ax.plot_date(dates_sterling, changes(n_day_average_counts), '-')
# ax.plot_date(dates_sterling, seansonally_adjusted, '-')



years = YearLocator()   # every year
months = MonthLocator()  # every month
weeks = WeekdayLocator(byweekday=MO)
yearsFmt = DateFormatter('%Y')
# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()


# format the coords message box
def price(x):
    return '%1.2f' % x

ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = price
ax.grid(True)

fig.autofmt_xdate()

# plt.hist(changes(n_day_average_counts), bins=31)
plt.show()

