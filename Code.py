import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load dataset ──────────────────────────────────
script_dir = Path(__file__).resolve().parent
candidate_names = [
    'Metro_Interstate_Traffic_Volume.csv',
    'Metro_Interstate_Traffic_Volume.csv.gz',
    'metro traffic.gz',
    'metro_traffic.csv',
    'metro_traffic.csv.gz'
]

csv_candidates = [script_dir / name for name in candidate_names]
csv_candidates.extend(sorted(script_dir.glob('*.csv')))
csv_candidates.extend(sorted(script_dir.glob('*.csv.gz')))
csv_candidates.extend(sorted(script_dir.glob('*.gz')))

seen = set()
for csv_path in csv_candidates:
    resolved = csv_path.resolve()
    if resolved in seen:
        continue
    seen.add(resolved)

    if resolved.exists() and resolved.is_file():
        try:
            df = pd.read_csv(resolved, compression='infer')
            print(f"Loaded dataset: {resolved.name}")
            break
        except Exception:
            continue
else:
    raise FileNotFoundError(
        'Dataset not found. Place the traffic dataset (.csv or .csv.gz/.gz) '
        'in the same folder as this script.'
    )

print(df.shape, df.dtypes)

# ── 2. Datetime parsing ───────────────────────────────
df['date_time'] = pd.to_datetime(df['date_time'])

df['hour']       = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.day_name()
df['month']      = df['date_time'].dt.month
df['year']       = df['date_time'].dt.year
df['is_weekend'] = df['date_time'].dt.dayofweek >= 5

# ── 3. Missing values check ───────────────────────────
df['holiday'] = df['holiday'].fillna('None')
print("Missing values:\n", df.isnull().sum())
df.dropna(inplace=True)

# ── 4. Outlier removal (traffic_volume) ──────────────
Q1 = df['traffic_volume'].quantile(0.01)
Q3 = df['traffic_volume'].quantile(0.99)
df = df[(df['traffic_volume'] >= Q1) &
        (df['traffic_volume'] <= Q3)]

# ── 5. Traffic level labels ───────────────────────────
df['traffic_level'] = pd.cut(
    df['traffic_volume'],
    bins=[0, 1500, 3500, 5000, 9999],
    labels=['Low', 'Medium', 'High', 'Critical']
)

print(df['traffic_level'].value_counts())

# ── 6. Additional traffic pattern analysis ─────────────
# Average traffic by hour of day
hourly = df.groupby('hour')['traffic_volume'].mean().reset_index()
peak_hour = hourly.loc[hourly['traffic_volume'].idxmax(), 'hour']
print(f"Peak hour: {peak_hour}:00")

# Average traffic by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday', 'Sunday']
daily = (df.groupby('day_of_week')['traffic_volume']
           .mean()
           .reindex(day_order))
print("\nAverage traffic by day of week:")
print(daily)

# Weekend vs weekday comparison
wk_compare = df.groupby('is_weekend')['traffic_volume'].mean()
print("\nWeekday avg:", round(wk_compare.get(False, np.nan), 1))
print("Weekend avg:", round(wk_compare.get(True, np.nan), 1))

# Heatmap pivot: hour x day
heatmap_data = (df.pivot_table(
    values='traffic_volume',
    index='hour',
    columns='day_of_week',
    aggfunc='mean'
)
.reindex(columns=day_order))
print("\nHeatmap pivot (hour x day):")
print(heatmap_data)

# Monthly trend
monthly = df.groupby(['year', 'month'])['traffic_volume'].mean().reset_index()
monthly['period'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str)
print("\nMonthly average traffic:")
print(monthly[['period', 'traffic_volume']])

# Weather impact
weather_impact = (df.groupby('weather_main')['traffic_volume']
                    .mean()
                    .sort_values(ascending=False))
print("\nWeather vs Traffic:\n", weather_impact)



sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle('Traffic Congestion Dashboard — City Planners',
             fontsize=18, fontweight='bold', y=1.01)

# ── Chart 1: Avg traffic by hour ──────────────────────
ax1 = axes[0, 0]
colors = ['#E24B4A' if v == hourly['traffic_volume'].max()
          else '#378ADD'
          for v in hourly['traffic_volume']]
ax1.bar(hourly['hour'], hourly['traffic_volume'], color=colors)
ax1.set_title('Peak Traffic Hours')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Avg Traffic Volume')
ax1.axvline(x=peak_hour, color='red', linestyle='--', label=f'Peak: {peak_hour}h')
ax1.legend()

# ── Chart 2: Heatmap hour x day ───────────────────────
ax2 = axes[0, 1]
sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax2,
  linewidths=0.3, cbar_kws={'label': 'Avg Volume'})
ax2.set_title('Congestion Heatmap (Hour × Day)')
ax2.set_xlabel('Day of Week')
ax2.set_ylabel('Hour of Day')

# ── Chart 3: Day of week bar ───────────────────────────
ax3 = axes[1, 0]
daily.plot(kind='bar', ax=ax3, color='#1D9E75', edgecolor='white')
ax3.set_title('Avg Traffic by Day of Week')
ax3.set_xlabel('')
ax3.set_ylabel('Avg Traffic Volume')
ax3.tick_params(axis='x', rotation=30)

# ── Chart 4: Weather impact ───────────────────────────
ax4 = axes[1, 1]
weather_impact.plot(kind='barh', ax=ax4, color='#7F77DD')
ax4.set_title('Weather Condition vs Traffic Volume')
ax4.set_xlabel('Avg Traffic Volume')

# ── Chart 5: Monthly trend line ───────────────────────
ax5 = axes[2, 0]
ax5.plot(monthly['period'], monthly['traffic_volume'],
marker='o', color='#D85A30', linewidth=2)
ax5.set_title('Monthly Traffic Trend')
ax5.set_xlabel('Month')
ax5.set_ylabel('Avg Traffic Volume')
ax5.tick_params(axis='x', rotation=45)

# ── Chart 6: Traffic level distribution ──────────────
ax6 = axes[2, 1]
level_counts = df['traffic_level'].value_counts()
level_counts.plot(kind='pie', ax=ax6, autopct='%1.1f%%',
                  colors=['#1D9E75','#378ADD','#EF9F27','#E24B4A'],
                  startangle=90)
ax6.set_title('Traffic Level Distribution')
ax6.set_ylabel('')

plt.tight_layout()
plt.savefig('traffic_dashboard.png', dpi=150, bbox_inches='tight')
plt.close(fig)
