# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rain, Cold and Ride-Hailing: a PySpark pipeline on 80M Uber/Lyft trips
#
# **Question:** how does weather change ride-hailing demand, prices, waiting time and driver pay in New York City, and does it hit Uber and Lyft differently?
#
# **Data (Jan-Apr 2024)**
#
# | Dataset | Source | Format | Size |
# |---|---|---|---|
# | High Volume For-Hire Vehicle trips (Uber, Lyft) | NYC Taxi & Limousine Commission | Parquet, 4 monthly files | ~80M rows, 1.8 GB compressed |
# | Taxi zone lookup | NYC TLC | CSV | 265 rows |
# | Daily weather, Central Park station USW00094728 | NOAA GHCN-Daily | CSV | 1 row per day |
#
# **Notebook map**
#
# 1. Setup and data download
# 2. Load
# 3. Transformations vs actions (lazy vs eager)
# 4. The pipeline: filters, `withColumn`, joins, `groupBy`
# 5. SQL queries
# 6. Write results (partitioned Parquet)
# 7. Performance analysis: `.explain()`, pushdown, join strategy
# 8. Caching
# 9. MLlib: predicting hourly demand

# %% [markdown]
# ## 1. Setup
#
# The notebook runs locally (it builds its own `SparkSession`) and on Databricks (it reuses the provided `spark`).

# %%
import os
import sys
import time
import urllib.request

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    spark = SparkSession.builder.getOrCreate()
    spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.ride_weather")
    ROOT = "/Volumes/workspace/default/ride_weather"
else:
    ROOT = os.getcwd()
    os.environ["PYSPARK_PYTHON"] = sys.executable       # workers must use the same Python as the driver
    os.makedirs(f"{ROOT}/spark-events", exist_ok=True)
    spark = (
        SparkSession.builder.appName("uber-lyft-weather")
        .master("local[6]")                              # 6 worker threads = 6 parallel tasks
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "24")    # default 200 is too many for one machine
        .config("spark.sql.debug.maxToStringFields", "100")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.eventLog.enabled", "true")        # keeps the Spark UI history after the run
        .config("spark.eventLog.dir", f"file://{ROOT}/spark-events")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

RAW = f"{ROOT}/data/raw"
OUT = f"{ROOT}/data/output"
print("Spark", spark.version, "| Databricks:", IS_DATABRICKS, "| root:", ROOT)


# %%
def timed(label, action):
    """Run an action (a zero-argument function) and print how long it took."""
    describe(label)
    start = time.time()
    result = action()
    seconds = time.time() - start
    print(f"[{seconds:6.2f}s] {label}")
    return result, seconds


def describe(label):
    """Label the next jobs in the Spark UI (not available on Databricks serverless)."""
    try:
        spark.sparkContext.setJobDescription(label)
    except Exception:
        pass


def num_partitions(df):
    try:
        return df.rdd.getNumPartitions()
    except Exception:
        return "n/a (serverless)"


# %% [markdown]
# ### Download the data (skipped if the files are already there)

# %%
TLC = "https://d37ci6vzurychx.cloudfront.net"
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
FILES = {f"{RAW}/fhvhv/fhvhv_tripdata_{m}.parquet": f"{TLC}/trip-data/fhvhv_tripdata_{m}.parquet" for m in MONTHS}
FILES[f"{RAW}/zones/taxi_zone_lookup.csv"] = f"{TLC}/misc/taxi_zone_lookup.csv"
FILES[f"{RAW}/weather/USW00094728.csv"] = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv"
)

for path, url in FILES.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    print(f"{os.path.getsize(path) / 1e6:9.1f} MB  {os.path.relpath(path, ROOT)}")

# %% [markdown]
# ## 2. Load
#
# Parquet stores the schema in each file's footer, so Spark reads no data rows here: it only opens the footers.

# %%
trips_raw = spark.read.parquet(f"{RAW}/fhvhv")

zone_schema = T.StructType([
    T.StructField("LocationID", T.IntegerType()),
    T.StructField("Borough", T.StringType()),
    T.StructField("Zone", T.StringType()),
    T.StructField("service_zone", T.StringType()),
])
zones_raw = spark.read.csv(f"{RAW}/zones", header=True, schema=zone_schema)
weather_raw = spark.read.csv(f"{RAW}/weather", header=True)   # all columns arrive as strings

print("trip columns:", len(trips_raw.columns), "| input partitions:", num_partitions(trips_raw))
trips_raw.printSchema()

# %%
(total_rows, _) = timed("count all raw trips", lambda: trips_raw.count())
print(f"{total_rows:,} raw trips")

# %% [markdown]
# ## 3. Transformations vs actions
#
# A **transformation** (`select`, `filter`, `withColumn`, `join`, `groupBy`...) only adds a step to a plan. An **action** (`count`, `show`, `collect`, `write`...) makes Spark execute the plan.

# %%
start = time.time()
lazy_df = (
    trips_raw
    .filter(F.col("trip_miles") > 20)                       # transformation
    .withColumn("fare_per_mile", F.col("base_passenger_fare") / F.col("trip_miles"))   # transformation
    .groupBy("hvfhs_license_num")                           # transformation
    .agg(F.count("*").alias("long_trips"), F.round(F.avg("fare_per_mile"), 2).alias("avg_fare_per_mile"))
)
print(f"Three transformations on 80M rows defined in {(time.time() - start) * 1000:.0f} ms. Nothing has run.")
print("What we hold is a plan, not data:", lazy_df)

# %%
# The plan exists and can be inspected before any data is read.
lazy_df.explain()

# %%
# An action triggers execution: now Spark scans the files.
rows, _ = timed("action: collect() on the lazy plan", lambda: lazy_df.collect())
for r in rows:
    print(r)

# %%
# Each action re-runs the whole plan from the source files (unless the result is cached).
timed("action again: count()", lambda: lazy_df.count())

# Lazy execution, but eager *analysis*: a wrong column name fails immediately, with no job started.
try:
    trips_raw["no_such_column"]
except Exception as e:
    print("\nFails at definition time:", type(e).__name__, "-", str(e).splitlines()[0][:110])

# %% [markdown]
# ## 4. The pipeline
#
# ### 4.1 Prune columns and filter early
#
# The first two steps of the pipeline are a `select` of the 13 columns we need (out of 24) and the row filters. Doing this before any join or aggregation means every later step handles fewer bytes.
#
# - **Filter 1 (date range):** keep pickups inside the study window.
# - **Filter 2 (data quality):** drop impossible or extreme trips.
# - **Filter 3 (scope):** private rides only (no shared-ride requests), with a known pickup zone.

# %%
NEEDED = [
    "hvfhs_license_num", "request_datetime", "pickup_datetime", "PULocationID", "DOLocationID",
    "trip_miles", "trip_time", "base_passenger_fare", "tips", "driver_pay",
    "congestion_surcharge", "airport_fee", "shared_request_flag",
]

trips_clean = (
    trips_raw
    .select(*NEEDED)
    # Filter 1: date range, written as a plain comparison on the stored column so it can be pushed down
    .filter(
        (F.col("pickup_datetime") >= F.lit("2024-01-01").cast("timestamp_ntz"))
        & (F.col("pickup_datetime") < F.lit("2024-05-01").cast("timestamp_ntz"))
    )
    # Filter 2: data quality
    .filter(
        F.col("trip_miles").between(0.2, 100)
        & F.col("trip_time").between(60, 4 * 3600)
        & F.col("base_passenger_fare").between(1, 500)
        & (F.col("driver_pay") > 0)
    )
    # Filter 3: scope
    .filter((F.col("shared_request_flag") == "N") & (~F.col("PULocationID").isin(264, 265)))
)

(clean_rows, _) = timed("count trips after filters", lambda: trips_clean.count())
print(f"{clean_rows:,} trips kept of {total_rows:,} ({clean_rows / total_rows:.1%}); {total_rows - clean_rows:,} removed")

# %% [markdown]
# ### 4.2 Column transformations with `withColumn`

# %%
trips_feat = (
    trips_clean
    .withColumn("company", F.when(F.col("hvfhs_license_num") == "HV0003", "Uber")
                            .when(F.col("hvfhs_license_num") == "HV0005", "Lyft")
                            .otherwise("Other"))
    .withColumn("pickup_date", F.to_date("pickup_datetime"))
    .withColumn("pickup_month", F.month("pickup_datetime"))
    .withColumn("pickup_hour", F.hour("pickup_datetime"))
    .withColumn("is_weekend", F.dayofweek("pickup_datetime").isin(1, 7))
    .withColumn("wait_min", F.expr("timestampdiff(SECOND, request_datetime, pickup_datetime)") / 60.0)
    .withColumn("wait_min", F.when(F.col("wait_min").between(0, 60), F.col("wait_min")))   # null if implausible
    .withColumn("fare_per_mile", F.col("base_passenger_fare") / F.col("trip_miles"))
    .withColumn("driver_share", F.col("driver_pay") / F.col("base_passenger_fare"))
    .withColumn("tipped", (F.col("tips") > 0).cast("int"))
    .withColumn("speed_mph", F.col("trip_miles") / (F.col("trip_time") / 3600.0))
    .drop("hvfhs_license_num", "shared_request_flag", "request_datetime")
)

# The two small tables. GHCN stores precipitation in tenths of mm and temperature in tenths of a degree C.
weather = (
    weather_raw
    .filter((F.col("DATE") >= "2024-01-01") & (F.col("DATE") < "2024-05-01"))
    .select(
        F.to_date("DATE").alias("pickup_date"),
        (F.col("PRCP").cast("double") / 10).alias("prcp_mm"),
        (F.col("SNOW").cast("double")).alias("snow_mm"),
        (F.col("TMAX").cast("double") / 10).alias("tmax_c"),
        (F.col("TMIN").cast("double") / 10).alias("tmin_c"),
    )
    .withColumn("weather_type",
                F.when(F.col("snow_mm") > 0, "snow")
                 .when(F.col("prcp_mm") >= 10, "heavy rain")
                 .when(F.col("prcp_mm") > 0, "light rain")
                 .otherwise("dry"))
    .withColumn("is_wet", F.col("prcp_mm") > 0)
)

zones = zones_raw.select(
    F.col("LocationID").alias("PULocationID"),
    F.col("Borough").alias("pu_borough"),
    F.col("Zone").alias("pu_zone"),
)

weather.groupBy("weather_type").agg(
    F.count("*").alias("days"),
    F.round(F.avg("prcp_mm"), 1).alias("avg_prcp_mm"),
    F.round(F.avg("tmax_c"), 1).alias("avg_tmax_c"),
).orderBy(F.desc("days")).show()

# %% [markdown]
# ### 4.3 Joins
#
# The fact table has ~75M rows; the two dimension tables have 265 and 121 rows. A `broadcast` hint ships each small table to every task as an in-memory hash map, so the big table is joined where it sits, with **no shuffle**.

# %%
enriched = (
    trips_feat
    .join(F.broadcast(zones), on="PULocationID", how="inner")
    .join(F.broadcast(weather), on="pickup_date", how="inner")
)
spark.catalog.dropTempView("trips_enriched")
enriched.createOrReplaceTempView("trips_enriched")
print(len(enriched.columns), "columns:", ", ".join(enriched.columns))

# %% [markdown]
# ### 4.4 `groupBy` with aggregations: weather type by company

# %%
by_weather = (
    enriched
    .groupBy("weather_type", "company")
    .agg(
        F.countDistinct("pickup_date").alias("days"),
        F.count("*").alias("trips"),
        F.round(F.avg("base_passenger_fare"), 2).alias("avg_fare"),
        F.round(F.avg("fare_per_mile"), 2).alias("fare_per_mile"),
        F.round(F.avg("wait_min"), 2).alias("avg_wait_min"),
        F.round(F.avg("driver_share") * 100, 1).alias("driver_share_pct"),
        F.round(F.avg("tipped") * 100, 1).alias("pct_tipped"),
        F.round(F.avg("speed_mph"), 1).alias("avg_mph"),
    )
    .withColumn("trips_per_day", F.round(F.col("trips") / F.col("days")).cast("long"))
    .orderBy("company", "weather_type")
)
(by_weather_rows, _) = timed("groupBy weather_type x company", lambda: by_weather.collect())
spark.createDataFrame(by_weather_rows).show()

# %% [markdown]
# ## 5. SQL queries
#
# The same DataFrame is registered as the temp view `trips_enriched`. SQL and the DataFrame API compile to the same plans.
#
# ### SQL 1: wet vs dry days, holding weekday/weekend constant
#
# Rainy days are not spread evenly across the week, so the comparison is made inside weekday and weekend groups.

# %%
sql_wet_dry = spark.sql("""
    SELECT company,
           CASE WHEN is_weekend THEN 'weekend' ELSE 'weekday' END        AS day_type,
           CASE WHEN is_wet THEN 'wet' ELSE 'dry' END                    AS weather,
           COUNT(DISTINCT pickup_date)                                   AS days,
           CAST(ROUND(COUNT(*) / COUNT(DISTINCT pickup_date)) AS BIGINT) AS trips_per_day,
           ROUND(AVG(base_passenger_fare), 2)                            AS avg_fare,
           ROUND(AVG(fare_per_mile), 2)                                  AS fare_per_mile,
           ROUND(AVG(wait_min), 2)                                       AS avg_wait_min,
           ROUND(AVG(driver_pay), 2)                                     AS avg_driver_pay,
           ROUND(AVG(driver_share) * 100, 1)                             AS driver_share_pct
    FROM trips_enriched
    GROUP BY 1, 2, 3
    ORDER BY company, day_type, weather
""")
(sql1_rows, _) = timed("SQL 1: wet vs dry by company and day type", lambda: sql_wet_dry.collect())
spark.createDataFrame(sql1_rows).show()

# %% [markdown]
# ### SQL 2: which pickup zones gain the most demand when it rains? (CTEs + window function)
#
# For each zone: average trips on wet days vs dry days, then rank zones inside each borough by the uplift. Only busy zones (500+ trips on a dry day) are ranked.

# %%
sql_zone_uplift = spark.sql("""
    WITH zone_day AS (
        SELECT pu_borough, pu_zone, pickup_date, is_wet, COUNT(*) AS trips
        FROM trips_enriched
        WHERE NOT is_weekend
        GROUP BY pu_borough, pu_zone, pickup_date, is_wet
    ),
    zone_avg AS (
        SELECT pu_borough, pu_zone,
               AVG(CASE WHEN is_wet THEN trips END)     AS wet_trips,
               AVG(CASE WHEN NOT is_wet THEN trips END) AS dry_trips
        FROM zone_day
        GROUP BY pu_borough, pu_zone
        HAVING AVG(CASE WHEN NOT is_wet THEN trips END) >= 500
    ),
    ranked AS (
        SELECT *,
               (wet_trips / dry_trips - 1) * 100 AS uplift_pct,
               ROW_NUMBER() OVER (PARTITION BY pu_borough ORDER BY wet_trips / dry_trips DESC) AS rnk
        FROM zone_avg
    )
    SELECT pu_borough, rnk, pu_zone,
           CAST(ROUND(dry_trips) AS INT) AS dry_day_trips,
           CAST(ROUND(wet_trips) AS INT) AS wet_day_trips,
           ROUND(uplift_pct, 1)          AS uplift_pct
    FROM ranked
    WHERE rnk <= 3
    ORDER BY pu_borough, rnk
""")
(sql2_rows, _) = timed("SQL 2: top rain-uplift zones per borough", lambda: sql_zone_uplift.collect())
spark.createDataFrame(sql2_rows).show(truncate=False)

# %% [markdown]
# ### SQL 3: does rain change the daily rhythm? Wait time and price by hour on weekdays

# %%
sql_hourly = spark.sql("""
    SELECT pickup_hour,
           ROUND(AVG(CASE WHEN NOT is_wet THEN wait_min END), 2)      AS wait_dry,
           ROUND(AVG(CASE WHEN is_wet THEN wait_min END), 2)          AS wait_wet,
           ROUND(AVG(CASE WHEN NOT is_wet THEN fare_per_mile END), 2) AS fare_mi_dry,
           ROUND(AVG(CASE WHEN is_wet THEN fare_per_mile END), 2)     AS fare_mi_wet
    FROM trips_enriched
    WHERE NOT is_weekend
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""")
(sql3_rows, _) = timed("SQL 3: hourly wait and price, wet vs dry", lambda: sql_hourly.collect())
spark.createDataFrame(sql3_rows).show(24)

# %% [markdown]
# ## 6. Write results
#
# The enriched trips are written as Parquet **partitioned by `pickup_month`**: one folder per month. Queries that filter on the month then read only the matching folders (partition pruning).
#
# Month is a low-cardinality key (4 values) that matches how the input files are split, so each write task holds rows for one or two months and no shuffle is needed before the write. Partitioning by date instead (121 values) would need a shuffle (`repartition("pickup_date")`) to avoid hundreds of small files.

# %%
OUTPUT_COLS = [
    "pickup_datetime", "pickup_date", "pickup_hour", "is_weekend", "company",
    "PULocationID", "DOLocationID", "pu_borough", "pu_zone",
    "trip_miles", "trip_time", "base_passenger_fare", "tips", "driver_pay",
    "wait_min", "fare_per_mile", "driver_share", "speed_mph",
    "weather_type", "is_wet", "prcp_mm", "tmax_c", "tmin_c", "pickup_month",
]

timed("write enriched trips, partitioned by month", lambda: (
    enriched.select(*OUTPUT_COLS)
    .write.mode("overwrite")
    .partitionBy("pickup_month")
    .parquet(f"{OUT}/trips_enriched")
))

# Small result tables: one file each.
for name, rows in [("by_weather_company", by_weather_rows), ("wet_vs_dry", sql1_rows),
                   ("zone_rain_uplift", sql2_rows), ("hourly_wet_vs_dry", sql3_rows)]:
    spark.createDataFrame(rows).coalesce(1).write.mode("overwrite").parquet(f"{OUT}/{name}")

for folder in sorted(os.listdir(f"{OUT}/trips_enriched")):
    full = f"{OUT}/trips_enriched/{folder}"
    if os.path.isdir(full):
        files = [f for f in os.listdir(full) if f.endswith(".parquet")]
        size = sum(os.path.getsize(f"{full}/{f}") for f in files)
        print(f"{folder:16s} {len(files):3d} files  {size / 1e6:8.1f} MB")

# %% [markdown]
# ### Reading it back: partition pruning
#
# The filter on `pickup_month` appears under `PartitionFilters`, so Spark opens only the `pickup_month=2` folder.

# %%
saved = spark.read.parquet(f"{OUT}/trips_enriched")
feb = saved.filter(F.col("pickup_month") == 2).groupBy("company").agg(F.count("*").alias("trips"))
feb.explain()
timed("February trips from the partitioned output", lambda: feb.show())
(saved_rows, _) = timed("count all rows in the output", lambda: saved.count())
assert saved_rows == clean_rows, (saved_rows, clean_rows)
print(f"Output row count matches the cleaned input: {saved_rows:,}")

# %% [markdown]
# ## 7. Performance analysis
#
# ### 7.1 The physical plan of the main aggregation
#
# Read it bottom-up. Things to look for:
#
# - `Scan parquet`: `ReadSchema` lists only the columns the query uses (column pruning) and `PushedFilters` lists the filters handed to the Parquet reader (predicate pushdown).
# - `BroadcastHashJoin` + `BroadcastExchange`: the small tables are broadcast, so the join adds no shuffle of the trip rows.
# - `HashAggregate` appears twice around an `Exchange hashpartitioning`: a partial aggregation inside each input partition, one shuffle of the already-reduced rows, then the final aggregation.
# - `AdaptiveSparkPlan`: Adaptive Query Execution re-plans at run time. Because this query has already run, the output shows the **Final Plan** (with `AQEShuffleRead ... coalesced`, where 24 shuffle partitions were merged into one) above the **Initial Plan**.
# - `count(distinct pickup_date)` costs an extra aggregation level: Spark first groups by (weather, company, date), then by (weather, company), with a shuffle for each.

# %%
by_weather.explain(mode="formatted")

# %% [markdown]
# ### 7.2 Does it matter where I write the filter?
#
# Below, the same query is written two ways: filters first (as in the pipeline) and filters last (after the joins). Catalyst's optimizer moves filters as close to the data source as it can, so both produce the same optimized plan.

# %%
def scan_lines(df):
    """The Parquet scan nodes of a physical plan, where pushdown is visible."""
    plan = df._jdf.queryExecution().executedPlan().toString()
    return [line.strip() for line in plan.splitlines() if "Scan parquet" in line or "FileScan parquet" in line]


quality = F.col("trip_miles").between(0.2, 100) & (F.col("driver_pay") > 0)
agg = [F.count("*").alias("trips"), F.avg("base_passenger_fare").alias("avg_fare")]

filter_early = (
    trips_raw.filter(quality)
    .join(F.broadcast(zones), "PULocationID")
    .groupBy("pu_borough").agg(*agg)
)
filter_late = (
    trips_raw
    .join(F.broadcast(zones), "PULocationID")
    .filter(quality)
    .groupBy("pu_borough").agg(*agg)
)

print("FILTER EARLY:", *scan_lines(filter_early), sep="\n  ")
print("\nFILTER LATE: ", *scan_lines(filter_late), sep="\n  ")
print("\nIdentical Parquet scans:", scan_lines(filter_early) == scan_lines(filter_late))

# %% [markdown]
# ### 7.3 When a pushed filter does and does not save work
#
# `PushedFilters` in the plan only means Spark *handed* the filter to the Parquet reader. The reader can skip data only if the file footer holds min/max statistics for each row group, and only if the filter is a plain comparison on a stored column.
#
# First, do the files have statistics? The raw TLC files do not; the files this pipeline wrote do.

# %%
import glob
import json
import urllib.request as request

import pyarrow.parquet as pq


def footer_stats(folder, column="pickup_datetime"):
    """Row groups in the first file of a folder, and how many carry min/max statistics for a column."""
    path = sorted(glob.glob(f"{folder}/**/*.parquet", recursive=True))[0]
    meta = pq.ParquetFile(path).metadata
    idx = [meta.row_group(0).column(i).path_in_schema for i in range(meta.num_columns)].index(column)
    with_stats = [g for g in range(meta.num_row_groups)
                  if meta.row_group(g).column(idx).statistics is not None
                  and meta.row_group(g).column(idx).statistics.has_min_max]
    first = meta.row_group(0).column(idx).statistics
    span = f"first group spans {first.min} .. {first.max}" if with_stats else "no min/max recorded"
    return f"{meta.num_row_groups} row groups, {len(with_stats)} with statistics ({span})"


print("raw TLC file:     ", footer_stats(f"{RAW}/fhvhv"))
print("pipeline output:  ", footer_stats(f"{OUT}/trips_enriched"))


# %% [markdown]
# Now the same two-day question (trips on 20-21 February) asked four ways. "Rows read" is the Parquet scan's output-row metric from the Spark UI, the same number the Query Details view shows on the edge under `Scan parquet`. It is the honest test of whether a filter saved any reading.

# %%
def scan_rows_read(label):
    """Output rows of the Parquet scan for the latest query with this label (local Spark UI only)."""
    try:
        time.sleep(1)                                    # let the UI finish recording metrics
        sc = spark.sparkContext
        url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/sql?details=true&length=1000"
        runs = [q for q in json.load(request.urlopen(url)) if q["description"] == label]
        scan = [n for n in max(runs, key=lambda q: q["id"])["nodes"] if n["nodeName"].startswith("Scan parquet")][0]
        return [m["value"] for m in scan["metrics"] if m["name"] == "number of output rows"][0]
    except Exception:
        return "n/a"


import re

lo = F.lit("2024-02-20").cast("timestamp_ntz")
hi = F.lit("2024-02-22").cast("timestamp_ntz")
ts_range = (F.col("pickup_datetime") >= lo) & (F.col("pickup_datetime") < hi)
count_trips = lambda df: df.agg(F.count("*").alias("trips"))

cases = {
    "A. raw files, timestamp range": count_trips(trips_raw.filter(ts_range)),
    "B. output files, timestamp range": count_trips(saved.filter(ts_range)),
    "C. output files, DATE column range": count_trips(saved.filter(F.col("pickup_date").between("2024-02-20", "2024-02-21"))),
    "D. output files, to_date() function": count_trips(saved.filter(F.to_date("pickup_datetime").isin("2024-02-20", "2024-02-21"))),
}
for label, df in cases.items():
    pushed = re.search(r"PushedFilters: (.*?), ReadSchema", scan_lines(df)[0]).group(1)
    (row, _) = timed(label, lambda df=df: df.first())
    print(f"           PushedFilters: {pushed}")
    print(f"           rows read by the scan: {scan_rows_read(label)} | rows matching: {row.trips:,}\n")

# %% [markdown]
# What this shows:
#
# - **A:** the filter is listed as pushed, but the raw files have no statistics, so the reader cannot skip anything.
# - **B:** the output files have statistics, yet in this Spark version the `timestamp_ntz` comparison still did not skip any row group. Being listed in `PushedFilters` is not proof of skipping.
# - **C:** the same question on the stored `pickup_date` column (type DATE) lets the reader skip most row groups. This is why the pipeline stores `pickup_date` instead of leaving it to be derived.
# - **D:** wrapping the column in a function leaves `PushedFilters` empty, so every row is read and then filtered.

# %% [markdown]
# ### 7.4 Join strategy: broadcast vs sort-merge
#
# With broadcasting switched off, Spark falls back to a sort-merge join: both sides are shuffled by the join key and sorted. For a 75M-row table that is the most expensive thing in the plan.

# %%
join_query = lambda z: trips_clean.join(z, "PULocationID").groupBy("pu_borough").agg(F.count("*").alias("trips"))

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")     # disable automatic broadcasting
smj = join_query(zones)
print("WITHOUT BROADCAST")
smj.explain()
(_, t_smj) = timed("join + aggregate with sort-merge join", lambda: smj.collect())

spark.conf.unset("spark.sql.autoBroadcastJoinThreshold")        # back to the default
bhj = join_query(F.broadcast(zones))
print("\nWITH BROADCAST")
bhj.explain()
(_, t_bhj) = timed("join + aggregate with broadcast hash join", lambda: bhj.collect())
print(f"\nBroadcast join is {t_smj / t_bhj:.1f}x faster")

# %% [markdown]
# ## 8. Caching
#
# Caching pays off when an expensive intermediate result is **reused by several actions** and is **small enough to fit in memory**. The 75M-row enriched table is not a good candidate on a 3 GB driver. A good candidate is an hourly summary: it takes a full scan to compute, has only a few tens of thousands of rows, and several analyses (and the ML model below) start from it.

# %%
hourly = (
    enriched
    .groupBy("pickup_date", "pickup_hour", "pu_borough", "company",
             "is_weekend", "weather_type", "is_wet", "prcp_mm", "tmax_c", "tmin_c")
    .agg(
        F.count("*").alias("trips"),
        F.sum("base_passenger_fare").alias("fare_sum"),
        F.sum("trip_miles").alias("miles_sum"),
        F.sum("driver_pay").alias("driver_pay_sum"),
    )
)

# Three different questions that all start from the hourly summary.
followups = {
    "busiest hours": lambda df: df.groupBy("pickup_hour").agg(F.sum("trips").alias("trips")).orderBy(F.desc("trips")).limit(3).collect(),
    "revenue by borough and weather": lambda df: df.groupBy("pu_borough", "weather_type").agg(F.sum("fare_sum")).collect(),
    "daily totals by company": lambda df: df.groupBy("pickup_date", "company").agg(F.sum("trips")).collect(),
}

print("WITHOUT cache: every action recomputes the summary from 80M rows")
t_nocache = sum(timed(f"no cache: {name}", lambda q=q: q(hourly))[1] for name, q in followups.items())

print("\nWITH cache: the first action computes and stores it, the rest read from memory")
hourly.cache()                                                   # lazy: marks the DataFrame, stores nothing yet
(hourly_rows, t_fill) = timed("cache fill: count()", lambda: hourly.count())
t_cached = sum(timed(f"cached:   {name}", lambda q=q: q(hourly))[1] for name, q in followups.items())

print(f"\nhourly summary rows: {hourly_rows:,}")
print(f"3 queries without cache: {t_nocache:6.2f}s")
print(f"cache fill + 3 queries:  {t_fill + t_cached:6.2f}s  (fill {t_fill:.2f}s, queries {t_cached:.2f}s)")
print(f"speed-up for the 3 queries once cached: {t_nocache / t_cached:.0f}x")

# %%
# The plan now starts from memory (InMemoryTableScan) instead of the Parquet files.
hourly.groupBy("pickup_hour").agg(F.sum("trips")).explain()

# %% [markdown]
# ## 9. MLlib: predicting hourly demand
#
# **Task:** predict the number of trips per borough per hour. **Question:** do weather features improve the prediction beyond the calendar (hour, weekend, borough)?
#
# Whole days are held out for testing (a random 20% of dates), so the model is never tested on an hour from a day it has seen.

# %%
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor

demand = (
    hourly
    .filter(F.col("pu_borough").isin("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"))
    .groupBy("pickup_date", "pickup_hour", "pu_borough", "is_weekend", "prcp_mm", "tmax_c", "tmin_c")
    .agg(F.sum("trips").cast("double").alias("trips"))
    .withColumn("is_weekend", F.col("is_weekend").cast("int"))
    .withColumn("day_of_week", F.dayofweek("pickup_date"))
)

test_dates = [r.pickup_date for r in demand.select("pickup_date").distinct().orderBy(F.hash("pickup_date")).limit(24).collect()]
train = demand.filter(~F.col("pickup_date").isin(test_dates)).cache()
test = demand.filter(F.col("pickup_date").isin(test_dates)).cache()
print(f"train rows: {train.count():,} | test rows: {test.count():,} | test days: {len(test_dates)}")


def fit_and_score(numeric_features, label):
    stages = [
        StringIndexer(inputCol="pu_borough", outputCol="borough_idx"),
        OneHotEncoder(inputCol="borough_idx", outputCol="borough_vec"),
        VectorAssembler(inputCols=["borough_vec"] + numeric_features, outputCol="features"),
        GBTRegressor(featuresCol="features", labelCol="trips", maxDepth=6, maxIter=60, seed=42),
    ]
    model = Pipeline(stages=stages).fit(train)
    pred = model.transform(test)
    rmse = RegressionEvaluator(labelCol="trips", metricName="rmse").evaluate(pred)
    r2 = RegressionEvaluator(labelCol="trips", metricName="r2").evaluate(pred)
    mae = RegressionEvaluator(labelCol="trips", metricName="mae").evaluate(pred)
    print(f"{label:28s} RMSE {rmse:8.1f} | MAE {mae:8.1f} | R2 {r2:.4f}")
    return model, pred


describe("MLlib: train and evaluate")
calendar = ["pickup_hour", "is_weekend", "day_of_week"]
mean_trips = test.agg(F.avg("trips")).first()[0]
print(f"mean trips per borough-hour in the test set: {mean_trips:,.0f}\n")
_, _ = fit_and_score(calendar, "calendar only")
model, pred = fit_and_score(calendar + ["prcp_mm", "tmax_c", "tmin_c"], "calendar + weather")

# %%
pred.select("pickup_date", "pickup_hour", "pu_borough", "prcp_mm", "tmax_c",
            F.col("trips").cast("int").alias("actual"), F.round("prediction").cast("int").alias("predicted")) \
    .orderBy(F.rand(seed=7)).show(10)

gbt = model.stages[-1]
feature_names = [f"borough_{i}" for i in range(4)] + calendar + ["prcp_mm", "tmax_c", "tmin_c"]
for name, score in sorted(zip(feature_names, gbt.featureImportances.toArray()), key=lambda x: -x[1]):
    print(f"{name:14s} {score:.3f}")

# %%
train.unpersist()
test.unpersist()
hourly.unpersist()
print("Pipeline finished successfully.")
