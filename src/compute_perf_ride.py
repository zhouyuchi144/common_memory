from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import math
import json
import sys


def cal_wilson_score(pos, n):
    if n == 0: return 0.0
    p = pos / n
    z = 1.96
    denominator = 1 + z**2 / n
    numerator = p + z**2 / (2 * n) - z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return numerator / denominator

def cal_perf_score(pos, n):
    if n == 0: return json.dumps({"percent_score": 0.0, "wilson_score": 0.0}, ensure_ascii=False)
    percent_score = pos / n
    wilson_score = cal_wilson_score(pos, n)
    return json.dumps({"percent_score": percent_score, "wilson_score": wilson_score}, ensure_ascii=False)

def main(current_date):
    file_input = f"/data/dw_ride_di/"
    file_output = f"/data/up_perf_ride_da/partition_date={current_date}/"
    print(file_input)
    print(file_output)

    spark = SparkSession.builder.appName("AddressPreference").getOrCreate()

    cal_perf_score_udf = F.udf(cal_perf_score, StringType())
    
    # df = spark.read.csv("file:///data/yuchi/common_memory/output_table/*.csv", header=True, inferSchema=True)
    df = spark.read.csv(file_input, header=True, inferSchema=True)

    current_date_col = F.to_date(F.lit(current_date))
    df = df.withColumn("create_date", F.to_date("update_time")) \
           .withColumn("days", F.datediff(current_date_col, "create_date")) \
           .withColumn("decay", F.pow(0.5, F.col("days") / 90)) \
           .withColumn("decay", F.when(F.col("decay") < 0, 0).otherwise(F.col("decay")))
    df = df.withColumn("w", F.when((F.col("order_status") == 20) | (F.col("order_status") == 99), 1.0)
                             .when(F.col("order_status") == 0, 0.8)
                             .otherwise(0.0))
    df = df.withColumn("weight", F.col("w") * F.col("decay"))
    df = df.cache()

    df_from = df.groupBy("user_id", "from_address").agg(F.sum("weight").alias("pos_from"))
    df_to = df.groupBy("user_id", "to_address").agg(F.sum("weight").alias("pos_to"))
    df_user = df.groupBy("user_id").agg(F.sum("weight").alias("n_total")).cache()

    df_rslt_from = df_from.join(df_user, ["user_id"])
    df_rslt_to = df_to.join(df_user, ["user_id"])
    df_rslt_from = df_rslt_from.withColumn("perf_addr_from", cal_perf_score_udf("pos_from", "n_total")) \
                               .select("user_id", "perf_addr_from")
    df_rslt_to = df_rslt_to.withColumn("perf_addr_to", cal_perf_score_udf("pos_to", "n_total")) \
                           .select("user_id", "perf_addr_to")

    df1 = df_from.select("user_id", F.col("from_address").alias("address"), F.col("pos_from").alias("pos_addr"))
    df2 = df_to.select("user_id", F.col("to_address").alias("address"), F.col("pos_to").alias("pos_addr"))
    df0 = df1.unionByName(df2).cache()

    df_addr = df0.groupBy("user_id", "address").agg(F.sum("pos_addr").alias("pos_addr"))
    df_user = df0.groupBy("user_id").agg(F.sum("pos_addr").alias("n_total"))

    df_rslt_addr = df_addr.join(df_user, ["user_id"])
    df_rslt_addr = df_rslt_addr.withColumn("perf_addr", cal_perf_score_udf("pos_addr", "n_total")) \
                               .select("user_id", "perf_addr")

    result_df = df_rslt_addr.join(df_rslt_from, "user_id", "left").join(df_rslt_to, "user_id", "left") \
                            .select("user_id", "perf_addr", "perf_addr_from", "perf_addr_to")

    result_df.write.mode("overwrite").csv(file_output, header=True)
    spark.stop()

if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_date = '2025-09-06'
    main(current_date)
    