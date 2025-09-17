from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, MapType
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
    if n == 0: return {"percent_score": 0.0, "wilson_score": 0.0}
    percent_score = pos / n
    wilson_score = cal_wilson_score(pos, n)
    return {"percent_score": percent_score, "wilson_score": wilson_score}

def proc_merge_rslt(msgs):
    rslt = []
    for msg in msgs:
        value = msg.value
        value["name"] = msg.name
        rslt.append(value)
    return json.dumps(rslt, ensure_ascii=False)

def process_compute_perference(df, perf_name, cal_perf_score_udf, proc_merge_rslt_udf):
    df = df.filter(F.col("address").isNotNull())
    df = df.groupBy("user_id", "address").agg(F.sum("weight").alias("pos"))
    df_total = df.groupBy("user_id").agg(F.sum("pos").alias("n"))
    df_rslt = df.join(df_total, ["user_id"])
    df_rslt = df_rslt.withColumn("perf_addr", cal_perf_score_udf("pos", "n"))
    df_rslt = df_rslt.groupBy("user_id") \
        .agg(F.collect_list(F.struct(F.col("address").alias("name"), F.col("perf_addr").alias("value"))).alias("msgs"))
    df_rslt = df_rslt.withColumn(perf_name, proc_merge_rslt_udf(F.col("msgs")))
    return df_rslt

def main(current_date):
    file_input = f"/data/dw_label_di/"
    file_output = f"/data/up_perf_label_da/partition_date={current_date}/"
    print(file_input)
    print(file_output)

    spark = SparkSession.builder.appName("AddressPreference").getOrCreate()

    cal_perf_score_udf = F.udf(cal_perf_score, MapType(StringType(), FloatType()))
    proc_merge_rslt_udf = F.udf(proc_merge_rslt, StringType())
    
    # df = spark.read.csv("file:///data/yuchi/common_memory/output_table/*.csv", header=True, inferSchema=True)
    df = spark.read.csv(file_input, header=True, inferSchema=True)

    current_date_col = F.to_date(F.lit(current_date))
    df = df.withColumn("create_date", F.to_date("create_date")) \
           .withColumn("days", F.datediff(current_date_col, "create_date")) \
           .withColumn("decay", F.pow(0.96, F.col("days") / 90)) \
           .withColumn("decay", F.when(F.col("decay") < 0, 0).otherwise(F.col("decay"))) \
           .withColumn("intent_id", F.col("intent_id").cast("string"))
    df = df.withColumn("w", F.when((F.col("intent_id") == "10007"), 1.0)
                             .when(F.col("intent_id") == "10012", 0.1)
                             .when((F.col("intent_id") == "10008"), 0.04)
                             .when(F.col("intent_id") == "10009", 0.02)
                             .when(F.col("intent_id") == "10004", 0.01)
                             .otherwise(0.0))
    df = df.withColumn("weight", F.col("w") * F.col("decay"))
    df = df.cache()

    df_rslt_home = process_compute_perference(df.select("user_id", "weight", F.col("address_home").alias("address")), "perf_addr_home", cal_perf_score_udf, proc_merge_rslt_udf)
    df_rslt_comp = process_compute_perference(df.select("user_id", "weight", F.col("address_company").alias("address")), "perf_addr_company", cal_perf_score_udf, proc_merge_rslt_udf)
    df_user = df.dropDuplicates(["user_id"])

    result_df = df_user.join(df_rslt_home, "user_id", "left").join(df_rslt_comp, "user_id", "left") \
                       .select("user_id", "perf_addr_home", "perf_addr_company")

    result_df.write.mode("overwrite").csv(file_output, header=True)
    spark.stop()

if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_date = '2025-09-06'
    main(current_date)
    