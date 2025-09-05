from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, MapType, StringType
import math
import sys



def cal_wilson_score(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    z = 1.96
    denominator = 1 + z**2 / n
    numerator = p + z**2 / (2 * n) - z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return numerator / denominator


def main(current_time):
    spark = SparkSession.builder.appName("AddressPreference").getOrCreate()
    
    df = spark.read.csv("file:///data/yuchi/common_memory/output_table/*.csv", header=True, inferSchema=True)
    
    df = df.withColumn("create_date", F.to_date("update_time"))
    df = df.withColumn("current_time", F.lit(current_time))
    df = df.withColumn("current_date", F.to_date("current_time"))
    df = df.withColumn("days", F.datediff("current_date", "create_date"))
    df = df.withColumn("decay", F.pow(0.5, F.col("days") / 90))
    df = df.withColumn("decay", F.when(F.col("decay") < 0, 0).otherwise(F.col("decay")))
    
    df = df.withColumn("w", F.when((F.col("order_status") == 20) | (F.col("order_status") == 99), 1.0)
                             .when(F.col("order_status") == 0, 0.8)
                             .otherwise(0.0))
    df = df.withColumn("weight", F.col("w") * F.col("decay"))
    

    wilson_udf = F.udf(cal_wilson_score, FloatType())


    df_from = df.groupBy("user_id", "from_address").agg(F.sum("weight").alias("pos_from"))
    df_to = df.groupBy("user_id", "to_address").agg(F.sum("weight").alias("pos_to"))
    df_user = df.groupBy("user_id").agg(F.sum("weight").alias("n_total"))

    df_rslt_from = df_from.join(df_user, ["user_id"]) \
                   .withColumn("ratio_score", F.col("pos_from") / F.col("n_total"))
    df_rslt_to = df_to.join(df_user, ["user_id"]) \
                 .withColumn("ratio_score", F.col("pos_to") / F.col("n_total"))
    df_rslt_from = df_rslt_from.withColumn("wilson_score", wilson_udf("pos_from", "n_total"))
    df_rslt_to = df_rslt_to.withColumn("wilson_score", wilson_udf("pos_to", "n_total"))
    df_rslt_from = df_rslt_from.groupBy("user_id").agg(
        F.map_from_entries(F.collect_list(F.struct("from_address", "ratio_score"))).alias("preference_from_ratio"),
        F.map_from_entries(F.collect_list(F.struct("from_address", "wilson_score"))).alias("preference_from_wilson")
    )
    df_rslt_to = df_rslt_to.groupBy("user_id").agg(
        F.map_from_entries(F.collect_list(F.struct("to_address", "ratio_score"))).alias("preference_to_ratio"),
        F.map_from_entries(F.collect_list(F.struct("to_address", "wilson_score"))).alias("preference_to_wilson")
    )


    df1 = df_from.select("user_id", F.col("from_address").alias("address"), F.col("pos_from").alias("pos_address"))
    df2 = df_to.select("user_id", F.col("to_address").alias("address"), F.col("pos_to").alias("pos_address"))
    df_address0 = df1.union(df2)

    df_address = df_address0.groupBy("user_id", "address").agg(F.sum("pos_address").alias("pos_address"))
    df_user = df_address0.groupBy("user_id").agg(F.sum("pos_address").alias("n_total"))

    df_rslt_address = df_address.join(df_user, ["user_id"]) \
                      .withColumn("ratio_score", F.col("pos_address") / F.col("n_total"))
    df_rslt_address = df_rslt_address.withColumn("wilson_score", wilson_udf("pos_address", "n_total"))

    df_rslt_address = df_rslt_address.groupBy("user_id").agg(
        F.map_from_entries(F.collect_list(F.struct("address", "ratio_score"))).alias("preference_address_ratio"),
        F.map_from_entries(F.collect_list(F.struct("address", "wilson_score"))).alias("preference_address_wilson")
    )


    result_df = df_rslt_address.join(df_rslt_from, "user_id", "left").join(df_rslt_to, "user_id", "left").select(
        "user_id",
        F.coalesce(df_rslt_address["preference_address_ratio"], F.create_map()).alias("preference_address_ratio"),
        F.coalesce(df_rslt_address["preference_address_wilson"], F.create_map()).alias("preference_address_wilson"),
        F.coalesce(df_rslt_from["preference_from_ratio"], F.create_map()).alias("preference_from_ratio"),
        F.coalesce(df_rslt_from["preference_from_wilson"], F.create_map()).alias("preference_from_wilson"),
        F.coalesce(df_rslt_to["preference_to_ratio"], F.create_map()).alias("preference_to_ratio"),
        F.coalesce(df_rslt_to["preference_to_wilson"], F.create_map()).alias("preference_to_wilson")
    )
    result_df = result_df.withColumn("preference_address_ratio", F.to_json("preference_address_ratio")) \
                         .withColumn("preference_address_wilson", F.to_json("preference_address_wilson")) \
                         .withColumn("preference_from_ratio", F.to_json("preference_from_ratio")) \
                         .withColumn("preference_from_wilson", F.to_json("preference_from_wilson")) \
                         .withColumn("preference_to_ratio", F.to_json("preference_to_ratio")) \
                         .withColumn("preference_to_wilson", F.to_json("preference_to_wilson"))


    result_df.write.mode("overwrite").csv("file:///data/yuchi/common_memory/output_perf", header=True)
    
    spark.stop()

if __name__ == "__main__":
    # current_time = sys.argv[1]
    current_time = '2025-09-05 17:00:00'
    main(current_time)
    