from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, lit
from pyspark.sql.types import StructType, StructField, StringType
import json
import sys

# 定义 UDF 解析参数文件的 parameters 列
def parse_parameters(parameters_str):
    try:
        data_list = json.loads(parameters_str)
    except:
        return (None, None, None, None)

    start_place_confirmed = False
    end_place_confirmed = False
    from_address = None
    from_city_code = None
    from_coordinate = None
    to_address = None
    to_city_code = None
    to_coordinate = None

    for item in data_list:
        if isinstance(item, dict):
            if item.get("propertyName") == "start_place" and item.get("status") == "CONFIRMED":
                start_place_confirmed = True
                property_value = item.get("propertyValue", "{}")
                try:
                    property_value_dict = json.loads(property_value)
                    from_address = property_value_dict.get("name", "")
                    from_city_code = property_value_dict.get("cityCode", "")
                    from_coordinate = property_value_dict.get("coordinate", {})
                    latitude = property_value_dict.get("coordinate", {}).get("latitude")
                    longitude = property_value_dict.get("coordinate", {}).get("longitude")
                    if latitude and longitude: from_coordinate = f"{latitude},{longitude}"
                except:
                    pass
            elif item.get("propertyName") == "end_place" and item.get("status") == "CONFIRMED":
                end_place_confirmed = True
                property_value = item.get("propertyValue", "{}")
                try:
                    property_value_dict = json.loads(property_value)
                    to_address = property_value_dict.get("name", "")
                    to_city_code = property_value_dict.get("cityCode", "")
                    latitude = property_value_dict.get("coordinate").get("latitude")
                    longitude = property_value_dict.get("coordinate").get("longitude")
                    if latitude and longitude: to_coordinate = f"{latitude},{longitude}"
                except:
                    pass

    if start_place_confirmed and end_place_confirmed:
        return (from_address, from_city_code, from_coordinate, to_address, to_city_code, to_coordinate)
    else:
        return (None, None, None, None, None, None)

# 定义 UDF 解析订单文件的地址列
def parse_order_address(address_str):
    try:
        data = json.loads(address_str)
        latitude = data.get("coordinate").get("latitude")
        longitude = data.get("coordinate").get("longitude")
        coordinate = None
        if latitude and longitude: coordinate = f"{latitude},{longitude}"
        return (data.get("address", None), data.get("cityCode", None), coordinate)
    except:
        return (None, None)


def process_parameters(df_parameters):
    """处理参数文件数据并提取地址信息"""
    param_schema = StructType([
        StructField("from_address", StringType(), True),
        StructField("from_city_code", StringType(), True),
        StructField("from_coordinate", StringType(), True),
        StructField("to_address", StringType(), True),
        StructField("to_city_code", StringType(), True),
        StructField("to_coordinate", StringType(), True)
    ])
    parse_params_udf = udf(parse_parameters, param_schema)

    return df_parameters \
        .withColumn("parsed_params", parse_params_udf(col("parameters"))) \
        .select(
            col("instance_id"),
            col("user_id").alias("param_user_id"),
            col("parsed_params.from_address").alias("param_from_address"),
            col("parsed_params.from_city_code").alias("param_from_city_code"),
            col("parsed_params.from_coordinate").alias("param_from_coordinate"),
            col("parsed_params.to_address").alias("param_to_address"),
            col("parsed_params.to_city_code").alias("param_to_city_code"),
            col("parsed_params.to_coordinate").alias("param_to_coordinate"),
            col("updated_time").alias("param_update_time")
         ) \
        .filter(col("param_from_address").isNotNull() & col("param_to_address").isNotNull())

def process_orders(df_order):
    """处理订单文件数据并提取地址信息"""
    addr_schema = StructType([
        StructField("address", StringType(), True),
        StructField("city_code", StringType(), True),
        StructField("coordinate", StringType(), True)
    ])
    parse_addr_udf = udf(parse_order_address, addr_schema)

    return df_order \
        .withColumn("parsed_from", parse_addr_udf(col("from_address"))) \
        .withColumn("parsed_to", parse_addr_udf(col("to_address"))) \
        .select(
        col("wf_instance_id"),
        col("user_id").alias("order_user_id"),
        col("parsed_from.address").alias("order_from_address"),
        col("parsed_from.city_code").alias("order_from_city_code"),
        col("parsed_from.coordinate").alias("order_from_coordinate"),
        col("parsed_to.address").alias("order_to_address"),
        col("parsed_to.city_code").alias("order_to_city_code"),
        col("parsed_to.coordinate").alias("order_to_coordinate"),
        col("updated_time").alias("order_update_time"),
        col("order_status")
    )

def main(current_date):
    # 初始化 SparkSession
    spark = SparkSession.builder.appName("TaxiDataProcess").getOrCreate()
    # 读取参数文件
    # df_parameters = spark.read.csv("file:///data/yuchi/common_memory/data/ride_parameters.csv", header=True, escape='"')
    # df_order = spark.read.csv("file:///data/yuchi/common_memory/data/ride_hailing_order.csv", header=True, escape='"')
    file_ride_param = f"/data/ride_parameters/partition_date={current_date}/"
    file_ride_order = f"/data/ride_hailing_order/partition_date={current_date}/"
    file_dw_ride = f"/data/dw_ride_di/partition_date={current_date}/"
    print(file_ride_param)
    print(file_ride_order)
    print(file_dw_ride)
    df_parameters = spark.read.csv(file_ride_param, header=True)
    df_parameters_processed = process_parameters(df_parameters)

    try:
        df_order = spark.read.csv(file_ride_order, header=True)
        # 订单文件非空，处理并关联数据
        df_order_processed = process_orders(df_order)

        # 左连接两个数据集
        df_joined = df_parameters_processed.join(
            df_order_processed,
            df_parameters_processed.instance_id == df_order_processed.wf_instance_id,
            "left"
        )

        # 构建结果数据集
        df_output = df_joined.withColumn(
            "instance_id",
            when(col("wf_instance_id").isNotNull(), col("wf_instance_id")).otherwise(col("instance_id"))
        ).withColumn(
            "user_id",
            when(col("wf_instance_id").isNotNull(), col("order_user_id")).otherwise(col("param_user_id"))
        ).withColumn(
            "order_status",
            when(col("wf_instance_id").isNotNull(), col("order_status")).otherwise(lit(0))
        ).withColumn(
            "from_address",
            when(col("wf_instance_id").isNotNull(), col("order_from_address")).otherwise(col("param_from_address"))
        ).withColumn(
            "from_city_code",
            when(col("wf_instance_id").isNotNull(), col("order_from_city_code")).otherwise(col("param_from_city_code"))
        ).withColumn(
            "from_coordinate",
            when(col("wf_instance_id").isNotNull(), col("order_from_coordinate")).otherwise(col("param_from_coordinate"))
        ).withColumn(
            "to_address",
            when(col("wf_instance_id").isNotNull(), col("order_to_address")).otherwise(col("param_to_address"))
        ).withColumn(
            "to_city_code",
            when(col("wf_instance_id").isNotNull(), col("order_to_city_code")).otherwise(col("param_to_city_code"))
        ).withColumn(
            "to_coordinate",
            when(col("wf_instance_id").isNotNull(), col("order_to_coordinate")).otherwise(col("param_to_coordinate"))
        ).withColumn(
            "update_time",
            when(col("wf_instance_id").isNotNull(), col("order_update_time")).otherwise(col("param_update_time"))
        ).select(
            "instance_id", "user_id", "order_status",
            "from_address", "from_city_code",
            "to_address", "to_city_code", "update_time"
        )
    except Exception as e:
        # 订单文件为空，直接使用参数文件数据
        df_output = df_parameters_processed.select(
            col("instance_id"),
            col("param_user_id").alias("user_id"),
            lit(0).alias("order_status"),
            col("param_from_address").alias("from_address"),
            col("param_from_city_code").alias("from_city_code"),
            col("param_from_coordinate").alias("from_coordinate"),
            col("param_to_address").alias("to_address"),
            col("param_to_city_code").alias("to_city_code"),
            col("param_to_coordinate").alias("to_coordinate"),
            col("param_update_time").alias("update_time")
        )

    # 写入输出文件
    # df_output.write.mode("overwrite").csv("file:///data/yuchi/common_memory/output_table", header=True)
    df_output.write.mode("overwrite").csv(file_dw_ride, header=True)
    spark.stop()


if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_date = '2025-09-05'
    main(current_date)
