from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
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
    to_address = None
    to_city_code = None

    for item in data_list:
        if isinstance(item, dict):
            if item.get("propertyName") == "start_place" and item.get("status") == "CONFIRMED":
                start_place_confirmed = True
                property_value = item.get("propertyValue", "{}")
                try:
                    property_value_dict = json.loads(property_value)
                    from_address = property_value_dict.get("label", "")
                    from_city_code = property_value_dict.get("cityCode", "")
                except:
                    pass
            elif item.get("propertyName") == "end_place" and item.get("status") == "CONFIRMED":
                end_place_confirmed = True
                property_value = item.get("propertyValue", "{}")
                try:
                    property_value_dict = json.loads(property_value)
                    to_address = property_value_dict.get("label", "")
                    to_city_code = property_value_dict.get("cityCode", "")
                except:
                    pass

    if start_place_confirmed and end_place_confirmed:
        return (from_address, from_city_code, to_address, to_city_code)
    else:
        return (None, None, None, None)

# 定义 UDF 解析订单文件的地址列
def parse_order_address(address_str):
    try:
        data = json.loads(address_str)
        address = data.get("address", "")
        city_code = data.get("cityCode", "")
        return (address, city_code)
    except:
        return (None, None)

def main(current_date):
    # 初始化 SparkSession
    spark = SparkSession.builder.appName("TaxiDataProcess").getOrCreate()
    # 读取参数文件
    # df_parameters = spark.read.csv("file:///data/yuchi/common_memory/data/ride_parameters.csv", header=True, escape='"')
    # df_order = spark.read.csv("file:///data/yuchi/common_memory/data/ride_hailing_order.csv", header=True, escape='"')
    file_ride_param = f"/data/ride_parameters/partition_date={current_date}/"
    file_ride_order = f"/data/ride_hailing_order/partition_date={current_date}/"
    file_dw_ride = f"/data/dw_ride_data/partition_date={current_date}/"
    print(file_ride_param)
    print(file_ride_order)
    print(file_dw_ride)
    df_parameters = spark.read.csv(file_ride_param, header=True)
    df_order = spark.read.csv(file_ride_order, header=True)
    # 注册 UDF
    parse_parameters_udf = udf(parse_parameters, StructType([
        StructField("from_address", StringType(), True),
        StructField("from_city_code", StringType(), True),
        StructField("to_address", StringType(), True),
        StructField("to_city_code", StringType(), True)
    ]))

    parse_order_address_udf = udf(parse_order_address, StructType([
        StructField("address", StringType(), True),
        StructField("city_code", StringType(), True)
    ]))

    # 解析 parameters 列并提取字段
    df_parameters_parsed = df_parameters.withColumn("parsed_params", parse_parameters_udf(col("parameters")))
    df_parameters_parsed = df_parameters_parsed.withColumn("param_from_address", col("parsed_params.from_address"))
    df_parameters_parsed = df_parameters_parsed.withColumn("param_from_city_code", col("parsed_params.from_city_code"))
    df_parameters_parsed = df_parameters_parsed.withColumn("param_to_address", col("parsed_params.to_address"))
    df_parameters_parsed = df_parameters_parsed.withColumn("param_to_city_code", col("parsed_params.to_city_code"))

    # 过滤有效数据（start_place 和 end_place 均为 CONFIRMED）
    df_parameters_valid = df_parameters_parsed.filter(
        col("param_from_address").isNotNull() &
        col("param_to_address").isNotNull()
    )

    # 选择参数文件需要的列
    df_parameters_selected = df_parameters_valid.select(
        col("instance_id"),
        col("user_id").alias("param_user_id"),
        col("param_from_address"),
        col("param_from_city_code"),
        col("param_to_address"),
        col("param_to_city_code"),
        col("updated_time").alias("param_update_time")
    )

    # 解析订单文件的地址列
    df_order_parsed = df_order.withColumn("parsed_from", parse_order_address_udf(col("from_address")))
    df_order_parsed = df_order_parsed.withColumn("parsed_to", parse_order_address_udf(col("to_address")))
    df_order_parsed = df_order_parsed.withColumn("order_from_address", col("parsed_from.address"))
    df_order_parsed = df_order_parsed.withColumn("order_from_city_code", col("parsed_from.city_code"))
    df_order_parsed = df_order_parsed.withColumn("order_to_address", col("parsed_to.address"))
    df_order_parsed = df_order_parsed.withColumn("order_to_city_code", col("parsed_to.city_code"))

    # 选择订单文件需要的列
    df_order_selected = df_order_parsed.select(
        col("wf_instance_id"),
        col("user_id").alias("order_user_id"),
        col("order_from_address"),
        col("order_from_city_code"),
        col("order_to_address"),
        col("order_to_city_code"),
        col("updated_time").alias("order_update_time"),
        col("order_status")
    )

    # 左连接参数文件和订单文件
    df_joined = df_parameters_selected.join(
        df_order_selected,
        df_parameters_selected.instance_id == df_order_selected.wf_instance_id,
        "left"
    )

    # 根据是否存在订单数据选择字段
    df_result = df_joined.withColumn(
        "instance_id",
        when(col("wf_instance_id").isNotNull(), col("wf_instance_id")).otherwise(col("instance_id"))
    ).withColumn(
        "user_id",
        when(col("wf_instance_id").isNotNull(), col("order_user_id")).otherwise(col("param_user_id"))
    ).withColumn(
        "final_from_address",
        when(col("wf_instance_id").isNotNull(), col("order_from_address")).otherwise(col("param_from_address"))
    ).withColumn(
        "final_from_city_code",
        when(col("wf_instance_id").isNotNull(), col("order_from_city_code")).otherwise(col("param_from_city_code"))
    ).withColumn(
        "final_to_address",
        when(col("wf_instance_id").isNotNull(), col("order_to_address")).otherwise(col("param_to_address"))
    ).withColumn(
        "final_to_city_code",
        when(col("wf_instance_id").isNotNull(), col("order_to_city_code")).otherwise(col("param_to_city_code"))
    ).withColumn(
        "final_update_time",
        when(col("wf_instance_id").isNotNull(), col("order_update_time")).otherwise(col("param_update_time"))
    ).withColumn(
        "final_order_status",
        when(col("wf_instance_id").isNotNull(), col("order_status")).otherwise(0)
    )

    # 选择输出列
    df_output = df_result.select(
        col("instance_id"),
        col("user_id"),
        col("final_order_status").alias("order_status"),
        col("final_from_address").alias("from_address"),
        col("final_from_city_code").alias("from_city_code"),
        col("final_to_address").alias("to_address"),
        col("final_to_city_code").alias("to_city_code"),
        col("final_update_time").alias("update_time")
    )

    # 显示结果（可选）
    df_output.show()

    # 写入输出文件
    # df_output.write.mode("overwrite").csv("file:///data/yuchi/common_memory/output_table", header=True)
    df_output.write.mode("overwrite").csv(file_dw_ride, header=True)

    # 停止 SparkSession
    spark.stop()



if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_time = '2025-09-05 17:00:00'
    main(current_date)