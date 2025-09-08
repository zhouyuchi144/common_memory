from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, collect_list, struct
from pyspark.sql.types import StructType, StructField, StringType
import json
import re
import sys
import requests

def llm_interface(messages):
    url = 'http://172.16.1.151:8000/v1/chat/completions'
    headers = {"Content-Type": "application/json"}
    params = {
        "model": "/data/model/qwen3-14b",
        "messages": messages,
        "temperature": 0.6, "stream": False, "chat_template_kwargs": {"enable_thinking": False}
    }
    try:
        resp = requests.post(url, json=params, headers=headers)
        output_data = resp.json()
        return output_data['choices'][0]['message']['content']
    except Exception as e:
        return ""

system_prompt = """# 任务
你是一个地址信息提取助手。请从用户的多轮对话中识别并提取家庭地址和公司地址。

# 要求：
家庭地址(home)和公司地址(comp)：完整地包含地址的全部细节
例如：菊园2号楼4单元301

# 输出
必须是严格的JSON格式，确保格式正确，不要增加任何字符
如果没有提取到某个地址，将其值设为空字符串。
如果存在多个值，用分号分隔。
示例输出：{"home": "菊园2号楼4单元301;知春路2楼208", "comp": "方正大厦东南门"}"""

def str2json_llm_output(arg1):
    if isinstance(arg1, str):
        pattern = re.compile(r'```(json)?\n?|```\n?')
        r = pattern.sub('', arg1)
        return json.loads(r)
    else:
        return arg1

def extract_address_batch(msgs, rslt):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in msgs:
        role = "user" if msg.role == "user" else "assistant"
        messages.append({"role": role, "content": msg.msg})

    for i in range(3):
        try:
            resp = llm_interface(messages)
            resp = str2json_llm_output(resp)
            field_map = {"home": "address_home", "comp": "address_company"}
            for field_llm, field_name in field_map.items():
                if resp.get(field_llm, ""):
                    rslt[field_name] = f"{rslt[field_name]};{resp[field_llm]}" if rslt.get(field_name) else resp[field_llm]
            return rslt
        except Exception as e:
            pass
    return rslt

def proc_extract_address(msgs):
    rslt = {}
    # 分批次处理消息（每次最多1000条）
    MAX_BATCH = 1000
    for i in range(0, len(msgs), MAX_BATCH):
        batch_msgs = msgs[i:i+MAX_BATCH+2]
        rslt = extract_address_batch(batch_msgs, rslt)
    return rslt if rslt else None

def main(current_date):
    # 初始化 SparkSession
    spark = SparkSession.builder.appName("ExtractUserProfileProcess").getOrCreate()
    # 读取参数文件
    file_chat_hist = f"/data/chat_hist/partition_date={current_date}/"
    file_dw_label = f"/data/dw_label/partition_date={current_date}/"
    print(file_chat_hist)
    print(file_dw_label)

    proc_extract_address_udf = udf(proc_extract_address, StructType([
        StructField("address_home", StringType(), True),
        StructField("address_company", StringType(), True)
    ]))

    df = spark.read.csv(file_chat_hist, header=True)
    # 过滤并排序
    filtered_df = df.filter(col("msg_type") == "must") \
        .withColumn("id", col("id").cast("int")) \
        .orderBy(col("id"))

    # 按用户ID分组，收集消息历史
    grouped_df = filtered_df.groupBy("user_id") \
        .agg(collect_list(struct(col("role").alias("role"), col("msg").alias("msg"))).alias("messages"))

    # 提取地址信息
    df_output = grouped_df.withColumn("addresses", proc_extract_address_udf(col("messages"))) \
        .select(
        col("user_id"),
        col("addresses.address_home").alias("address_home"),
        col("addresses.address_company").alias("address_company")
    )

    df_output = df_output.filter(
        col("address_home").isNotNull() &
        col("address_company").isNotNull()
    )

    df_output.show()
    df_output.write.mode("overwrite").csv(file_dw_label, header=True)

    # 停止 SparkSession
    spark.stop()



if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_time = '2025-09-05 17:00:00'
    main(current_date)
