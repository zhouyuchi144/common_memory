from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, collect_list, struct
from pyspark.sql.types import StructType, StructField, StringType
import json
import re
import sys
import requests

system_prompt = """# 任务
你是一个严谨的信息提取助手。从用户多轮对话中提取【我家的地址】和【我公司地址】

# 核心指令：必须严格分两步思考和执行
对每行【多轮对话】数据，分别提取【我家的地址】和【我公司地址】

## 第1步：地址有效性审查（这是最重要的前提）
在提取任何地址之前，你必须先判断它是否为【具体地址】
如果地址审查无效，则直接丢弃，不要进入第2步

###【具体地址】定义：
- 必须包含街道、小区、楼栋、门牌号等详细信息
###【无效地址】定义：
- 【非具体地址】：如果一个地点仅包含城市/区县/商圈（如“天津市”、“朝阳区”、“中关村”），则它不是具体地址，必须忽略。例如：`我家在朝阳区`，提取home=""
- 线索词：像“家”、“公司”、“办公室”这类词是判断归属的**线索**，它们本身**绝不能**被提取为地址。例如：用户说`我在家`，提取home=""

## 第2步：归属判断和提取（仅对通过第1步审查的有效地址执行）
只有当一个地址是【具体地址】时，才根据以下规则判断其归属。
- 我家的地址 (home): 必须与"我家在..."、"我住在..."、"回我家..."、"我的住址是..."等直接表明归属的词语强关联。
- 我公司地址 (comp)：必须与"我公司在..."、"我们单位是..."、"我去上班的地方..."等直接表明归属的词语强关联。
- 禁止联想推断。
  例子1：**绝对不能**因为行程的终点是“家”，就推断其起点是“公司”。示例：用户说“从昊海大厦地出发回家”，昊海大厦不是“公司”
  例子2：**绝对不能**因为用户在某个地点开会、或参加活动，就推断该地点是“公司”。示例：用户说“在福道大厦开会”，福道大厦不是“公司”

# 输出
必须是严格的JSON格式，确保格式正确，不要增加任何字符
- 如果没有提取到某个地址，将其值设为空字符串。
- 如果存在多个值，用分号分隔。
- 示例：{"home": "菊园2号楼301;知春路(公交站)", "comp": "昊海大厦(东南门)"}
"""

def llm_interface(messages):
    sdkey = "Bearer sk-9920838060a4455184ef7a433135b06c"
    url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
    headers = {"Authorization": sdkey, "Content-Type": "application/json"}
    params = {
        "model": "qwen-plus",
        "input":{"messages": messages},
        "parameters": {"temperature": 0.6, "result_format": "message", "stream": False, "enable_thinking": False}
    }
    resp = requests.post(url, json=params, headers=headers)
    # print(resp.json())
    if resp.status_code == 200:
        output_data = resp.json()
        return output_data['output']['choices'][0]['message']['content']
    return ""

def str2json_llm_output(arg1):
    if isinstance(arg1, str):
        pattern = re.compile(r'";')
        arg1 = pattern.sub('",', arg1)
        pattern = re.compile(r'(```(json)?\n?|```\n?|\n(```)?)')
        r = pattern.sub('', arg1)
        return json.loads(r)
    else:
        return arg1

def address_clean(rslt):
    rslt_clean = {}
    for name, value in rslt.items():
        unique_values = set()
        for v in value.split(";"):
            if len(v) <= 3: continue
            unique_values.add(v)
        if unique_values: rslt_clean[name] = ";".join(unique_values)
    return rslt_clean

# def extract_address_batch2(msgs, rslt):
#     messages = [{"role": "system", "content": system_prompt}]
#     for msg in msgs:
#         role = "user" if msg.role == "user" else "assistant"
#         messages.append({"role": role, "content": msg.msg})
#     # print(messages)
#
#     for i in range(3):
#         try:
#             resp = llm_interface(messages)
#             # print(resp)
#             resp = str2json_llm_output(resp)
#             field_map = {"home": "address_home", "comp": "address_company"}
#             for field_llm, field_name in field_map.items():
#                 if resp.get(field_llm, ""):
#                     rslt[field_name] = f"{rslt[field_name]};{resp[field_llm]}" if rslt.get(field_name) else resp[field_llm]
#             break
#         except Exception as e:
#             pass
#     return rslt

def extract_address_batch(msgs, rslt):
    chat = "\n".join(msgs)
    user_message = {"role": "user", "content": f"# 输入\n【多轮会话】：（每行数据表示一次多轮对话）\n{chat}"}
    messages = [{"role": "system", "content": system_prompt}, user_message]
    # print(user_message)

    for i in range(3):
        try:
            resp = llm_interface(messages)
            # print("--  resp=",resp, "mst=",user_message)
            resp = str2json_llm_output(resp)
            field_map = {"home": "address_home", "comp": "address_company"}
            for field_llm, field_name in field_map.items():
                if resp.get(field_llm, ""):
                    rslt[field_name] = f"{rslt[field_name]};{resp[field_llm]}" if rslt.get(field_name) else resp[field_llm]
            break
        except Exception as e:
            pass

    if rslt: rslt = address_clean(rslt)
    return rslt

def gen_messages(msgs):
    msgs = sorted(msgs, key=lambda x: (x['id']))
    ms = []
    pre_conversation_id = "NULL"
    m = ""
    for msg in msgs:
        new_conversation_id = msg.conversation_id
        role = "用户" if msg.role == "user" else "你"
        if pre_conversation_id != new_conversation_id:
            if m and re.search(r'(家|住|公司|单位|上班|下班)', m): ms.append(m)
            m = f"{role}:{msg.msg}"
            pre_conversation_id = new_conversation_id
        else:
            m = f"{m}; {role}:{msg.msg}"
    if m and re.search(r'(家|住|公司|单位|上班|下班)', m): ms.append(m)

    return ms

def proc_extract_address(msgs):
    rslt = {}
    ms = gen_messages(msgs)
    # 分批次处理消息（每次最多200条）
    MAX_BATCH = 200
    for i in range(0, len(ms), MAX_BATCH):
        batch_ms = ms[i:i + MAX_BATCH]
        rslt = extract_address_batch(batch_ms, rslt)
        user_prompt = '\n'.join(batch_ms)
        if rslt: print(f"===={i}result: ", rslt, f"{user_prompt}")
    # # 分批次处理消息（每次最多1000条）
    # MAX_BATCH = 1000
    # for i in range(0, len(msgs), MAX_BATCH):
    #     batch_msgs = msgs[i:i+MAX_BATCH+10]
    #     rslt = extract_address_batch2(batch_msgs, rslt)
    return rslt if rslt else None

def main(current_date):
    # 初始化 SparkSession
    spark = SparkSession.builder.appName("ExtractUserProfileProcess").getOrCreate()
    # 读取参数文件
    file_chat_hist = f"/data/chat_hist/partition_date={current_date}/"
    file_dw_label = f"/data/dw_label_di/partition_date={current_date}/"
    print(file_chat_hist)
    print(file_dw_label)

    proc_extract_address_udf = udf(proc_extract_address, StructType([
        StructField("address_home", StringType(), True),
        StructField("address_company", StringType(), True)
    ]))

    df = spark.read.csv(file_chat_hist, header=True)
    # 过滤并排序
    filtered_df = df.filter(col("msg_type") == "must") \
        .withColumn("intent_id", col("intent_id").cast("string")) \
        .withColumn("user_id", col("user_id").cast("string")) \
        .filter(col("intent_id") != "10007") \
        .withColumn("id", col("id").cast("int"))

    # 按用户ID分组，收集消息历史
    grouped_df = filtered_df.groupBy("user_id", "intent_id") \
        .agg(collect_list(struct(col("role").alias("role"), col("msg").alias("msg"), col("conversation_id").alias("conversation_id"), col("id").alias("id"))).alias("messages"))

    # 提取地址信息
    df_output = grouped_df.withColumn("addresses", proc_extract_address_udf(col("messages")))
    df_output.cache()
    df_output = df_output.filter(col("addresses").isNotNull()) \
        .select(
        col("user_id"),
        col("intent_id"),
        col("addresses.address_home").alias("address_home"),
        col("addresses.address_company").alias("address_company")
    )

    df_output.show(truncate=False)
    df_output.write.mode("overwrite").csv(file_dw_label, header=True)

    # 停止 SparkSession
    spark.stop()



if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_time = '2025-09-05 17:00:00'
    main(current_date)
