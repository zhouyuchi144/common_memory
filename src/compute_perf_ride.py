from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, MapType
from typing import List, Dict, Set
from pypinyin import lazy_pinyin
import requests
import math
import json
import sys


def llm_rerank_interface(query, documents, model="gte-rerank-v2"):
    url = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'
    sdkey = "Bearer sk-9920838060a4455184ef7a433135b06c"
    headers = {"Authorization": sdkey, "Content-Type": "application/json"}
    params = {"model": model,
              "input":{"query": query, "documents": documents},
              "parameters": {"return_documents": True}
              }
    resp = requests.post(url, json=params, headers=headers)
    # print(resp.json())
    if resp.status_code == 200:
        output_data = resp.json()
        return output_data['output']['results']
    return ""

def rerank_rag_recall(query_content, contents):
    result = llm_rerank_interface(query_content, contents)
    print(result)
    ultra_cs = result[0].get('relevance_score', 0.0)
    super_match_contents1 = set()
    match_contents2 = set()
    for res in result:
        score = res.get('relevance_score', 0.0)
        content = res.get('document').get('text')
        if score >= 0.5:
            super_match_contents1.add(content)
            ultra_cs = res.get('relevance_score')
        else:
            if ultra_cs - score > 0.4 and score < 0.3: break
            if 0.5 <= ultra_cs < 0.6 and ultra_cs - score >= 0.3: break
            if score < 0.1: break
            match_contents2.add(content)
    rslt = super_match_contents1 if super_match_contents1 else match_contents2
    return list(rslt)

def proc_rule_match_addr(query_addr: str, recall_addrs: List[str]) -> List[str]:
    query_addr_pinyin = lazy_pinyin(query_addr)
    query_addr_pinyin_set: Set[str] = set(query_addr_pinyin)
    recall_addr_map: Dict[str, List[str]] = {addr: lazy_pinyin(addr) for addr in recall_addrs}
    super_match_addr = []
    match_addr = []
    if len(query_addr) <= 2:
        query = ''.join(query_addr_pinyin)
        super_match_addr = [addr for addr, addr_pinyin in recall_addr_map.items() if query in ''.join(addr_pinyin)]
    else:
        min_match_chars = len(query_addr) - 2
        for addr, addr_pinyin in recall_addr_map.items():
            tmp_set = query_addr_pinyin_set.copy()
            match_count = 0
            for char in addr_pinyin:
                if char in tmp_set:
                    match_count += 1
                    tmp_set.remove(char)
            if match_count >= min_match_chars:
                if tmp_set:
                    match_addr.append(addr)
                else:
                    super_match_addr.append(addr)
    return (super_match_addr, match_addr)

def find_match_addr(query_addr: str, recall_addrs: List[str]) -> List[str]:
    rule_super_match_addrs, rule_match_addrs = proc_rule_match_addr(query_addr, recall_addrs)
    if len(query_addr) <= 2: return rule_super_match_addrs

    match_addrs = rerank_rag_recall(query_addr, rule_super_match_addrs + rule_match_addrs)
    return list(set(rule_super_match_addrs + match_addrs))


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
        value["coordinate"] = msg.coordinate
        rslt.append(value)
    return json.dumps(rslt, ensure_ascii=False)

def comp_match_addr_info(addr_query, target_addrs, perf_target):
    rslt = {}
    tmp_match_homes = find_match_addr(addr_query.get("name"), target_addrs)
    if tmp_match_homes:
        max_home_score = -1
        max_home = ""
        for home in tmp_match_homes:
            if perf_target[home]["wilson_score"] > max_home_score: max_home_score = perf_target[home]["wilson_score"]
            max_home = home
        rslt["name"] = addr_query["name"]
        rslt["wilson_score"] = addr_query["wilson_score"] * perf_target[max_home]["wilson_score"]
        rslt["percent_score"] = addr_query["percent_score"] * perf_target[max_home]["percent_score"]
    return rslt

def proc_merge_perf(perf_addr, perf_addr_label):
    perf_addr_label = json.loads(perf_addr_label)
    rslt_perf_label = []
    perf_label = {home["name"]: home for home in perf_addr_label if home.get("name")}
    labels = perf_label.keys()
    for addr in perf_addr:
        tmp_rslt = comp_match_addr_info(addr, labels, perf_label)
        if tmp_rslt: rslt_perf_label.append(tmp_rslt)
    return json.dumps(rslt_perf_label, ensure_ascii=False)

def process_compute_perference(df, perf_name, cal_perf_score_udf, proc_merge_rslt_udf):
    df = df.filter(F.col("address").isNotNull())
    df = df.groupBy("user_id", "address", "coordinate").agg(F.sum("weight").alias("pos"))
    df = df.cache()
    df_total = df.groupBy("user_id").agg(F.sum("pos").alias("n"))
    df_rslt = df.join(df_total, ["user_id"])
    df_rslt = df_rslt.withColumn("perf_addr", cal_perf_score_udf("pos", "n"))
    df_rslt = df_rslt.groupBy("user_id") \
        .agg(F.collect_list(F.struct(F.col("address").alias("name"), F.col("coordinate"), F.col("perf_addr").alias("value"))).alias("msgs"))
    df_rslt = df_rslt.withColumn(perf_name, proc_merge_rslt_udf(F.col("msgs")))
    return (df_rslt, df)

def main(current_date):
    file_input = f"/data/dw_ride_di/"
    file_input_label = f"/data/up_perf_label_da/partition_date={current_date}/"
    file_output = f"/data/up_perf_ride_da/partition_date={current_date}/"
    print(file_input)
    print(file_output)

    cal_perf_score_udf = F.udf(cal_perf_score, MapType(StringType(), FloatType()))
    proc_merge_rslt_udf = F.udf(proc_merge_rslt, StringType())
    proc_merge_perf_udf = F.udf(proc_merge_perf, StringType())

    spark = SparkSession.builder.appName("AddressPreference").getOrCreate()
    # df = spark.read.csv("file:///data/yuchi/common_memory/output_table/*.csv", header=True, inferSchema=True)
    df = spark.read.csv(file_input, header=True, inferSchema=True)
    df_label = spark.read.csv(file_input_label, header=True, inferSchema=True)

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

    (df_rslt_from, df_from) = process_compute_perference(df.select("user_id", "weight", F.col("from_address").alias("address"), F.col("from_coordinate").alias("coordinate")), "perf_addr_from", cal_perf_score_udf, proc_merge_rslt_udf)
    (df_rslt_to, df_to) = process_compute_perference(df.select("user_id", "weight", F.col("to_address").alias("address"), F.col("to_coordinate").alias("coordinate")), "perf_addr_to", cal_perf_score_udf, proc_merge_rslt_udf)

    df_from = df_from.select("user_id", F.col("pos").alias("weight"), F.col("address"), F.col("coordinate"))
    df_to = df_to.select("user_id", F.col("pos").alias("weight"), F.col("address"), F.col("coordinate"))
    df0 = df_from.unionByName(df_to)
    (df_rslt_addr, df_addr) = process_compute_perference(df0.select("user_id", "weight", F.col("address"), F.col("coordinate")), "perf_addr", cal_perf_score_udf, proc_merge_rslt_udf)

    result_df = df_rslt_addr.join(df_rslt_from, "user_id", "left").join(df_rslt_to, "user_id", "left") \
                            .select("user_id", "perf_addr", "perf_addr_from", "perf_addr_to")

    result_df = result_df.join(df_label, "user_id", "left")
    result_df = result_df.withColumn("perf_addr_home", proc_merge_perf_udf("perf_addr", "perf_addr_home")) \
                         .withColumn("perf_addr_company", proc_merge_perf_udf("perf_addr", "perf_addr_company"))
    result_df = result_df.select("user_id", "perf_addr", "perf_addr_from", "perf_addr_to", "perf_addr_home", "perf_addr_company")

    result_df.write.mode("overwrite").csv(file_output, header=True)
    spark.stop()

if __name__ == "__main__":
    current_date = sys.argv[1]
    # current_date = '2025-09-06'
    main(current_date)
