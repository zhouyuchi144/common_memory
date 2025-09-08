from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink, KafkaRecordSerializationSchema
from pyflink.common import WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import MapFunction  # 使用同步 MapFunction
import requests
import json
import re

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
        print("====2.1 resp:", output_data)
        return output_data['choices'][0]['message']['content']
    except Exception as e:
        return ""

def str2json_llm_output(arg1):
    if isinstance(arg1, str):
        pattern = re.compile(r'```(json)?\n?|```\n?')
        r = pattern.sub('', arg1)
        return json.loads(r)
    else:
        return arg1

class SyncExtractAddress(MapFunction):
    def map(self, value):
        system_msg = """# 任务
你是一个地址信息提取助手。请从用户的多轮对话中识别并提取家庭地址和公司地址。

# 要求：
家庭地址(home)和公司地址(comp)：完整地包含地址的全部细节
例如：菊园2号楼4单元301

# 输出
必须是严格的JSON格式，确保格式正确，不要增加任何字符
如果没有提取到某个地址，将其值设为空字符串。
示例输出：{"home": "菊园2号楼4单元301", "comp": "方正大厦东南门"}"""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": value}
        ]
        print(f"====1 user message:", value)
        try:
            resp = llm_interface(messages)
            resp = str2json_llm_output(resp)
            print("====2.2 resp:", resp)
            field_map = {"home": "addr_home", "comp": "addr_company"}
            rslt = {}
            has_value = False
            for field_llm, field_name in field_map.items():
                rslt[field_name] = resp.get(field_llm, "")
                if resp.get(field_llm, ""):
                    has_value = True
            if has_value:
                return json.dumps(rslt, ensure_ascii=False)
            else:
                return None  # Flink 会过滤掉 None 值（除非 sink 允许）
        except Exception as e:
            return None

def main():
    kafka_server = "alikafka-pre-cn-qzy4eerms02d-1-vpc.alikafka.aliyuncs.com:9092,alikafka-pre-cn-qzy4eerms02d-2-vpc.alikafka.aliyuncs.com:9092,alikafka-pre-cn-qzy4eerms02d-3-vpc.alikafka.aliyuncs.com:9092"
    env = StreamExecutionEnvironment.get_execution_environment()
    # 添加Kafka连接器jar包（根据实际路径修改）
    # env.add_jars("file:///path/to/flink-sql-connector-kafka-3.4.0-1.20")
    
    # 配置Kafka源
    source = KafkaSource.builder() \
        .set_bootstrap_servers(kafka_server) \
        .set_topics("test_to_com_mem") \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()
    
    # 配置Kafka sink
    sink = KafkaSink.builder() \
        .set_bootstrap_servers(kafka_server) \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
                .set_topic("com_mem_label_preprod")
                .set_value_serialization_schema(SimpleStringSchema())
                .build()
        ) \
        .build()
    
    # 数据流
    ds = env.from_source(source, WatermarkStrategy.no_watermarks(), "Kafka Source")
    processed_ds = ds.map(SyncExtractAddress()).filter(lambda x: x is not None)  # 过滤掉 None
    processed_ds.sink_to(sink)
    
    env.execute("Extract Address from Conversation")

if __name__ == "__main__":
    main()