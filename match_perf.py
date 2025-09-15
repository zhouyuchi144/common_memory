import requests
from typing import List, Dict, Set
from pypinyin import lazy_pinyin




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

# 测试用例
if __name__ == "__main__":
    query_addr1 = "爱家家园"
    recall_addrs1 = ["李家5号", "爱家家园2单元"]
    print(f"查询地址: {query_addr1}, 召回地址: {recall_addrs1}")
    print(f"匹配结果: {find_match_addr(query_addr1, recall_addrs1)}")  # 输出: ["爱家家园2单元"]

    query_addr2 = "李家5号"
    recall_addrs2 = ["李家5号", "爱家家园2单元"]
    print(f"\n查询地址: {query_addr2}, 召回地址: {recall_addrs2}")
    print(f"匹配结果: {find_match_addr(query_addr2, recall_addrs2)}")  # 输出: ["李家5号"]

    query_addr = "学院南路"
    recall_addrs = ["学院南路中央组织部66号住房", "方正大厦(东南门)"]
    print(f"\n查询地址: {query_addr}, 召回地址: {recall_addrs}")
    print(f"匹配结果: {find_match_addr(query_addr, recall_addrs)}")  # 输出: ["学院南路中央组织部66号住房"]

    query_addr = "方正大厦"
    recall_addrs = ["学院南路中央组织部66号住房", "方正大厦(东南门)"]
    print(f"\n查询地址: {query_addr}, 召回地址: {recall_addrs}")
    print(f"匹配结果: {find_match_addr(query_addr, recall_addrs)}")  # 输出: ["学院南路中央组织部66号住房"]


    query_addr = "上帝五阶"
    recall_addrs = ["学院南路中央组织部66号住房", "方正大厦(东南门)", "上帝五阶", "上地五街8号", "上地三街8号"]
    print(f"\n查询地址: {query_addr}, 召回地址: {recall_addrs}")
    print(f"匹配结果: {find_match_addr(query_addr, recall_addrs)}")  # 输出: ["学院南路中央组织部66号住房"]

    query_addr = "上帝五阶"
    recall_addrs = ["学院南路中央组织部66号住房", "方正大厦(东南门)", "上地五街8号", "上地三街8号"]
    print(f"\n查询地址: {query_addr}, 召回地址: {recall_addrs}")
    print(f"匹配结果: {find_match_addr(query_addr, recall_addrs)}")  # 输出: ["学院南路中央组织部66号住房"]


