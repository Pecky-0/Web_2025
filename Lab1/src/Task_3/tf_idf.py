import os
import pickle
import math
import time
import statistics
from collections import defaultdict, Counter


def load_index(p):
    with open(p, 'rb') as f:
        data = pickle.load(f)
    inverted = data.get('inverted_index', {})
    doc_ids = data.get('doc_ids', None)
    total_docs = data.get('total_docs', None)
    return inverted, doc_ids, total_docs


def postings_to_doc_tf_list(postings):
    """从 postings 项中提取 (docid, tf) 列表"""
    out = []
    if not postings:
        return out
    # postings 可能是嵌套列表；将其扁平化一级
    items = []
    if isinstance(postings, list):
        for it in postings:
            if isinstance(it, list):
                for sub in it:
                    items.append(sub)
            else:
                items.append(it)
    else:
        try:
            for it in postings:
                items.append(it)
        except Exception:
            pass

    for it in items:
        if isinstance(it, tuple) and len(it) >= 2:
            out.append((it[0], int(it[1])))
        elif isinstance(it, tuple) and len(it) == 1:
            out.append((it[0], 1))
        elif isinstance(it, dict):
            # dict 可能使用 'doc_id' 或 'docid'，频率字段可能为 'freq' 或 'count'
            doc = None
            tf = None
            if 'doc_id' in it:
                doc = it.get('doc_id')
            elif 'docid' in it:
                doc = it.get('docid')
            elif 'doc' in it:
                doc = it.get('doc')
            # 频次字段处理
            tf = it.get('freq') or it.get('count') or it.get('tf')
            try:
                tf = int(tf) if tf is not None else 1
            except Exception:
                tf = 1
            if doc is not None:
                out.append((doc, tf))
        else:
            # 尝试把对象当作可索引的 (docid, freq) 处理
            try:
                d = it[0]
                f = it[1]
                out.append((d, int(f)))
            except Exception:
                #把标量视为 docid，tf=1
                try:
                    out.append((str(it), 1))
                except Exception:
                    continue
    return out


def tf_weight(tf):
    #权重 = 1 + log10(tf)
    if tf <= 0:
        return 0.0
    return 1.0 + math.log10(tf)


def idf_value(N, df):
    # idf = log10(N / df)
    if df <= 0:
        return 0.0
    return math.log10(float(N) / float(df))


def precompute_doc_norms(inverted, N):
    """遍历整个倒排索引，计算每个文档，得到dict docid->norm."""
    doc_norm_sq = defaultdict(float)
    term_df = {}
    # 首先计算每个词的文档频率（df）
    for term, postings in inverted.items():
        plist = postings_to_doc_tf_list(postings)
        df = len(plist)
        term_df[term] = df
    # 然后对每个词计算 idf，并累加每个文档的权重平方用于范数计算
    for term, postings in inverted.items():
        df = term_df.get(term, 0)
        if df == 0:
            continue
        idf = idf_value(N, df)
        if idf == 0.0:
            continue
        for docid, tf in postings_to_doc_tf_list(postings):
            w = tf_weight(tf) * idf
            doc_norm_sq[docid] += w * w
    doc_norm = {d: math.sqrt(s) for d, s in doc_norm_sq.items()}
    return doc_norm, term_df


def rank_query(query_terms, inverted, doc_norm, term_df, N, top_k=10):
    # query_terms: 查询词项列表（字符串）
    q_tf = Counter([t.lower() for t in query_terms])
    # 构建查询向量权重
    q_weights = {}
    for t, cnt in q_tf.items():
        df = term_df.get(t, 0)
        idf = idf_value(N, df)
        if idf == 0.0:
            continue
        q_weights[t] = tf_weight(cnt) * idf

    if not q_weights:
        return []
    # 计算查询向量范数
    q_norm_sq = sum(w * w for w in q_weights.values())
    q_norm = math.sqrt(q_norm_sq)
    if q_norm == 0:
        return []

    # 通过扫描查询词的 postings 累加点积
    scores = defaultdict(float)
    for t, q_w in q_weights.items():
        postings = inverted.get(t, [])
        for docid, tf in postings_to_doc_tf_list(postings):
            # 计算文档中该词的权重
            df = term_df.get(t, 0)
            idf = idf_value(N, df) if df>0 else 0.0
            doc_w = tf_weight(tf) * idf
            scores[docid] += doc_w * q_w
    # 计算并完成余弦相似度评分
    results = []
    for docid, dot in scores.items():
        dn = doc_norm.get(docid, 0.0)
        if dn == 0.0:
            continue
        sim = dot / (dn * q_norm)
        results.append((docid, sim))
    # 按相似度降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    p = os.path.join(repo, 'outputs', 'Task_3', 'inverted_index.pkl')
    if not os.path.exists(p):
        print('inverted index not found at', p)
        return

    inverted, doc_ids, total_docs = load_index(p)
    # 估计文档总数 N
    if total_docs:
        N = int(total_docs)
    else:
    # 从 doc_ids 或倒排索引中统计唯一文档数
        if isinstance(doc_ids, dict):
            N = len(doc_ids)
        else:
            # 扫描倒排索引，收集文档id
            docs = set()
            for postings in inverted.values():
                for d, _ in postings_to_doc_tf_list(postings):
                    docs.add(d)
            N = len(docs)

    print('loaded inverted index: terms=', len(inverted), 'N=', N)
    t0 = time.time()
    doc_norm, term_df = precompute_doc_norms(inverted, N)
    t1 = time.time()
    print('precompute doc norms done, docs with norm=', len(doc_norm), 'time=%.2fs' % (t1 - t0))

    # 几个样例
    queries = [
        ('Q1', ['photography', 'workshop']),
        ('Q2', ['web', 'development', 'meetup']),
        ('Q3', ['networking', 'startup']),
        ('Q4', ['community', 'volunteer', 'organizer']),
        ('Q5', ['food', 'drinks', 'event']),
    ]

    out_dir = os.path.join(repo, 'outputs', 'Task_3', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'tf_idf_result.csv')

    rows = []
    for qk, terms in queries:
        t0 = time.time()
        res = rank_query(terms, inverted, doc_norm, term_df, N, top_k=20)
        t1 = time.time()
        mean_time_ms = (t1 - t0) * 1000.0
        print('%s query=%s => returned %d docs, time=%.3fms' % (qk, ' '.join(terms), len(res), mean_time_ms))
        for rank, (docid, score) in enumerate(res, start=1):
            rows.append((qk, ' '.join(terms), rank, docid, score))

    # 写入
    WRITE_CSV = True
    if WRITE_CSV:
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write('query_key,query,rank,docid,score\n')
            for r in rows:
                f.write(','.join([str(x) for x in r]) + '\n')
        print('wrote', out_csv)


if __name__ == '__main__':
    main()
