import os
import pickle
import re
import time
import statistics
from collections import defaultdict

# 轻量绘图(感觉不好用，还是在word里面吧)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_index(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    inverted = defaultdict(list, data.get('inverted_index', {}))
    return inverted


def docid_to_int(docid):
    m = re.search(r"(\d+)(?!.*\d)", str(docid))
    if m:
        return int(m.group(1))
    try:
        return int(docid)
    except Exception:
        return None


def postings_to_sorted_list(postings):
    """将 postings 规范化为有序的 docID 列表
    postings 可能是元组列表 (docid, freq)，或包含doc_id的字典列表，
    也可能是嵌套的跳表结构。总之返回一个按数值（或字符串）排序的列表"""
    ids = []
    if not postings:
        return ids
    # postings 可能是多层跳表结构
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
        if isinstance(it, tuple) and len(it) >= 1:
            ids.append(it[0])
        elif isinstance(it, dict) and 'doc_id' in it:
            ids.append(it['doc_id'])
        else:
            try:
                ids.append(it[0])
            except Exception:
                continue

    #尽量转换为整数
    converted = []
    for d in ids:
        n = docid_to_int(d)
        converted.append(n if n is not None else str(d))

    #先按数值排序，数值不可用时按字符串排序
    converted_sorted = sorted(converted, key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
    return converted_sorted


def intersect_lists(a, b):
    # a、b 均为已排序的 int 或字符串列表
    i = j = 0
    res = []
    while i < len(a) and j < len(b):
        ai = a[i]
        bj = b[j]
        if ai == bj:
            res.append(ai)
            i += 1
            j += 1
        else:
            # 支持混合类型的比较
            if isinstance(ai, int) and isinstance(bj, int):
                if ai < bj:
                    i += 1
                else:
                    j += 1
            else:
                if str(ai) < str(bj):
                    i += 1
                else:
                    j += 1
    return res


def union_lists(a, b):
    i = j = 0
    res = []
    # 按合并算法遍历两个已排序列表，生成并集
    while i < len(a) and j < len(b):
        ai = a[i]
        bj = b[j]
        if ai == bj:
            res.append(ai)
            i += 1
            j += 1
        else:
            if isinstance(ai, int) and isinstance(bj, int):
                if ai < bj:
                    res.append(ai); i += 1
                else:
                    res.append(bj); j += 1
            else:
                if str(ai) < str(bj):
                    res.append(ai); i += 1
                else:
                    res.append(bj); j += 1
    while i < len(a):
        res.append(a[i]); i += 1
    while j < len(b):
        res.append(b[j]); j += 1
    return res


def difference_lists(a, b):
    # 计算差集a-b
    i = j = 0
    res = []
    while i < len(a) and j < len(b):
        ai = a[i]
        bj = b[j]
        if ai == bj:
            i += 1; j += 1
        else:
            if isinstance(ai, int) and isinstance(bj, int):
                if ai < bj:
                    res.append(ai); i += 1
                else:
                    j += 1
            else:
                if str(ai) < str(bj):
                    res.append(ai); i += 1
                else:
                    j += 1
    while i < len(a):
        res.append(a[i]); i += 1
    return res


def time_func(f, runs=30, warmup=2):
    # 先热身调用
    for _ in range(warmup):
        f()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        f()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    pkl = os.path.join(repo_root, 'outputs', 'Task_3', 'inverted_index.pkl')
    if not os.path.exists(pkl):
        print('Cannot find', pkl)
        return

    inverted = load_index(pkl)
    print('Loaded inverted index: terms=', len(inverted))

    # 几个样例
    queries = {
        'Q1': {'expr': "(community OR meetup) AND NOT (ticket OR sale)", 'left_terms': ['community', 'meetup'], 'not_terms': ['ticket', 'sale']},
        'Q2': {'expr': "(photography AND workshop) AND (hiking AND outdoor)", 'ands1': ['photography', 'workshop'], 'ands2': ['hiking', 'outdoor']},
        'Q3': {'expr': "web AND development AND meetup AND NOT (please OR join)", 'ands': ['web', 'development', 'meetup'], 'not_terms': ['please', 'join']},
        'Q4': {'expr': "(volunteer OR organizer) AND (music AND meetup)", 'parts': [('or', ['volunteer', 'organizer']), ('and', ['music', 'meetup'])]},
        'Q5': {'expr': "(networking AND startup) AND (food OR drinks)", 'parts': [('and', ['networking', 'startup']), ('or', ['food', 'drinks'])]},
        'Q6': {'expr': "(community AND event) AND NOT (spam OR advertisement)", 'parts': [('and', ['community', 'event']), ('not', ['spam', 'advertisement'])]},
    }

    # 为所有需要的词准备 postings
    token_re = re.compile(r"[A-Za-z0-9_]+")
    needed = set()
    for v in queries.values():
        for k, val in v.items():
            if k == 'expr':
                continue
            # 若字段是特殊的 'parts'，则提取其内部的 term 列表
            if k == 'parts' and isinstance(val, list):
                for p in val:
                    if isinstance(p, (list, tuple)) and len(p) >= 2 and isinstance(p[1], list):
                        for t in p[1]:
                            needed.add(t.lower())
                continue
            #通用的term列表字段处理
            if isinstance(val, list):
                for t in val:
                    if isinstance(t, str):
                        needed.add(t.lower())
    postings = {t: postings_to_sorted_list(inverted.get(t, [])) for t in needed}

    results = {}

    #辅助函数
    def eval_q1(strategy):
        A = postings.get('community', [])
        B = postings.get('meetup', [])
        C = postings.get('ticket', [])
        D = postings.get('sale', [])

        #OR的长度直接左右相加
        left_est = len(A) + len(B)
        right_est = len(C) + len(D)

        order = ['left', 'right']
        if strategy == 'shortfirst':
            order = ['left', 'right'] if left_est <= right_est else ['right', 'left']
        elif strategy == 'longfirst':
            order = ['left', 'right'] if left_est >= right_est else ['right', 'left']

        # 按选择的顺序评估
        computed = {}
        for comp in order:
            if comp == 'left':
                computed['left'] = union_lists(A, B)
            else:
                computed['right'] = union_lists(C, D)

        return difference_lists(computed['left'], computed['right'])

    def eval_q2(strategy):
        p = postings.get('photography', [])
        w = postings.get('workshop', [])
        h = postings.get('hiking', [])
        o = postings.get('outdoor', [])

        # AND用最小长度
        est1 = min(len(p), len(w))
        est2 = min(len(h), len(o))

        parts = [('part1', (p, w), est1), ('part2', (h, o), est2)]
        if strategy == 'shortfirst':
            parts.sort(key=lambda x: x[2])
        elif strategy == 'longfirst':
            parts.sort(key=lambda x: x[2], reverse=True)

        # 先分别计算每个 part，然后对这些 part 的结果再求交
        r = None
        for name, (a, b), _ in parts:
            part_res = intersect_lists(a, b)
            if r is None:
                r = part_res
            else:
                r = intersect_lists(r, part_res)
        return r if r is not None else []

    def eval_q3(strategy):
        dev = postings.get('development', [])
        meet = postings.get('meetup', [])
        pls = postings.get('please', [])
        jn = postings.get('join', [])

        # 注意：确保 web 的 postings 已定义（来自 queries 中的 'web' token）
        web = postings.get('web', [])
        pos_terms = [web, dev, meet]
        pos_est = min(len(web), len(dev), len(meet))
        neg_est = len(pls) + len(jn)

        parts = [('pos', pos_terms, pos_est), ('neg', (pls, jn), neg_est)]
        if strategy == 'shortfirst':
            parts.sort(key=lambda x: x[2])
        elif strategy == 'longfirst':
            parts.sort(key=lambda x: x[2], reverse=True)

        computed = {}
        for name, terms, _ in parts:
            if name == 'pos':
                # 计算正向部分的交集
                r = intersect_lists(terms[0], terms[1])
                r = intersect_lists(r, terms[2])
                computed['pos'] = r
            else:
                # 计算负向部分的并集
                computed['neg'] = union_lists(terms[0], terms[1])

        return difference_lists(computed['pos'], computed['neg'])

    # 通用评估器
    def eval_parts(parts, strategy):
        # parts: 由 (type, [terms]) 组成，type 属于 {'or','and','not'}
        pos_parts = []
        neg_parts = []
        for ptype, terms in parts:
            if ptype == 'not':
                neg_parts.append((ptype, terms))
            else:
                # 计算长度
                lens = [len(postings.get(t.lower(), [])) for t in terms]
                est = sum(lens) if ptype == 'or' else (min(lens) if lens else 0)
                pos_parts.append((ptype, terms, est))

        # 根据策略对正向 parts 排序（shortfirst 升序，longfirst 降序）
        if strategy == 'shortfirst':
            pos_parts.sort(key=lambda x: x[2])
        elif strategy == 'longfirst':
            pos_parts.sort(key=lambda x: x[2], reverse=True)

        # 依次计算各正向 part，使用交集将它们合并
        acc = None
        for ptype, terms, _ in pos_parts:
            if ptype == 'or':
                cur = []
                for t in terms:
                    cur = union_lists(cur, postings.get(t.lower(), []))
            else:  
                cur = postings.get(terms[0].lower(), [])
                for t in terms[1:]:
                    cur = intersect_lists(cur, postings.get(t.lower(), []))
            if acc is None:
                acc = cur
            else:
                acc = intersect_lists(acc, cur)

        if acc is None:
            acc = []

        # 计算负向部分的并集并做差
        if neg_parts:
            neg_union = []
            for _, terms in neg_parts:
                for t in terms:
                    neg_union = union_lists(neg_union, postings.get(t.lower(), []))
            acc = difference_lists(acc, neg_union)

        return acc

    strategies = ['left', 'shortfirst', 'longfirst']
    # 评估每个查询；若查询含parts字段则使用 eval_parts 评估器
    for qk, qv in queries.items():
        results[qk] = {'expr': qv['expr']}
        for s in strategies:
            if 'parts' in qv:
                mean, sd = time_func(lambda: eval_parts(qv['parts'], s))
            else:
                if qk == 'Q1':
                    mean, sd = time_func(lambda: eval_q1(s))
                elif qk == 'Q2':
                    mean, sd = time_func(lambda: eval_q2(s))
                elif qk == 'Q3':
                    mean, sd = time_func(lambda: eval_q3(s))
                else:
                    mean, sd = (0.0, 0.0)
            results[qk][s] = (mean, sd)

    # 保存 CSV 结果
    out_dir = os.path.abspath(os.path.join(repo_root, 'outputs', 'Task_3', 'experiments'))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'short_first_merge_result.csv')
    #写入
    WRITE_CSV = True
    if WRITE_CSV:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('query,expr,strategy,mean_ms,sd_ms\n')
            for qk, v in results.items():
                for strat, vals in v.items():
                    if strat == 'expr':
                        continue
                    mean, sd = vals
                    f.write(f"{qk},{v['expr']},{strat},{mean:.6f},{sd:.6f}\n")

        print('Results saved to', csv_path)



if __name__ == '__main__':
    main()
