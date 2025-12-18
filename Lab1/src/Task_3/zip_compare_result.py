import os
import pickle
import re
import time
import statistics
from collections import defaultdict


def load_index(p):
    with open(p,'rb') as f:
        data = pickle.load(f)
    inv = data.get('inverted_index', data)
    return inv


def docid_to_int(docid):
    m = re.search(r"(\d+)(?!.*\d)", str(docid))
    if m:
        return int(m.group(1))
    try:
        return int(docid)
    except Exception:
        return None


def postings_to_sorted_list(postings):
    ids = []
    if not postings:
        return ids
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

    converted = []
    for d in ids:
        n = docid_to_int(d)
        converted.append(n if n is not None else str(d))

    converted_sorted = sorted(converted, key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
    return converted_sorted


def intersect_lists(a, b):
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


def time_func(f, runs=50, warmup=3):
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
    p_un = os.path.join(repo_root, 'outputs', 'Task_3', 'inverted_index.pkl')
    p_cmp = os.path.join(repo_root, 'outputs', 'Task_4', 'enhanced_inverted_index.pkl')
    if not os.path.exists(p_un) or not os.path.exists(p_cmp):
        print('cannot find one of the index files', p_un, p_cmp)
        return

    inv_un = load_index(p_un)
    inv_cmp = load_index(p_cmp)
    # 读取 Task_4 的完整数据
    with open(p_cmp, 'rb') as f:
        cmp_full = pickle.load(f)
    print('loaded indexes:', len(inv_un), len(inv_cmp))

    #几个样例
    queries = {
        'Q1': {'parts': [('or', ['community','meetup']), ('not', ['ticket','sale'])], 'expr': "(community OR meetup) AND NOT (ticket OR sale)"},
        'Q2': {'parts': [('and', ['photography','workshop']), ('and', ['hiking','outdoor'])], 'expr': "(photography AND workshop) AND (hiking AND outdoor)"},
        'Q3': {'parts': [('and', ['web','development','meetup']), ('not', ['please','join'])], 'expr': "web AND development AND meetup AND NOT (please OR join)"},
        'Q4': {'parts': [('or', ['volunteer','organizer']), ('and', ['music','meetup'])], 'expr': "(volunteer OR organizer) AND (music AND meetup)"},
        'Q5': {'parts': [('and', ['networking','startup']), ('or', ['food','drinks'])], 'expr': "(networking AND startup) AND (food OR drinks)"},
        'Q6': {'parts': [('and', ['community','event']), ('not', ['spam','advertisement'])], 'expr': "(community AND event) AND NOT (spam OR advertisement)"},
        'Q7': {'parts': [('and', ['photography','meetup']), ('or', ['food','drinks','networking']), ('not', ['ticket'])], 'expr': "(photography AND meetup) AND (food OR drinks OR networking) AND NOT ticket"},
        'Q8': {'parts': [('and', ['web','development']), ('and', ['photography','workshop']), ('not', ['spam','advertisement'])], 'expr': "(web AND development) AND (photography AND workshop) AND NOT (spam OR advertisement)"},
        'Q9': {'parts': [('or', ['music','food','community']), ('and', ['volunteer','organizer']), ('not', ['sale'])], 'expr': "(music OR food OR community) AND (volunteer AND organizer) AND NOT sale"},
        'Q10': {'parts': [('and', ['networking','meetup','startup']), ('or', ['web','development','photography'])], 'expr': "(networking AND meetup AND startup) AND (web OR development OR photography)"},
        'Q11': {'parts': [('or', ['hiking','outdoor','music']), ('not', ['join','please'])], 'expr': "(hiking OR outdoor OR music) AND NOT (join OR please)"},
        'Q12': {'parts': [('and', ['community','meetup']), ('and', ['food','drinks']), ('not', ['spam'])], 'expr': "(community AND meetup) AND (food AND drinks) AND NOT spam"},
        'Q13': {'parts': [('or', ['photography','music','web']), ('and', ['development','startup'])], 'expr': "(photography OR music OR web) AND (development AND startup)"},
        'Q14': {'parts': [('and', ['community','volunteer']), ('or', ['event','meetup','music'])], 'expr': "(community AND volunteer) AND (event OR meetup OR music)"},
        'Q15': {'parts': [('and', ['photography','networking']), ('not', ['sale','ticket'])], 'expr': "(photography AND networking) AND NOT (sale OR ticket)"},
        'Q16': {'parts': [('or', ['web','development','photography','music']), ('not', ['advertisement'])], 'expr': "(web OR development OR photography OR music) AND NOT advertisement"},
    }

    out_dir = os.path.join(repo_root, 'outputs', 'Task_3', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'zip_compare_result.csv')

    rows = []
    for idx_name, inv in [('uncompressed', inv_un), ('compressed_direct', inv_cmp), ('compressed_stream', inv_cmp)]:
    # 为所需词构建 postings 映射
        needed = set()
        for v in queries.values():
            for ptype, terms in v['parts']:
                for t in terms:
                    needed.add(t.lower())
        # 三种模式
        if idx_name == 'compressed_stream':
            blocks = cmp_full.get('blocks', [])
            bm = cmp_full.get('block_metadata', {})
            stream_postings = {}
            for t in needed:
                meta = bm.get(t)
                if not meta:
                    stream_postings[t] = []
                    continue
                try:
                    block_idx = meta[0]
                    blk = blocks[block_idx]
                    entry = blk.get(t)
                    if entry is None:
                        stream_postings[t] = postings_to_sorted_list(inv.get(t, []))
                    else:
                        raw = entry.get('postings') if isinstance(entry, dict) and 'postings' in entry else entry
                        stream_postings[t] = postings_to_sorted_list(raw)
                except Exception:
                    stream_postings[t] = postings_to_sorted_list(inv.get(t, []))
            postings = stream_postings
        else:
            postings = {t: postings_to_sorted_list(inv.get(t, [])) for t in needed}

    # 使用 postings 的评估器
        def eval_parts(parts):
            pos_parts = []
            neg_parts = []
            for ptype, terms in parts:
                if ptype == 'not':
                    neg_parts.append((ptype, terms))
                else:
                    lens = [len(postings.get(t.lower(), [])) for t in terms]
                    est = sum(lens) if ptype == 'or' else (min(lens) if lens else 0)
                    pos_parts.append((ptype, terms, est))
            pos_parts.sort(key=lambda x: x[2])

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
            if neg_parts:
                neg_union = []
                for _, terms in neg_parts:
                    for t in terms:
                        neg_union = union_lists(neg_union, postings.get(t.lower(), []))
                acc = difference_lists(acc, neg_union)
            return acc

        for qk, qv in queries.items():
            # 计时评估
            mean, sd = time_func(lambda: eval_parts(qv['parts']))
            # 计算一次结果以获取返回数量
            res = eval_parts(qv['parts'])
            rows.append((qk, qv['expr'], idx_name, mean, sd, len(res)))
            print(f"{idx_name} {qk} mean={mean:.3f}ms sd={sd:.3f}ms result_count={len(res)}")

    
    WRITE_CSV = True
    if WRITE_CSV:
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write('query,expr,index_type,mean_ms,sd_ms,result_count\n')
            for r in rows:
                f.write(','.join([str(x) for x in r]) + '\n')

        print('wrote', out_csv)


if __name__ == '__main__':
    main()
