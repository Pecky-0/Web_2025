import os
import pickle
import time
import statistics


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


def load_enhanced_index(p):
    with open(p, 'rb') as f:
        data = pickle.load(f)
    return data


def phrase_search_adjacent(term_positions, phrase_terms):
    # phrase_terms: 规范化的词项列表
    # 返回在文档中连续出现这些词项的 docid 列表
    if not phrase_terms:
        return []
    # 获取每个词的位置信息映射
    maps = [term_positions.get(t, {}) for t in phrase_terms]
    # 候选文档 = 所有词项键的交集
    doc_sets = [set(m.keys()) for m in maps]
    if not doc_sets:
        return []
    cand = set.intersection(*doc_sets)
    res = []
    for doc in cand:
    # 对于每个候选文档，检查位置链是否连续
        pos_lists = [set(m[doc]) for m in maps]
    # 遍历第一个词的所有位置并检查后续词是否在相邻位置
        found = False
        for p in pos_lists[0]:
            ok = True
            for i in range(1, len(pos_lists)):
                if (p + i) not in pos_lists[i]:
                    ok = False
                    break
            if ok:
                found = True
                break
        if found:
            res.append(doc)
    return res


def intersect_docs_without_positions(term_positions, phrase_terms):
    maps = [term_positions.get(t, {}) for t in phrase_terms]
    doc_sets = [set(m.keys()) for m in maps]
    if not doc_sets:
        return []
    return list(set.intersection(*doc_sets))


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    p = os.path.join(repo, 'outputs', 'Task_4', 'enhanced_inverted_index.pkl')
    if not os.path.exists(p):
        print('enhanced index not found at', p)
        return
    data = load_enhanced_index(p)
    term_positions = data.get('term_positions', {})
    print('term_positions loaded, terms=', len(term_positions))

    # 短语示例: web development, community meetup
    phrases = [
        ('phrase_web_dev', ['web', 'development']),
        ('phrase_community_meetup', ['community', 'meetup']),
    ]

    out_dir = os.path.join(repo, 'outputs', 'Task_3', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'search_phrase_result.csv')

    rows = []
    for name, terms in phrases:
    # 仅做交集
        mean_base, sd_base = time_func(lambda: intersect_docs_without_positions(term_positions, terms))
        res_base = intersect_docs_without_positions(term_positions, terms)
    # 要求位置相邻
        mean_phrase, sd_phrase = time_func(lambda: phrase_search_adjacent(term_positions, terms))
        res_phrase = phrase_search_adjacent(term_positions, terms)
        rows.append((name, ' '.join(terms), 'intersection_no_pos', mean_base, sd_base, len(res_base)))
        rows.append((name, ' '.join(terms), 'adjacent_phrase', mean_phrase, sd_phrase, len(res_phrase)))
        print(f"{name}: terms={terms} -> intersection={len(res_base)}, phrase_adjacent={len(res_phrase)}")
    # 打印
        print('  sample_intersection:', (res_base[:5]))
        print('  sample_phrase_adjacent:', (res_phrase[:5]))

    # 写入 CSV
    WRITE_CSV = True
    if WRITE_CSV:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('name,phrase,mode,mean_ms,sd_ms,result_count\n')
            for r in rows:
                f.write(','.join([str(x) for x in r]) + '\n')
        print('wrote', csv_path)


if __name__ == '__main__':
    main()
