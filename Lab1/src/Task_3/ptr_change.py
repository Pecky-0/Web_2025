import os
import pickle
import time
import math
import random
from statistics import mean

WORKDIR = r"c:\Users\11065\Desktop\web_lab1"
TASK3 = os.path.join(WORKDIR, 'outputs', 'Task_3')
ORIG_PATH = os.path.join(TASK3, 'inverted_index.pkl')
V1_PATH = os.path.join(TASK3, 'inverted_index_var1.pkl')
V2_PATH = os.path.join(TASK3, 'inverted_index_var2.pkl')
OUT_CSV = os.path.join(TASK3, 'experiments', 'ptr_change_result.csv')
REPORT1_PATH = os.path.join(TASK3, 'index_build_report_var1.txt')
REPORT2_PATH = os.path.join(TASK3, 'index_build_report_var2.txt')
NORM_ORIG = os.path.join(TASK3, 'inverted_index_norm_orig.pkl')
NORM_V1 = os.path.join(TASK3, 'inverted_index_norm_var1.pkl')
NORM_V2 = os.path.join(TASK3, 'inverted_index_norm_var2.pkl')

# 参数
VAR1_MIN_FOR_SKIPS = 50
VAR2_MIN_FOR_SKIPS = 200
MIN_LONG = 100
NUM_PAIRS = 200
REPEAT = 3
# 是否在完整运行中保存大型 pickles 与写构建报告
SKIP_SAVES = False


def docid_to_int(docid):
	try:
		if isinstance(docid, int):
			return docid
		if '_' in str(docid):
			return int(str(docid).split('_')[-1])
		return int(docid)
	except Exception:
		return abs(hash(str(docid))) % (10 ** 9)


def extract_doc_info(raw_list):
	"""尝试把倒排条目标准化为 (docid, freq) 的 tuple 列表（只针对基础层）。"""
	out = []
	if raw_list is None:
		return out
	if isinstance(raw_list, list) and len(raw_list) > 0 and isinstance(raw_list[0], list):
		base = raw_list[0]
	else:
		base = raw_list
	for item in base:
		if item is None:
			continue
		if isinstance(item, tuple) and len(item) >= 2:
			out.append((item[0], item[1]))
			continue
		if isinstance(item, dict):
			did = item.get('doc_id') or item.get('docid') or item.get('id')
			freq = item.get('freq') or item.get('tf') or 1
			out.append((did, freq))
			continue
		if isinstance(item, list) and len(item) >= 2:
			out.append((item[0], item[1]))
			continue
		try:
			a0 = item[0]
			a1 = item[1]
			out.append((a0, a1))
		except Exception:
			continue
	return out


def build_variant1(inverted_index):
	"""平方根步长跳转（内存中构建）"""
	var_index = {}
	terms_with_skips = 0
	total_skips = 0
	for term, raw in inverted_index.items():
		docs = extract_doc_info(raw)
		L = len(docs)
		if L <= VAR1_MIN_FOR_SKIPS:
			var_index[term] = docs
			continue
		step = max(2, int(math.sqrt(L)))
		new_list = []
		skips_this_term = 0
		for i, (docid, freq) in enumerate(docs):
			entry = {'doc_id': docid, 'freq': freq, 'skip_ptr': None}
			skip_i = i + step
			if skip_i < L:
				entry['skip_ptr'] = skip_i
				skips_this_term += 1
			new_list.append(entry)
		var_index[term] = new_list
		if skips_this_term > 0:
			terms_with_skips += 1
			total_skips += skips_this_term
	return var_index, terms_with_skips, total_skips


def build_variant2(inverted_index):
	"""块级指针策略（内存中构建）"""
	var_index = {}
	terms_with_skips = 0
	total_blocks = 0
	for term, raw in inverted_index.items():
		docs = extract_doc_info(raw)
		L = len(docs)
		if L <= VAR2_MIN_FOR_SKIPS:
			var_index[term] = docs
			continue
		block_size = max(50, int(L // 20))
		new_list = []
		blocks_this_term = 0
		for i, (docid, freq) in enumerate(docs):
			entry = {'doc_id': docid, 'freq': freq, 'block_ptr': None}
			if (i % block_size) == 0:
				next_block = i + block_size
				if next_block < L:
					entry['block_ptr'] = next_block
					blocks_this_term += 1
			new_list.append(entry)
		var_index[term] = new_list
		if blocks_this_term > 0:
			terms_with_skips += 1
			total_blocks += blocks_this_term
	return var_index, terms_with_skips, total_blocks


def flatten_base(postings):
	if postings is None:
		return []
	if isinstance(postings, list) and len(postings) > 0 and isinstance(postings[0], list):
		return postings[0]
	return postings


def normalize_index(raw_index, keep_skip_keys=('skip_ptr', 'block_ptr')):
	"""将原始索引标准化为 term -> list(dict) 格式（仅适配传入的子集）。"""
	norm = {}
	for term, postings in raw_index.items():
		base = flatten_base(postings)
		tuples = []
		for i, item in enumerate(base):
			if item is None:
				continue
			if isinstance(item, tuple) and len(item) >= 2:
				did, freq = item[0], item[1]
				raw = item
			elif isinstance(item, dict):
				did = item.get('doc_id') or item.get('docid') or item.get('id')
				freq = item.get('freq') or item.get('tf') or 1
				raw = item
			elif isinstance(item, list) and len(item) >= 2:
				did, freq = item[0], item[1]
				raw = item
			else:
				try:
					did = item[0]
					freq = item[1]
					raw = item
				except Exception:
					continue
			tuples.append((did, freq, raw))

		tuples_sorted = tuples
		norm_list = []
		did2idx = {}
		for idx, (did, freq, raw) in enumerate(tuples_sorted):
			norm_list.append({'doc_id': did, 'freq': freq})
			did2idx[did] = idx

		for idx, (did, freq, raw) in enumerate(tuples_sorted):
			for key in keep_skip_keys:
				if isinstance(raw, dict) and key in raw and raw.get(key) is not None:
					target = raw.get(key)
					mapped_idx = None
					if isinstance(target, int):
						try:
							target_item = base[target]
							if isinstance(target_item, tuple):
								target_did = target_item[0]
							elif isinstance(target_item, dict):
								target_did = target_item.get('doc_id') or target_item.get('docid')
							elif isinstance(target_item, list):
								target_did = target_item[0]
							else:
								target_did = None
							if target_did is not None and target_did in did2idx:
								mapped_idx = did2idx[target_did]
						except Exception:
							mapped_idx = None
					if isinstance(target, str) and target in did2idx:
						mapped_idx = did2idx[target]
					if mapped_idx is not None:
						norm_list[idx][key] = mapped_idx

		norm[term] = norm_list
	return norm


def fast_norm_from_raw(raw_index):
	"""为原始未添加指针的posting 做标准化"""
	out = {}
	for term, postings in raw_index.items():
		base = flatten_base(postings)
		lst = []
		if not base:
			out[term] = []
			continue
		for item in base:
			if item is None:
				continue
			if isinstance(item, tuple) and len(item) >= 2:
				lst.append({'doc_id': item[0], 'freq': item[1]})
			elif isinstance(item, dict):
				did = item.get('doc_id') or item.get('docid') or item.get('id')
				freq = item.get('freq') or item.get('tf') or 1
				lst.append({'doc_id': did, 'freq': freq})
			elif isinstance(item, list) and len(item) >= 2:
				lst.append({'doc_id': item[0], 'freq': item[1]})
			else:
				try:
					lst.append({'doc_id': item[0], 'freq': item[1]})
				except Exception:
					continue
		out[term] = lst
	return out


def fast_norm_from_variant(var_index):
	"""直接把变体的 posting 转为标准化 dict 列表，保留 skip_ptr/block_ptr"""
	out = {}
	for term, postings in var_index.items():
		lst = []
		if not postings:
			out[term] = []
			continue
		for item in postings:
			if item is None:
				continue
			if isinstance(item, dict):
				entry = {'doc_id': item.get('doc_id') or item.get('docid') or item.get('id'), 'freq': item.get('freq') or item.get('tf') or 1}
				# 保留可能存在的指针键
				if 'skip_ptr' in item and item.get('skip_ptr') is not None:
					entry['skip_ptr'] = item.get('skip_ptr')
				if 'block_ptr' in item and item.get('block_ptr') is not None:
					entry['block_ptr'] = item.get('block_ptr')
				lst.append(entry)
			elif isinstance(item, tuple) and len(item) >= 2:
				lst.append({'doc_id': item[0], 'freq': item[1]})
			elif isinstance(item, list) and len(item) >= 2:
				lst.append({'doc_id': item[0], 'freq': item[1]})
			else:
				try:
					lst.append({'doc_id': item[0], 'freq': item[1]})
				except Exception:
					continue
		out[term] = lst
	return out


def intersect_naive_ids(A_ids, B_ids):
	i = j = 0
	out = 0
	while i < len(A_ids) and j < len(B_ids):
		ai = docid_to_int(A_ids[i]); bi = docid_to_int(B_ids[j])
		if ai == bi:
			out += 1; i += 1; j += 1
		elif ai < bi:
			i += 1
		else:
			j += 1
	return out


def intersect_skip_dict(A, B, skip_key='skip_ptr'):
	i = j = 0
	out = 0
	while i < len(A) and j < len(B):
		a = A[i]; b = B[j]
		ai = docid_to_int(a['doc_id']); bi = docid_to_int(b['doc_id'])
		if ai == bi:
			out += 1; i += 1; j += 1
		elif ai < bi:
			skip_i = a.get(skip_key)
			if isinstance(skip_i, int) and skip_i < len(A):
				sd = A[skip_i]
				if docid_to_int(sd['doc_id']) <= bi:
					i = skip_i; continue
			i += 1
		else:
			skip_j = b.get(skip_key)
			if isinstance(skip_j, int) and skip_j < len(B):
				sd = B[skip_j]
				if docid_to_int(sd['doc_id']) <= ai:
					j = skip_j; continue
			j += 1
	return out


def counts_from_norm(inv, key):
	terms_with = 0; total = 0
	for t, p in inv.items():
		if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
			cnt = sum(1 for e in p if key in e and isinstance(e.get(key), int))
			if cnt > 0:
				terms_with += 1; total += cnt
	return terms_with, total


def load_pickle_index(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	# 返回 (inverted_index, meta_dict)
	if isinstance(data, dict) and 'inverted_index' in data:
		return data.get('inverted_index'), data
	return data, {}


def save_meta_pickle(inv_index, original_meta, path, variant_name):
	meta = {
		'inverted_index': inv_index,
		'doc_lengths': original_meta.get('doc_lengths') if isinstance(original_meta, dict) else None,
		'doc_ids': original_meta.get('doc_ids') if isinstance(original_meta, dict) else None,
		'term_positions': original_meta.get('term_positions') if isinstance(original_meta, dict) else None,
		'total_docs': original_meta.get('total_docs') if isinstance(original_meta, dict) else None,
		'skip_pointers_added': True,
		'variant': variant_name
	}
	print(f'Saving pickle for {variant_name} -> {path} ...')
	with open(path, 'wb') as f:
		pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
		try:
			f.flush()
			os.fsync(f.fileno())
		except Exception:
			pass
	print(f'Saved pickle for {variant_name}')


def write_report(path, variant, build_time, save_time, terms_total, terms_with_ptrs, total_ptrs):
	with open(path, 'w', encoding='utf-8') as f:
		f.write(f'{variant} 构建报告\n')
		f.write(f'构建耗时 (生成结构): {build_time:.2f} 秒\n')
		f.write(f'保存耗时: {save_time:.2f} 秒\n')
		f.write(f'词项总数: {terms_total}\n')
		f.write(f'使用指针的词项数 (近似): {terms_with_ptrs}\n')
		f.write(f'总指针数 (近似): {total_ptrs}\n')


def main():
	if not os.path.exists(ORIG_PATH):
		print('原始索引不存在:', ORIG_PATH)
		return
	print('加载原始索引...')
	raw_inv, meta = load_pickle_index(ORIG_PATH)
	# raw_inv 是 inverted_index 映射（term -> postings），meta 为原始 pickle 的完整元数据
	raw_orig = raw_inv

	# 内存中构建两个变体（并保存为 pickles）
	print('构建变体索引（内存中）...')
	t0 = time.time()
	var1_index, v1_terms_with, v1_total = build_variant1(raw_orig)
	t1 = time.time()
	var2_index, v2_terms_with, v2_total = build_variant2(raw_orig)
	t2 = time.time()

	# 保存 var1/var2 的 pickles（meta 包装）并写报告
	try:
		if not SKIP_SAVES:
			save_meta_pickle(var1_index, meta, V1_PATH, 'var1_sqrt')
			save_meta_pickle(var2_index, meta, V2_PATH, 'var2_block')
			save_time = time.time()
			# 写构建报告
			write_report(REPORT1_PATH, 'Variant 1 (sqrt-step)', t1 - t0, save_time - t1, len(var1_index), v1_terms_with, v1_total)
			write_report(REPORT2_PATH, 'Variant 2 (block-level)', t2 - t1, time.time() - t2, len(var2_index), v2_terms_with, v2_total)
		else:
			save_time = time.time()
			print('SKIP_SAVES=True: 跳过保存大型 pickles 与构建报告（快速运行模式）')
	except Exception as e:
		print('保存变体 pickles 或写报告时出错:', e)

	# 更新文件大小（刚保存的文件）
	size_orig = os.path.getsize(ORIG_PATH) if os.path.exists(ORIG_PATH) else 0
	size_v1 = os.path.getsize(V1_PATH) if os.path.exists(V1_PATH) else 0
	size_v2 = os.path.getsize(V2_PATH) if os.path.exists(V2_PATH) else 0

	# 选择长倒排表词项并对这些词项进行标准化（节省时间）
	print('选择长倒排表词项...')
	lengths = {}
	for term, postings in raw_orig.items():
		base = flatten_base(postings)
		try:
			lengths[term] = len(base)
		except Exception:
			lengths[term] = 0
	long_terms = [t for t, l in lengths.items() if l > MIN_LONG]
	print('长词项数量:', len(long_terms))

	# 仅对长词项做标准化（并保存标准化 pickle 以便离线检查）
	print('标准化长词项（原始与变体）...')
	sub_orig = {t: raw_orig[t] for t in long_terms if t in raw_orig}
	sub_v1 = {t: var1_index[t] for t in long_terms if t in var1_index}
	sub_v2 = {t: var2_index[t] for t in long_terms if t in var2_index}

	# 使用快速标准化路径：原始索引用轻量转换，变体直接转换并保留指针
	norm_orig = fast_norm_from_raw(sub_orig)
	norm_v1 = fast_norm_from_variant(sub_v1)
	norm_v2 = fast_norm_from_variant(sub_v2)
	try:
		if not SKIP_SAVES:
			print(f'Saving normalized pickles to {NORM_ORIG}, {NORM_V1}, {NORM_V2} ...')
			with open(NORM_ORIG, 'wb') as f:
				pickle.dump({'inverted_index': norm_orig}, f, protocol=pickle.HIGHEST_PROTOCOL)
				try:
					f.flush(); os.fsync(f.fileno())
				except Exception:
					pass
			with open(NORM_V1, 'wb') as f:
				pickle.dump({'inverted_index': norm_v1}, f, protocol=pickle.HIGHEST_PROTOCOL)
				try:
					f.flush(); os.fsync(f.fileno())
				except Exception:
					pass
			with open(NORM_V2, 'wb') as f:
				pickle.dump({'inverted_index': norm_v2}, f, protocol=pickle.HIGHEST_PROTOCOL)
				try:
					f.flush(); os.fsync(f.fileno())
				except Exception:
					pass
			print('Saved normalized pickles')
		else:
			print('SKIP_SAVES=True：跳过保存标准化 pickles（快速运行模式）')
	except Exception as e:
		print('保存标准化 pickle 时出错:', e)

	# 从标准化结构统计指针数量
	cw1, cp1 = counts_from_norm(norm_v1, 'skip_ptr')
	cw2, cp2 = counts_from_norm(norm_v2, 'block_ptr')

	# 生成词对, 重复测量
	pairs = []
	if len(long_terms) >= 2:
		all_pairs = [(a, b) for a in long_terms for b in long_terms if a != b]
		if len(all_pairs) <= NUM_PAIRS:
			pairs = all_pairs
		else:
			pairs = random.sample(all_pairs, NUM_PAIRS)

	results = {'orig': [], 'v1': [], 'v2': []}
	print('运行基准：pairs=%d repeat=%d' % (len(pairs), REPEAT))
	for r in range(REPEAT):
		random.shuffle(pairs)
		naive_times = []
		v1_times = []
		v2_times = []
		for a, b in pairs:
			A = norm_orig.get(a, [])
			B = norm_orig.get(b, [])
			if not A or not B:
				continue
			A_ids = [x['doc_id'] for x in A]
			B_ids = [x['doc_id'] for x in B]
			t0 = time.time(); _ = intersect_naive_ids(A_ids, B_ids); t1 = time.time(); naive_times.append(t1 - t0)

			A1 = norm_v1.get(a, [])
			B1 = norm_v1.get(b, [])
			if A1 and B1:
				t0 = time.time(); _ = intersect_skip_dict(A1, B1, skip_key='skip_ptr'); t1 = time.time(); v1_times.append(t1 - t0)

			A2 = norm_v2.get(a, [])
			B2 = norm_v2.get(b, [])
			if A2 and B2:
				t0 = time.time(); _ = intersect_skip_dict(A2, B2, skip_key='block_ptr'); t1 = time.time(); v2_times.append(t1 - t0)

		results['orig'].append(mean(naive_times) if naive_times else None)
		results['v1'].append(mean(v1_times) if v1_times else None)
		results['v2'].append(mean(v2_times) if v2_times else None)
		print('repeat', r + 1, 'orig_mean', results['orig'][-1], 'v1_mean', results['v1'][-1], 'v2_mean', results['v2'][-1])

	def safe_mean(lst):
		vals = [x for x in lst if x is not None]
		return mean(vals) if vals else None

	final = {
		'orig_mean': safe_mean(results['orig']),
		'v1_mean': safe_mean(results['v1']),
		'v2_mean': safe_mean(results['v2'])
	}

	# 写入输出
	
	os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
	with open(OUT_CSV, 'w', encoding='utf-8') as f:
		f.write('variant,filesize_bytes,terms_with_ptrs,total_ptrs,runtime_average_sec\n')
		f.write(f'orig,{size_orig},0,0,{final.get("orig_mean")}\n')
		f.write(f'var1,{size_v1},{cw1},{cp1},{final.get("v1_mean")}\n')
		f.write(f'var2,{size_v2},{cw2},{cp2},{final.get("v2_mean")}\n')

	print('已写入合并结果 CSV:', OUT_CSV)


if __name__ == '__main__':
	main()
