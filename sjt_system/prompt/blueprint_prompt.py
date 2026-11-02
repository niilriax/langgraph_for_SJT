"""Prompt for selecting situations into the two-way blueprint."""


BLUEPRINT_PROMPT = """
根据 facet、行为证据扩展和题数要求，设计 PSJT 双向细目表。测量轴是 facet + behavior_id，候选情境轴是 mechanism_id + situation_id。

选择规则
────────
- 返回恰好 row_total 个互不相同的测量行，每行恰好一个 candidate_reference；所有候选引用共 generation_total 个，互不相同，从扩展阶段的情境池中选。
- ID 原样引用，不要创建或推测。
- 优先保证构念代表性：题数允许时覆盖每个 behavior_id；同一行为内，优先选不同的激活机制。
- 每个 facet 按 facet_retention_quotas 返回指定行数，不在 facet 之间调配。

机制筛选
────────
用对应证据的 behavior_dimension、high_expression、low_expression、boundary_condition 检验每个候选机制。出现下列情况，就不选它的任何情境：
- 机制描述的行为模式不在 high_expression 或 low_expression 范围内（如坚持与停止投入、一般的服从与反抗）；
- 机制依赖 boundary_condition 明确排除的行为信号；
- 机制换成别的 facet 也照样成立，说明它没有被本 facet 的行为约束。

修正与不足
────────
- 有 blueprint_semantic_feedback 时，把它当作对上一版的硬性修正：替换其中指出的候选引用，并重新检查所有行之间的唯一性。
- 构念纯净的候选不足 row_total 时，只返回纯净的最大子集，由程序报错；不要用超出范围的机制凑数，也不要重复情境。
""".strip()
