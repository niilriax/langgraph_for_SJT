"""Program-owned diversity controls for initial item generation.

Items of one facet are planned so that their level differences run along
different behavioral channels, option texts avoid naming the trait, and a
post-generation check flags option templates shared by too many items.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

# Facet-agnostic channels along which the four behavioral levels can differ.
DIFFERENCE_CHANNELS = (
    "参与规模：卷入的人数多少、熟人还是陌生人",
    "投入时长：停留或投入多久、是否提前结束",
    "发起与组织：由谁发起、是否自己牵头安排",
    "沟通表达：说什么、怎么回应、是否表态",
    "位置与距离：选择待在哪里、离关注焦点多近",
    "事后跟进：之后是否再联系、是否延续或补救",
    "取舍安排：放弃或改变哪项原有安排",
    "借助他人：是否请人陪同、代劳或转交",
)

# Words that name the measured trait or its poles. Options and skeleton
# anchors must describe behavior, not label it.
FACET_LABEL_WORDS: dict[str, tuple[str, ...]] = {
    "extraversion_gregariousness": ("主动", "积极", "热闹", "爱社交", "合群", "喜欢人多", "享受"),
    "neuroticism_self_consciousness": ("害怕", "怕", "紧张", "尴尬", "难为情", "羞", "丢脸"),
    "openness_ideas": ("好奇", "开放", "思辨", "感兴趣"),
    "agreeableness_compliance": ("顺从", "服从", "妥协", "让步"),
    "conscientiousness_self_discipline": ("自律", "拖延", "坚持", "懒"),
}

_STOP_BIGRAMS = {
    "同学", "一起", "继续", "自己", "对方", "大家", "别人", "他们", "我们", "然后",
    "之后", "一下", "一会", "时候", "没有", "不是", "还是", "可以", "但是", "如果",
    "朋友", "熟人", "室友",
}
_CHINESE = re.compile(r"[一-鿿]")


def difference_channels_for(facet_id: str, count: int) -> list[str]:
    """Deterministic per-facet rotation so neighbouring slots differ."""
    seed = int(hashlib.sha256(str(facet_id).encode("utf-8")).hexdigest()[:8], 16)
    order = list(DIFFERENCE_CHANNELS)
    random.Random(seed).shuffle(order)
    return [order[index % len(order)] for index in range(count)]


def label_words_for(facet_id: Any) -> list[str]:
    return list(FACET_LABEL_WORDS.get(str(facet_id or ""), ()))


def label_hits(text: Any, words: Sequence[str]) -> list[str]:
    value = str(text or "")
    return [word for word in words if word and word in value]


def skeleton_label_problems(skeleton: Mapping[str, Any], words: Sequence[str]) -> list[str]:
    problems = []
    for row in skeleton.get("option_structure") or []:
        if not isinstance(row, Mapping):
            continue
        hits = label_hits(row.get("behavioral_tendency"), words)
        if hits:
            problems.append(
                f"{row.get('behavioral_level')} 的 behavioral_tendency 使用了构念标签词 {hits}；"
                "改写为该情境中可观察的具体行为，不要用这些词")
    return problems


def _opening(text: str) -> str:
    chars = _CHINESE.findall(re.sub(r"^你", "", str(text or "").strip()))
    return "".join(chars[:2])


def _bigrams(text: str) -> set[str]:
    chars = "".join(_CHINESE.findall(str(text or "")))
    return {chars[i:i + 2] for i in range(len(chars) - 1)} - _STOP_BIGRAMS


def _level_texts(item: Mapping[str, Any]) -> dict[int, str]:
    key = item.get("scoring_key") or {}
    return {int(key[row["option_id"]]): str(row.get("text") or "")
            for row in item.get("response_options") or []
            if isinstance(row, Mapping) and row.get("option_id") in key}


def overuse_limit(item_count: int) -> int:
    return max(3, math.ceil(item_count * 0.25))


def option_template_report(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flag option templates shared by too many items of one facet.

    For every score level, a label word in any option, or an opening / content
    bigram used by more items than ``overuse_limit`` marks the excess items
    (the earliest ``limit`` users of a token are kept)."""
    items = list(items)
    words = label_words_for(items[0].get("target_dimension_id")) if items else []
    limit = overuse_limit(len(items))
    flagged: dict[str, list[str]] = {}
    overused: dict[int, dict[str, int]] = {}

    def flag(item_id: str, reason: str) -> None:
        flagged.setdefault(item_id, [])
        if reason not in flagged[item_id]:
            flagged[item_id].append(reason)

    levels = sorted({level for item in items for level in _level_texts(item)})
    for level in levels:
        users: dict[str, list[str]] = {}
        for item in items:
            text = _level_texts(item).get(level, "")
            item_id = str(item["item_id"])
            for word in label_hits(text, words):
                flag(item_id, f"{level}分选项使用构念标签词“{word}”")
            tokens = {f"开头“{_opening(text)}”"} if _opening(text) else set()
            tokens |= {f"“{token}”" for token in _bigrams(text)}
            for token in tokens:
                users.setdefault(token, []).append(item_id)
        overused[level] = {token: len(ids) for token, ids in users.items() if len(ids) > limit}
        for token, ids in users.items():
            if len(ids) > limit:
                for item_id in ids[limit:]:
                    flag(item_id, f"{level}分选项与过多题目共用{token}（{len(ids)}题）")
    return {"limit": limit, "flagged": flagged, "overused": overused, "label_words": words}


def option_diversity_context(items: Sequence[Mapping[str, Any]], item_id: str,
                             reasons: Sequence[str]) -> dict[str, Any]:
    """Sibling options and overused tokens shown to the writer on a rewrite."""
    siblings = [item for item in items if str(item["item_id"]) != str(item_id)]
    report = option_template_report(items)
    return {
        "why_rewrite": list(reasons),
        "sibling_options_by_score": {
            str(level): [_level_texts(item).get(level, "") for item in siblings]
            for level in sorted({lv for item in siblings for lv in _level_texts(item)})
        },
        "overused_tokens_by_score": {str(level): sorted(tokens) for level, tokens in report["overused"].items()},
        "facet_label_words": report["label_words"],
        "instruction": ("重写本题的情境与四个选项：选项不要使用上面已被多题共用的开头和词，"
                        "不要沿用兄弟题选项的句式；不得出现构念标签词；行为等级和计分不变。"),
    }


def option_avoidance_hint(items: Sequence[Mapping[str, Any]], facet_id: Any) -> dict[str, Any]:
    """Label words and already-overused option tokens of one facet's items."""
    siblings = [item for item in items if item.get("target_dimension_id") == facet_id]
    report = option_template_report(siblings) if siblings else {"overused": {}}
    return {
        "facet_label_words": label_words_for(facet_id),
        "overused_tokens_by_score": {str(level): sorted(tokens)
                                     for level, tokens in report["overused"].items() if tokens},
    }


def option_label_problems(item: Mapping[str, Any], option_ids: Sequence[str] | None = None) -> list[str]:
    """Label words used in (the given) option texts of one item."""
    words = label_words_for(item.get("target_dimension_id"))
    wanted = set(option_ids) if option_ids is not None else None
    problems = []
    for row in item.get("response_options") or []:
        if wanted is not None and row.get("option_id") not in wanted:
            continue
        hits = label_hits(row.get("text"), words)
        if hits:
            problems.append(f"选项 {row.get('option_id')} 使用了构念标签词 {hits}")
    return problems


# Scenario style: telegraphic stems (semicolon-joined fragments, listed
# alternatives) and free-choice or reviewer-facing statements read unlike
# ordinary SJT stems, which only tell what happened.
SCENARIO_MAX_CHARS = 90
FREE_CHOICE_PHRASES = (
    "不强制", "没有强制", "无人强制", "没人勉强", "不勉强", "没人强迫", "可换人", "都行", "都可以",
    "去不去都行", "做不做都行", "去不去", "做不做", "要不要都", "不挨罚", "没规定", "没有规定",
    "没人规定", "没要求", "没有要求", "没人要求", "没有人要求", "自愿", "随你", "随自己", "由你定",
    "都由你", "也行", "无所谓", "也可以", "也可",
)
REVIEWER_NOTE_PHRASES = ("违法", "违规", "不算冒犯", "没人盯着", "没有人盯着", "无人责怪", "没人责怪", "不会被处罚")


OPTION_FORMULA_PHRASES = ("省事", "省得", "两边都顾", "兼顾", "权衡", "顺势", "视其为", "着手")


def _outside_quotes(text: str) -> str:
    import re
    return re.sub(r"[“\"「『][^”\"」』]*[”\"」』]", "", text)


def option_style_problems(options: list[str]) -> list[str]:
    """Rule-based AI-flavour problems in one item's option texts."""
    problems = []
    joined = [str(text or "") for text in options]
    if any("；" in text or ";" in text for text in joined):
        problems.append("选项里用了分号；每个选项写成一句话：做法，后面最多跟一句简短理由")
    hits = sorted({phrase for text in joined for phrase in OPTION_FORMULA_PHRASES if phrase in text})
    if hits:
        problems.append(f"选项用了套话或书面腔（{'、'.join(hits)}）；换成学生自己会说的话")
    both = [text for text in joined if "既" in text and "也" in text]
    if both:
        problems.append("选项用了“既……也……”句式；换成自然的说法")
    return problems


def scenario_style_problems(scenario: str) -> list[str]:
    """Rule-based naturalness problems in one scenario text."""
    from sjt_system.evaluation.virtual_content_review import text_length_units
    text = str(scenario or "")
    problems = []
    if "；" in text or ";" in text:
        problems.append("情境用分号拼接短语；请改写成 1–3 个完整、自然的句子")
    if text_length_units(text) > SCENARIO_MAX_CHARS:
        problems.append(f"情境 {text_length_units(text)} 字，超过 {SCENARIO_MAX_CHARS} 字上限")
    hits = [phrase for phrase in FREE_CHOICE_PHRASES if phrase in text]
    hits = [phrase for phrase in hits if not any(phrase != other and phrase in other for other in hits)]
    if hits:
        problems.append(f"情境写了自由选择声明（{'、'.join(hits)}）；删去声明，只交代发生了什么")
    if "我" in _outside_quotes(text):
        problems.append("情境里出现了“我”；情境一律用“你”称呼作答者，只有引号里的人物原话可以用“我”")
    notes = [phrase for phrase in REVIEWER_NOTE_PHRASES if phrase in text]
    if notes:
        problems.append(f"情境写了给审题人看的说明（{'、'.join(notes)}）；删去")
    return problems
