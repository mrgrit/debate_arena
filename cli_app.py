# toron_v3.py  (FINAL: defaults in code + load_panels inside run + single tool call_panel)
# -----------------------------------------------------------------------------
# 특징
# - export 없이 실행 가능: main에서 기본값(os.environ.setdefault) 주입
# - 환경변수로 주면 기본값을 덮어씀(setdefault라 안전)
# - 패널 수(N) 가변: PANELS_JSON으로 조정
# - Phase 비율 가변: PHASE_RATIOS로 조정
# - 고정 페르소나 없음: 모든 패널 기본 balanced, 심판이 매 라운드 역할(role) 동적 배정
# - 단일 @tool: call_panel(panel_id, prompt)
# - 심판이 상태 업데이트(요약/쟁점/합의/미해결)도 JSON으로 갱신
#
# 설치:
#   python3 -m pip install -U langchain-core langchain-ollama
#   (외부 모델 쓰면) python3 -m pip install -U langchain-openai
#
# 실행:
#   python3 toron_v3.py
# -----------------------------------------------------------------------------

import os
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage


# -------------------------
# Helpers
# -------------------------
def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


def _clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: int(max_chars * 0.7)] + "\n...\n" + s[-int(max_chars * 0.25):]


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _force_json_only(llm, messages: List[Any], max_repair: int = 3) -> Dict[str, Any]:
    last = ""
    for _ in range(max_repair):
        out = llm.invoke(messages).content
        last = out
        obj = _extract_json_object(out)
        if obj is not None:
            return obj
        messages.append(
            HumanMessage(
                content=(
                    "이전 출력은 올바른 JSON 오브젝트가 아니다.\n"
                    "규칙: 오직 JSON 하나만 출력.\n"
                    "지금 즉시 올바른 JSON만 다시 출력해라."
                )
            )
        )
    repair = (
        "아래 TEXT를 올바른 JSON 오브젝트 1개로만 변환해라. 추가 텍스트 금지.\n\n"
        f"TEXT:\n{last}\n"
    )
    out2 = llm.invoke(messages + [HumanMessage(content=repair)]).content
    obj2 = _extract_json_object(out2)
    if obj2 is not None:
        return obj2
    raise ValueError("Judge가 끝까지 JSON 오브젝트를 출력하지 못했습니다. (JUDGE_MODEL 교체 권장)")


# -------------------------
# Config parsing
# -------------------------
def parse_ratios(s: str) -> List[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    ratios = [float(p) for p in parts] if parts else [0.10, 0.60, 0.20, 0.10]
    total = sum(ratios)
    if total <= 0:
        return [0.10, 0.60, 0.20, 0.10]
    return [r / total for r in ratios]


def build_phase_schedule(total_rounds: int, ratios: List[float]) -> List[int]:
    ratios = (ratios + [0.0, 0.0, 0.0, 0.0])[:4]
    counts = [round(total_rounds * r) for r in ratios]
    diff = total_rounds - sum(counts)
    # 반올림 오차는 Phase2에 흡수
    counts[1] += diff

    phase_of_round = [0]
    for phase_idx, c in enumerate(counts, start=1):
        phase_of_round += [phase_idx] * max(0, c)
    if len(phase_of_round) < total_rounds + 1:
        phase_of_round += [4] * ((total_rounds + 1) - len(phase_of_round))
    return phase_of_round[: total_rounds + 1]


# -------------------------
# LLM factory
# -------------------------
def get_chat_model(
    provider: str,
    model: str,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    provider = (provider or "ollama").lower().strip()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        if base_url:
            try:
                return ChatOllama(model=model, temperature=temperature, base_url=base_url)
            except TypeError:
                return ChatOllama(model=model, temperature=temperature)
        return ChatOllama(model=model, temperature=temperature)

    if provider in ("openai", "openai_compatible", "openai-compatible"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url or None,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    raise ValueError(f"unknown provider {provider}")


# -------------------------
# Panel registry
# -------------------------
@dataclass
class PanelSpec:
    id: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


def load_panels() -> Dict[str, PanelSpec]:
    panels_json = _env("PANELS_JSON", "[]")
    arr = json.loads(panels_json)
    panels: Dict[str, PanelSpec] = {}
    for it in arr:
        pid = str(it.get("id", "")).strip()
        if not pid:
            continue
        model = str(it.get("model", "")).strip()
        if not model:
            continue
        panels[pid] = PanelSpec(
            id=pid,
            provider=it.get("provider", "ollama"),
            model=model,
            base_url=it.get("base_url"),
            api_key=it.get("api_key"),
        )
    if len(panels) < 2:
        raise RuntimeError("PANELS_JSON must contain at least 2 valid panels with id/model.")
    return panels


# ✅ 전역 PANELS는 run()에서 채움 (main의 기본값 주입이 먼저 먹도록)
PANELS: Dict[str, PanelSpec] = {}


# -------------------------
# Single Tool: call_panel(panel_id, prompt)
# -------------------------
@tool("call_panel", description="지정한 panel_id의 LLM 패널에게 prompt를 전달하고 답변 텍스트를 반환한다.")
def call_panel(panel_id: str, prompt: str) -> str:
    """Call a configured panel by id."""
    panel = PANELS.get(panel_id)
    if not panel:
        return f"ERROR: unknown panel_id={panel_id}. Available={sorted(PANELS.keys())}"
    llm = get_chat_model(panel.provider, panel.model, 0.2, panel.base_url, panel.api_key)
    return llm.invoke([HumanMessage(content=prompt)]).content


# -------------------------
# State
# -------------------------
def init_state(question: str) -> Dict[str, Any]:
    return {
        "question": question,
        "running_summary": "",
        "issue_board": [],
        "agreements": [],
        "unresolved": [],
        "history": [],
        "recent_window": [],
        "next_issue_id": 1,
        "last_roles": {},
    }


# -------------------------
# Phase meta
# -------------------------
PHASE_META = {
    1: {"name": "정의/가정/쟁점+초안", "goal": "용어 정의, 가정/제약, 쟁점 도출, 초안 결론 생성"},
    2: {"name": "반례/리스크/대안+공격", "goal": "반례/리스크 발굴, 대안 탐색, 서로 논리 공격/저격"},
    3: {"name": "수렴/정리", "goal": "합의 가능한 부분 수렴, 남은 쟁점만 정리"},
    4: {"name": "최종 다듬기", "goal": "최종 결론/권고안 문장 개선, 실행 가능한 권고안/체크리스트"},
}


# -------------------------
# Judge model getter
# -------------------------
def get_judge_llm():
    provider = _env("JUDGE_PROVIDER", _env("MODERATOR_PROVIDER", "ollama"))
    model = _env("JUDGE_MODEL", _env("MODERATOR_MODEL", "gemma3:4b"))
    base_url = _env("JUDGE_BASE_URL", _env("MODERATOR_BASE_URL", ""))
    api_key = _env("JUDGE_API_KEY", _env("MODERATOR_API_KEY", ""))
    return get_chat_model(provider, model, 0.2, base_url or None, api_key or None)


# -------------------------
# Dynamic role assignment
# -------------------------
ROLE_ASSIGNER_SYSTEM = """너는 토론의 심판으로서 '이번 라운드 역할(role)'을 패널들에게 배정한다.
중요: 특정 패널을 영구적으로 특정 성향으로 고정하지 말고, 편향을 줄이기 위해 라운드마다 역할을 분산/순환하라.

역할 목록(아래 중에서만 선택):
- balanced
- critic
- risk
- innovator
- fact_check
- synthesizer

규칙:
- Phase 2에서는 가능한 한 다양한 역할을 배정(중복 최소화).
- Phase 1에서는 balanced와 fact_check 중심.
- Phase 3에서는 synthesizer와 balanced 중심.
- Phase 4에서는 synthesizer 중심 + 필요시 balanced.
- 이전 라운드 역할(last_roles)을 참고해 같은 역할이 한 패널에 계속 고정되지 않도록 바꿔라.
- 결과는 오직 JSON 하나만 출력.

JSON 스키마:
{
  "roles": {"PANEL_ID": "role", "...": "role"},
  "round_goal_brief": "이번 라운드에 패널들이 특히 주의할 점(1~2문장)"
}
""".strip()


def _fallback_roles(panel_ids: List[str], phase: int, last_roles: Dict[str, str]) -> Dict[str, str]:
    if phase == 1:
        pool = ["balanced", "fact_check"]
    elif phase == 2:
        pool = ["critic", "risk", "innovator", "fact_check", "balanced"]
    elif phase == 3:
        pool = ["synthesizer", "balanced", "fact_check"]
    else:
        pool = ["synthesizer", "balanced"]

    roles = {}
    for i, pid in enumerate(panel_ids):
        candidate = pool[i % len(pool)]
        if last_roles.get(pid) == candidate:
            candidate = pool[(i + 1) % len(pool)]
        roles[pid] = candidate
    return roles


def assign_roles(state: Dict[str, Any], round_no: int, phase: int, panel_ids: List[str], ratios: List[float]) -> Dict[str, Any]:
    judge = get_judge_llm()
    summary_max = int(_env("SUMMARY_MAX_CHARS", "2000"))

    payload = {
        "round": round_no,
        "phase": phase,
        "phase_name": PHASE_META[phase]["name"],
        "phase_goal": PHASE_META[phase]["goal"],
        "phase_ratios": ratios,
        "panel_ids": panel_ids,
        "running_summary": _clamp_text(state["running_summary"], summary_max),
        "open_issues": [i for i in state["issue_board"] if i.get("status") == "open"],
        "recent_window": state["recent_window"],
        "last_roles": state.get("last_roles", {}),
    }

    messages = [
        SystemMessage(content=ROLE_ASSIGNER_SYSTEM),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    try:
        obj = _force_json_only(judge, messages, max_repair=3)
        roles = obj.get("roles", {})
        if not isinstance(roles, dict):
            raise ValueError("roles not dict")

        allowed = {"balanced", "critic", "risk", "innovator", "fact_check", "synthesizer"}
        fixed_roles = {}
        for pid in panel_ids:
            r = str(roles.get(pid, "balanced")).strip()
            fixed_roles[pid] = r if r in allowed else "balanced"

        round_goal = str(obj.get("round_goal_brief", "")).strip()
        if not round_goal:
            round_goal = "이번 라운드 목표를 수행하되, 과도한 편향 없이 근거/반례/대안을 균형있게 다뤄라."

        state["last_roles"] = fixed_roles
        return {"roles": fixed_roles, "round_goal_brief": round_goal}

    except Exception:
        roles = _fallback_roles(panel_ids, phase, state.get("last_roles", {}))
        state["last_roles"] = roles
        return {"roles": roles, "round_goal_brief": "(fallback) Phase 목표에 맞게 역할을 분산하여 수행하라."}


# -------------------------
# Announcement + task templates
# -------------------------
ROLE_GUIDE = {
    "balanced": "균형 관점. 장단점/근거/반례를 고르게 다루고 결론을 과도하게 단정하지 말 것.",
    "critic": "논리 결함/근거 부족/모순/비약을 지적하되, 인정할 점은 인정하고 대안을 함께 제시할 것.",
    "risk": "리스크/실패 시나리오/운영·보안·법적 위험을 중점적으로 발굴하고 완화책도 제시할 것.",
    "innovator": "대안/새 관점/새로운 옵션을 제시하되, 현실적 제약/트레이드오프를 함께 명시할 것.",
    "fact_check": "용어/전제/주장의 정확성을 점검. 모호하면 가정으로 분리하고 검증 방법 제시.",
    "synthesizer": "수렴/정리 역할. 합의/남은 쟁점을 분리하고 실행 가능한 결론으로 묶기.",
}


def build_announcement(round_no: int, phase: int, total_rounds: int, ratios: List[float], round_goal_brief: str) -> str:
    meta = PHASE_META[phase]
    return f"""[ROUND {round_no}/{total_rounds}] [PHASE {phase}: {meta['name']}]
PHASE 비율(가변): {", ".join([f"{r:.2f}" for r in ratios])}
이번 PHASE 목표: {meta['goal']}
이번 라운드 추가 지시: {round_goal_brief}

공통 규칙:
- 최근 window의 다른 패널 발언을 읽고: 동조/비판/다른 관점/틈새지적/저격을 수행할 것
- 치우친 주장만 반복하지 말고, 타당한 부분은 인정 + 개선안을 제시할 것
""".strip()


def build_phase_task(phase: int) -> str:
    if phase == 1:
        return """임무(Phase1):
- 용어/정의 2개
- 가정/제약 2개
- 핵심 쟁점 3개
- 초안 결론 1개 + 근거 3줄
출력 형식:
- 정의:
- 가정/제약:
- 쟁점:
- 초안 결론:
"""
    if phase == 2:
        return """임무(Phase2):
- 다른 패널 발언에서 약점/틈새 2개 지적(저격)
- 반례/리스크 2개
- 대안 1~2개 + 트레이드오프
출력 형식:
- 저격/지적:
- 반례/리스크:
- 대안(+트레이드오프):
"""
    if phase == 3:
        return """임무(Phase3):
- 합의 가능한 부분 2개
- 남은 쟁점 1~2개 + 해결 방향
출력 형식:
- 합의:
- 남은 쟁점/해결:
"""
    return """임무(Phase4):
- 최종 결론 문장 다듬기(더 명확/실행가능)
- 권고안 3개(체크리스트 형태)
출력 형식:
- 최종 문장:
- 권고안:
"""


def build_panel_prompt(panel_id: str, role: str, state: Dict[str, Any], announcement: str, phase: int) -> str:
    open_issues = [i for i in state["issue_board"] if i.get("status") == "open"]
    recent = state["recent_window"]
    summary_max = int(_env("SUMMARY_MAX_CHARS", "2000"))
    running_summary = _clamp_text(state["running_summary"], summary_max)

    return f"""{announcement}

[질문]
{state['question']}

[누적 요약(압축)]
{running_summary}

[열린 쟁점(open)]
{json.dumps(open_issues, ensure_ascii=False)}

[합의된 부분]
{json.dumps(state['agreements'], ensure_ascii=False)}

[미해결]
{json.dumps(state['unresolved'], ensure_ascii=False)}

[최근 라운드 발언(다른 패널 포함)]
{json.dumps(recent, ensure_ascii=False)}

[너의 이번 라운드 역할]
{role}
역할 가이드: {ROLE_GUIDE.get(role, ROLE_GUIDE["balanced"])}

{build_phase_task(phase)}
""".strip()


# -------------------------
# State updater / Final judge
# -------------------------
STATE_UPDATER_SYSTEM = """너는 토론 진행을 기록/정리하는 심판 보조(상태 업데이트)다.
입력으로 이전 상태 요약/쟁점/합의/미해결과 이번 라운드 패널 발언이 주어진다.
임무:
1) 누적 요약(running_summary)을 업데이트(짧고 핵심만, 반복 제거).
2) 열린 쟁점(issue)을 추가/수정/해결(resolved) 처리.
3) 합의(agreements)와 미해결(unresolved)을 업데이트.

오직 JSON 하나만 출력(추가 텍스트 금지).

JSON 스키마:
{
  "running_summary": "string (짧게)",
  "new_issues": [{"topic":"...","notes":"..."}],
  "resolve_issues": [issue_id_int],
  "update_issue_notes": [{"id": issue_id_int, "notes":"..."}],
  "agreements_add": ["..."],
  "agreements_remove": ["..."],
  "unresolved_add": ["..."],
  "unresolved_remove": ["..."]
}
""".strip()


FINAL_JUDGE_SYSTEM = """너는 토론의 최종 심판(모더레이터)이다.
입력으로 질문과 최종 상태(요약/쟁점/합의/미해결) 그리고 최근 핵심 발언이 주어진다.
임무:
1) 최종 결론(한국어)
2) 핵심 근거 3~5개
3) 가정/불확실성 명시
4) 패널 기여도를 공정하게 100점 만점으로 배분(합 100, 정수)
5) 점수 근거를 짧게 설명

오직 JSON 하나만 출력(추가 텍스트 금지).

JSON 스키마:
{
  "final_answer": "string",
  "key_reasons": ["..."],
  "assumptions_or_uncertainties": ["..."],
  "scores": {"PANEL_ID": 0, "...": 0},
  "score_rationale": {"PANEL_ID": "짧은 근거", "...": "..."}
}
""".strip()


def _as_list(x) -> List[Any]:
    return x if isinstance(x, list) else []


def update_state_with_round(state: Dict[str, Any], round_no: int, phase: int, announcement: str,
                            roles: Dict[str, str], panel_outputs: Dict[str, str]) -> None:
    state["history"].append({
        "round": round_no,
        "phase": phase,
        "announcement": announcement,
        "roles": roles,
        "outputs": panel_outputs,
    })

    N = int(_env("RECENT_WINDOW_ROUNDS", "2"))
    compact = {
        "round": round_no,
        "phase": phase,
        "roles": roles,
        "outputs": {pid: _clamp_text(panel_outputs[pid], 420) for pid in panel_outputs},
    }
    state["recent_window"].append(compact)
    state["recent_window"] = state["recent_window"][-N:]

    judge = get_judge_llm()
    summary_max = int(_env("SUMMARY_MAX_CHARS", "2000"))

    payload = {
        "question": state["question"],
        "prev_running_summary": _clamp_text(state["running_summary"], summary_max),
        "open_issues": [i for i in state["issue_board"] if i.get("status") == "open"],
        "agreements": state["agreements"],
        "unresolved": state["unresolved"],
        "round": round_no,
        "phase": phase,
        "announcement": announcement,
        "roles": roles,
        "panel_outputs": {k: _clamp_text(v, 1200) for k, v in panel_outputs.items()},
    }

    upd = _force_json_only(
        judge,
        [SystemMessage(content=STATE_UPDATER_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
        max_repair=3
    )

    new_summary = str(upd.get("running_summary", "")).strip()
    if new_summary:
        state["running_summary"] = _clamp_text(new_summary, summary_max)

    # new issues
    for ni in _as_list(upd.get("new_issues")):
        topic = str(ni.get("topic", "")).strip()
        notes = str(ni.get("notes", "")).strip()
        if not topic:
            continue
        state["issue_board"].append({
            "id": state["next_issue_id"],
            "topic": topic,
            "status": "open",
            "notes": notes,
        })
        state["next_issue_id"] += 1

    # resolve issues
    resolve_ids = set()
    for rid in _as_list(upd.get("resolve_issues")):
        try:
            resolve_ids.add(int(rid))
        except Exception:
            continue
    for issue in state["issue_board"]:
        if issue.get("id") in resolve_ids:
            issue["status"] = "resolved"

    # update issue notes
    for upn in _as_list(upd.get("update_issue_notes")):
        try:
            iid = int(upn.get("id"))
        except Exception:
            continue
        notes = str(upn.get("notes", "")).strip()
        if not notes:
            continue
        for issue in state["issue_board"]:
            if issue.get("id") == iid:
                issue["notes"] = notes
                break

    # agreements add/remove
    agr_add = [str(x).strip() for x in _as_list(upd.get("agreements_add")) if str(x).strip()]
    agr_rm  = set(str(x).strip() for x in _as_list(upd.get("agreements_remove")) if str(x).strip())
    state["agreements"] = [a for a in state["agreements"] if a not in agr_rm]
    for a in agr_add:
        if a not in state["agreements"]:
            state["agreements"].append(a)

    # unresolved add/remove
    unr_add = [str(x).strip() for x in _as_list(upd.get("unresolved_add")) if str(x).strip()]
    unr_rm  = set(str(x).strip() for x in _as_list(upd.get("unresolved_remove")) if str(x).strip())
    state["unresolved"] = [u for u in state["unresolved"] if u not in unr_rm]
    for u in unr_add:
        if u not in state["unresolved"]:
            state["unresolved"].append(u)


def _fix_scores_to_100(scores: Dict[str, Any], panel_ids: List[str]) -> Dict[str, int]:
    vals = {}
    for pid in panel_ids:
        v = scores.get(pid, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        vals[pid] = max(0, min(100, v))

    total = sum(vals.values())
    if total == 100:
        return vals
    if total == 0:
        base = 100 // len(panel_ids)
        rem = 100 - base * len(panel_ids)
        out = {pid: base for pid in panel_ids}
        for pid in panel_ids[:rem]:
            out[pid] += 1
        return out

    scaled = {pid: round(vals[pid] * 100 / total) for pid in panel_ids}
    diff = 100 - sum(scaled.values())
    order = sorted(panel_ids, key=lambda p: scaled[p], reverse=True)
    i = 0
    while diff != 0:
        pid = order[i % len(order)]
        if diff > 0:
            if scaled[pid] < 100:
                scaled[pid] += 1
                diff -= 1
        else:
            if scaled[pid] > 0:
                scaled[pid] -= 1
                diff += 1
        i += 1
    return {pid: int(scaled[pid]) for pid in panel_ids}


def judge_final(state: Dict[str, Any], panel_ids: List[str]) -> Dict[str, Any]:
    judge = get_judge_llm()
    summary_max = int(_env("SUMMARY_MAX_CHARS", "2000"))

    payload = {
        "question": state["question"],
        "running_summary": _clamp_text(state["running_summary"], summary_max),
        "open_issues": [i for i in state["issue_board"] if i.get("status") == "open"],
        "agreements": state["agreements"],
        "unresolved": state["unresolved"],
        "recent_window": state["recent_window"],
        "panel_ids": panel_ids,
    }

    out = _force_json_only(
        judge,
        [SystemMessage(content=FINAL_JUDGE_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
        max_repair=3
    )

    scores = out.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    out["scores"] = _fix_scores_to_100(scores, panel_ids)

    rationale = out.get("score_rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    for pid in panel_ids:
        rationale.setdefault(pid, "")
    out["score_rationale"] = rationale

    out["debate_trace"] = {
        "issue_board": state["issue_board"],
        "agreements": state["agreements"],
        "unresolved": state["unresolved"],
        "recent_window": state["recent_window"],
    }
    return out


# -------------------------
# Main loop
# -------------------------
def run(question: str):
    global PANELS
    PANELS = load_panels()

    total_rounds = int(_env("TOTAL_ROUNDS", "50"))
    ratios = parse_ratios(_env("PHASE_RATIOS", "0.10,0.60,0.20,0.10"))
    phase_of_round = build_phase_schedule(total_rounds, ratios)

    state = init_state(question)
    panel_ids = sorted(PANELS.keys())

    print(f"[config] total_rounds={total_rounds} phase_ratios={ratios} panels={panel_ids}")
    print("[note] 고정 persona 없음. 매 라운드 역할(role)을 심판이 분산/순환 배정(편향 완화).")

    for r in range(1, total_rounds + 1):
        phase = phase_of_round[r]

        role_pack = assign_roles(state, r, phase, panel_ids, ratios)
        roles = role_pack["roles"]
        round_goal_brief = role_pack["round_goal_brief"]

        announcement = build_announcement(r, phase, total_rounds, ratios, round_goal_brief)

        print("\n" + "=" * 95)
        print(announcement)
        print("[ROLE ASSIGNMENT]", roles)
        print("=" * 95)

        panel_outputs: Dict[str, str] = {}
        for pid in panel_ids:
            role = roles.get(pid, "balanced")
            prompt = build_panel_prompt(pid, role, state, announcement, phase)
            out = call_panel.invoke({"panel_id": pid, "prompt": prompt})
            panel_outputs[pid] = out

            panel_model = PANELS[pid].model
            print(f"\n--- Round {r} Panel {pid} ({panel_model}) role={role} ---")
            print(_clamp_text(out, 1400))

        update_state_with_round(state, r, phase, announcement, roles, panel_outputs)

        print("\n[STATE] summary (truncated):")
        print(_clamp_text(state["running_summary"], 900))
        print("[STATE] open issues:", len([i for i in state["issue_board"] if i.get("status") == "open"]))
        print("[STATE] agreements:", len(state["agreements"]), "unresolved:", len(state["unresolved"]))

    final = judge_final(state, panel_ids)
    print("\n\n=== FINAL JSON ===")
    print(json.dumps(final, ensure_ascii=False, indent=2))


# -------------------------
# Defaults in code (no export needed)
# -------------------------
def _set_default_env():
    os.environ.setdefault("TOTAL_ROUNDS", "50")
    os.environ.setdefault("PHASE_RATIOS", "0.10,0.60,0.20,0.10")
    os.environ.setdefault("RECENT_WINDOW_ROUNDS", "2")
    os.environ.setdefault("SUMMARY_MAX_CHARS", "2000")

    # 기본 패널(원하면 여기만 바꿔서 바로 실행 가능)
    os.environ.setdefault(
        "PANELS_JSON",
        json.dumps(
            [
                {"id": "A", "provider": "ollama", "model": "llama3.1:8b"},
                {"id": "B", "provider": "ollama", "model": "ministral-3:8b"},
                {"id": "C", "provider": "ollama", "model": "gemma3:12b"},
            ],
            ensure_ascii=False,
        ),
    )

    # 심판 모델 기본값
    os.environ.setdefault("JUDGE_PROVIDER", "ollama")
    os.environ.setdefault("JUDGE_MODEL", "gpt-oss:20b")


if __name__ == "__main__":
    _set_default_env()
    q = input("질문> ").strip()
    run(q)
