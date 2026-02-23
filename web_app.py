# debate_arena_popup_graph_svg_v3_stats_restore.py
# -----------------------------------------------------------------------------
# FIX: "out/in attack/support/counter/nudge" 패널 통계 박스(좌측 하단) 복구
#
# 이번 버전 포함 사항
# - SVG 그래프(기본) + 클릭 시 팝업 확대
# - 엣지 라벨 겹침 완화(충돌 회피 + 줄바꿈/축약)
# - 패널별 상호작용 통계(out/in) UI 복구 (좌측 카드 하단)
# - 노드 클릭 = 필터(Feed에서 해당 패널 발언만 강조/노출)
# - 점수 공정성 보정(min_each + judge/activity 재분배) 유지
#
# Requirements:
#   pip install fastapi uvicorn langchain-core langchain-ollama
#
# Run:
#   python3 debate_arena_popup_graph_svg_v3_stats_restore.py
#   http://localhost:8000
# -----------------------------------------------------------------------------

import json
import re
import uuid
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# Utilities
# =========================
def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: int(max_chars * 0.7)] + "\n...\n" + s[-int(max_chars * 0.25) :]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
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


async def force_json_only(llm, messages: List[Any], max_repair: int = 3) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    last = ""
    for _ in range(max_repair):
        out = await loop.run_in_executor(None, lambda: llm.invoke(messages).content)
        last = out
        obj = extract_json_object(out)
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
    out2 = await loop.run_in_executor(None, lambda: llm.invoke(messages + [HumanMessage(content=repair)]).content)
    obj2 = extract_json_object(out2)
    if obj2 is not None:
        return obj2
    raise ValueError("LLM이 끝까지 JSON 오브젝트를 출력하지 못했습니다. 모델을 교체하세요.")


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
    counts[1] += diff
    phase_of_round = [0]
    for phase_idx, c in enumerate(counts, start=1):
        phase_of_round += [phase_idx] * max(0, c)
    if len(phase_of_round) < total_rounds + 1:
        phase_of_round += [4] * ((total_rounds + 1) - len(phase_of_round))
    return phase_of_round[: total_rounds + 1]


def validate_ollama_host(host: str) -> str:
    host = (host or "").strip()
    if not host:
        raise ValueError("host is empty")
    u = urlparse(host)
    if u.scheme not in ("http", "https"):
        raise ValueError("host must start with http:// or https://")
    if not u.netloc:
        raise ValueError("host missing netloc (e.g., http://1.2.3.4:11434)")
    return host.rstrip("/")


def http_get_json(url: str, timeout: float = 8.0) -> Dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "DebateArena/SVG"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)


# =========================
# LangChain / Ollama client
# =========================
def get_ollama_chat(model: str, host: str, temperature: float = 0.2):
    from langchain_ollama import ChatOllama

    try:
        return ChatOllama(model=model, temperature=temperature, base_url=host)
    except TypeError:
        import os

        os.environ["OLLAMA_HOST"] = host
        return ChatOllama(model=model, temperature=temperature)


# =========================
# Prompts
# =========================
PHASE_META = {
    1: {"name": "정의/가정/쟁점+초안", "goal": "용어 정의, 가정/제약, 쟁점 도출, 초안 결론 생성"},
    2: {"name": "반례/리스크/대안+공격", "goal": "반례/리스크 발굴, 대안 탐색, 서로 논리 공격/저격"},
    3: {"name": "수렴/정리", "goal": "합의 가능한 부분 수렴, 남은 쟁점만 정리"},
    4: {"name": "최종 다듬기", "goal": "최종 결론/권고안 문장 개선, 실행 가능한 권고안/체크리스트"},
}

ROLE_GUIDE = {
    "balanced": "균형. 장단점/근거/반례를 고르게 다루고 과도한 단정 금지.",
    "critic": "논리 결함/근거 부족/모순 지적 + 인정할 점 인정 + 개선안 제시.",
    "risk": "리스크/실패 시나리오/운영·보안·법적 위험 발굴 + 완화책 제시.",
    "innovator": "대안/새 관점/옵션 제시 + 현실 제약/트레이드오프 포함.",
    "fact_check": "전제/용어/주장 정확성 점검. 모호하면 가정 분리 + 검증 방법 제시.",
    "synthesizer": "수렴/정리. 합의/남은 쟁점 분리 + 실행 가능한 결론으로 묶기.",
}

ROLE_ASSIGNER_SYSTEM = """너는 토론 심판이다. 이번 라운드에 패널별 역할(role)을 배정하라.
역할 목록:
- balanced, critic, risk, innovator, fact_check, synthesizer

규칙:
- 편향을 줄이기 위해 역할을 분산/순환하라(영구 고정 금지).
- Phase2에서는 다양성 극대화(중복 최소화).
- Phase1: balanced/fact_check 중심, Phase3: synthesizer/balanced 중심, Phase4: synthesizer 중심.
- 결과는 JSON만 출력.

JSON:
{"roles":{"PANEL_ID":"role"},"round_goal_brief":"1~2문장"}
""".strip()

UTTERANCE_SUMMARY_SYSTEM = """너는 실시간 중계용 '심판 요약기'다.
입력: 이번 발언, 최근 상태.
임무:
1) 2~3문장 요약(summary)
2) 관계(edges): attack/support/counter/nudge (가능할 때만)
JSON만 출력.

{"summary":"...","edges":[{"from":"P1","to":"P2","type":"attack|support|counter|nudge","label":"짧게"}]}
""".strip()

STATE_UPDATER_SYSTEM = """너는 토론 상태 업데이트 담당이다.
입력: 이전 상태 + 이번 라운드 발언.
임무:
1) running_summary 업데이트(짧게)
2) new_issues 추가
3) agreements/unresolved 업데이트
JSON만 출력.

{
  "running_summary":"...",
  "new_issues":[{"topic":"...","notes":"..."}],
  "agreements_add":["..."],
  "agreements_remove":["..."],
  "unresolved_add":["..."],
  "unresolved_remove":["..."]
}
""".strip()

FINAL_JUDGE_SYSTEM = """너는 토론의 최종 심판이다.
입력: 질문 + 최종 상태 + activity.
임무:
- 최종 결론(한국어)
- 근거 3~5
- 불확실성
- 점수 합 100(정수)
- 정상 발언한 패널을 0점으로 주지 말 것(특별 결격 없으면 최소 5점 이상 권장)
- 점수 근거(패널별 1~2문장)
오직 JSON만.

{
  "final_answer":"...",
  "key_reasons":["..."],
  "assumptions_or_uncertainties":["..."],
  "scores":{"P1":34,"P2":33,"P3":33},
  "score_rationale":{"P1":"...","P2":"...","P3":"..."}
}
""".strip()


# =========================
# Scoring fairness
# =========================
def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def normalize_scores_with_floor(
    judge_scores: Dict[str, Any],
    panel_ids: List[str],
    activity: Dict[str, Dict[str, Any]],
    min_each: int = 5,
) -> Dict[str, int]:
    participated = []
    for pid in panel_ids:
        a = activity.get(pid, {})
        if _safe_int(a.get("utterances", 0)) > 0 and _safe_int(a.get("chars", 0)) > 20:
            participated.append(pid)
    enforce_floor = (len(participated) == len(panel_ids))

    js = {pid: max(0, min(100, _safe_int(judge_scores.get(pid, 0)))) for pid in panel_ids}

    base = {pid: 0 for pid in panel_ids}
    remaining = 100
    if enforce_floor:
        floor_total = min_each * len(panel_ids)
        if floor_total >= 100:
            eq = 100 // len(panel_ids)
            rem = 100 - eq * len(panel_ids)
            out = {pid: eq for pid in panel_ids}
            for pid in panel_ids[:rem]:
                out[pid] += 1
            return out
        for pid in panel_ids:
            base[pid] = min_each
        remaining = 100 - floor_total

    aw = {}
    for pid in panel_ids:
        a = activity.get(pid, {})
        chars = max(0, _safe_int(a.get("chars", 0)))
        edges = max(0, _safe_int(a.get("edges", 0)))
        aw[pid] = chars + 250 * edges

    sum_js = sum(js.values())
    sum_aw = sum(aw.values())
    w = {}
    for pid in panel_ids:
        p_js = (js[pid] / sum_js) if sum_js > 0 else (1.0 / len(panel_ids))
        p_aw = (aw[pid] / sum_aw) if sum_aw > 0 else (1.0 / len(panel_ids))
        w[pid] = 0.7 * p_js + 0.3 * p_aw

    alloc = {pid: int(round(remaining * w[pid])) for pid in panel_ids}
    diff = remaining - sum(alloc.values())
    order = sorted(panel_ids, key=lambda p: w[p], reverse=True)
    i = 0
    while diff != 0:
        pid = order[i % len(order)]
        if diff > 0:
            alloc[pid] += 1
            diff -= 1
        else:
            if alloc[pid] > 0:
                alloc[pid] -= 1
                diff += 1
        i += 1

    out = {pid: base[pid] + alloc[pid] for pid in panel_ids}
    drift = 100 - sum(out.values())
    if drift != 0:
        out[order[0]] = max(0, min(100, out[order[0]] + drift))
    drift = 100 - sum(out.values())
    if drift != 0:
        out[order[0]] = max(0, min(100, out[order[0]] + drift))
    return out


# =========================
# WebSocket Hub
# =========================
class Hub:
    def __init__(self):
        self.clients: Dict[str, Set[WebSocket]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.ready: Dict[str, asyncio.Event] = {}

    def ensure_ready(self, session_id: str) -> asyncio.Event:
        if session_id not in self.ready:
            self.ready[session_id] = asyncio.Event()
        return self.ready[session_id]

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.clients.setdefault(session_id, set()).add(ws)
        self.ensure_ready(session_id).set()

    def disconnect(self, session_id: str, ws: WebSocket):
        if session_id in self.clients and ws in self.clients[session_id]:
            self.clients[session_id].remove(ws)

    async def broadcast(self, session_id: str, payload: Dict[str, Any]):
        dead = []
        for ws in list(self.clients.get(session_id, set())):
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_id, ws)


hub = Hub()


# =========================
# Debate Engine
# =========================
@dataclass
class SessionConfig:
    host: str
    judge_model: str
    panel_models: Dict[str, str]
    total_rounds: int
    phase_ratios: List[float]
    summary_max_chars: int = 2000
    recent_window_rounds: int = 2
    min_score_each: int = 5


SESSION_CFG: Dict[str, SessionConfig] = {}


def init_state(question: str, panel_ids: List[str]) -> Dict[str, Any]:
    return {
        "question": question,
        "running_summary": "",
        "issue_board": [],
        "agreements": [],
        "unresolved": [],
        "recent_window": [],
        "next_issue_id": 1,
        "last_roles": {},
        "edges": [],
        "activity": {pid: {"utterances": 0, "chars": 0, "edges": 0} for pid in panel_ids},
    }


def fallback_roles(panel_ids: List[str], phase: int, last_roles: Dict[str, str]) -> Dict[str, str]:
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


def build_phase_task(phase: int) -> str:
    if phase == 1:
        return """임무(Phase1):
- 용어/정의 2개
- 가정/제약 2개
- 핵심 쟁점 3개
- 초안 결론 1개 + 근거 3줄
출력:
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
출력:
- 저격/지적:
- 반례/리스크:
- 대안(+트레이드오프):
"""
    if phase == 3:
        return """임무(Phase3):
- 합의 가능한 부분 2개
- 남은 쟁점 1~2개 + 해결 방향
출력:
- 합의:
- 남은 쟁점/해결:
"""
    return """임무(Phase4):
- 최종 결론 문장 다듬기(더 명확/실행가능)
- 권고안 3개(체크리스트)
출력:
- 최종 문장:
- 권고안:
"""


def build_announcement(round_no: int, phase: int, total_rounds: int, ratios: List[float], round_goal_brief: str) -> str:
    meta = PHASE_META[phase]
    return f"""[ROUND {round_no}/{total_rounds}] [PHASE {phase}: {meta['name']}]
PHASE 비율: {", ".join([f"{r:.2f}" for r in ratios])}
PHASE 목표: {meta['goal']}
이번 라운드 추가 지시: {round_goal_brief}

공통 규칙:
- 최근 window의 다른 패널 발언을 읽고: 공감/반박/저격/틈새 지적을 수행
- 과도한 편향/단정 금지: 인정할 점은 인정 + 개선안 제시
""".strip()


def build_panel_prompt(panel_id: str, role: str, state: Dict[str, Any], announcement: str, phase: int, summary_max: int) -> str:
    open_issues = [i for i in state["issue_board"] if i.get("status") == "open"]
    return f"""{announcement}

[질문]
{state['question']}

[누적 요약(압축)]
{clamp_text(state["running_summary"], summary_max)}

[열린 쟁점(open)]
{json.dumps(open_issues, ensure_ascii=False)}

[합의된 부분]
{json.dumps(state["agreements"], ensure_ascii=False)}

[미해결]
{json.dumps(state["unresolved"], ensure_ascii=False)}

[최근 라운드 발언(상호참조)]
{json.dumps(state["recent_window"], ensure_ascii=False)}

[너의 이번 라운드 역할]
{role}
역할 가이드: {ROLE_GUIDE.get(role, ROLE_GUIDE["balanced"])}

{build_phase_task(phase)}
""".strip()


async def assign_roles(cfg: SessionConfig, state: Dict[str, Any], round_no: int, phase: int, panel_ids: List[str]) -> Dict[str, Any]:
    judge = get_ollama_chat(cfg.judge_model, cfg.host, temperature=0.2)
    payload = {
        "round": round_no,
        "phase": phase,
        "phase_name": PHASE_META[phase]["name"],
        "phase_goal": PHASE_META[phase]["goal"],
        "phase_ratios": cfg.phase_ratios,
        "panel_ids": panel_ids,
        "running_summary": clamp_text(state["running_summary"], 1200),
        "open_issues": [i for i in state["issue_board"] if i.get("status") == "open"],
        "recent_window": state["recent_window"],
        "last_roles": state.get("last_roles", {}),
    }
    try:
        obj = await force_json_only(
            judge,
            [SystemMessage(content=ROLE_ASSIGNER_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
            max_repair=3,
        )
        roles = obj.get("roles", {})
        allowed = set(ROLE_GUIDE.keys())
        fixed = {}
        for pid in panel_ids:
            r = str(roles.get(pid, "balanced")).strip()
            fixed[pid] = r if r in allowed else "balanced"
        brief = str(obj.get("round_goal_brief", "")).strip() or "이번 라운드는 Phase 목표에 맞춰 근거/반례/대안을 균형있게."
        state["last_roles"] = fixed
        return {"roles": fixed, "round_goal_brief": brief}
    except Exception:
        fixed = fallback_roles(panel_ids, phase, state.get("last_roles", {}))
        state["last_roles"] = fixed
        return {"roles": fixed, "round_goal_brief": "(fallback) Phase 목표에 맞게 역할을 분산하여 수행하라."}


async def utterance_mini_summary(cfg: SessionConfig, state: Dict[str, Any], panel_id: str, role: str, utterance: str, panel_ids: List[str]) -> Dict[str, Any]:
    judge = get_ollama_chat(cfg.judge_model, cfg.host, temperature=0.2)
    payload = {
        "panel_id": panel_id,
        "role": role,
        "utterance": clamp_text(utterance, 1200),
        "running_summary": clamp_text(state["running_summary"], 1200),
        "recent_window": state["recent_window"],
        "panel_ids": panel_ids,
    }
    obj = await force_json_only(
        judge,
        [SystemMessage(content=UTTERANCE_SUMMARY_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
        max_repair=3,
    )
    edges = obj.get("edges", [])
    if not isinstance(edges, list):
        edges = []
    fixed_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        frm = str(e.get("from", panel_id)).strip()
        to = str(e.get("to", "")).strip()
        typ = str(e.get("type", "nudge")).strip()
        lab = str(e.get("label", "")).strip()[:80]
        if frm and to and frm != to:
            fixed_edges.append({"from": frm, "to": to, "type": typ, "label": lab})
    return {"summary": str(obj.get("summary", "")).strip(), "edges": fixed_edges}


async def update_state_round(cfg: SessionConfig, state: Dict[str, Any], round_no: int, phase: int, announcement: str, roles: Dict[str, str], round_outputs: Dict[str, str]):
    judge = get_ollama_chat(cfg.judge_model, cfg.host, temperature=0.2)
    payload = {
        "question": state["question"],
        "prev_running_summary": clamp_text(state["running_summary"], cfg.summary_max_chars),
        "agreements": state["agreements"],
        "unresolved": state["unresolved"],
        "round": round_no,
        "phase": phase,
        "announcement": announcement,
        "roles": roles,
        "panel_outputs": {k: clamp_text(v, 1200) for k, v in round_outputs.items()},
    }
    upd = await force_json_only(
        judge,
        [SystemMessage(content=STATE_UPDATER_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
        max_repair=3,
    )

    new_summary = str(upd.get("running_summary", "")).strip()
    if new_summary:
        state["running_summary"] = clamp_text(new_summary, cfg.summary_max_chars)

    new_issues = upd.get("new_issues", [])
    if isinstance(new_issues, list):
        for ni in new_issues:
            topic = str(ni.get("topic", "")).strip() if isinstance(ni, dict) else str(ni).strip()
            notes = str(ni.get("notes", "")).strip() if isinstance(ni, dict) else ""
            if topic:
                state["issue_board"].append({"id": state["next_issue_id"], "topic": topic, "status": "open", "notes": notes})
                state["next_issue_id"] += 1

    def apply_list_changes(lst: List[str], add, rm):
        add = [str(x).strip() for x in add if str(x).strip()] if isinstance(add, list) else []
        rm = set(str(x).strip() for x in rm if str(x).strip()) if isinstance(rm, list) else set()
        lst = [x for x in lst if x not in rm]
        for x in add:
            if x not in lst:
                lst.append(x)
        return lst

    state["agreements"] = apply_list_changes(state["agreements"], upd.get("agreements_add", []), upd.get("agreements_remove", []))
    state["unresolved"] = apply_list_changes(state["unresolved"], upd.get("unresolved_add", []), upd.get("unresolved_remove", []))

    compact = {
        "round": round_no,
        "phase": phase,
        "roles": roles,
        "outputs": {pid: clamp_text(round_outputs[pid], 320) for pid in round_outputs},
    }
    state["recent_window"].append(compact)
    state["recent_window"] = state["recent_window"][-cfg.recent_window_rounds :]


async def final_judge(cfg: SessionConfig, state: Dict[str, Any], panel_ids: List[str]) -> Dict[str, Any]:
    judge = get_ollama_chat(cfg.judge_model, cfg.host, temperature=0.2)
    payload = {
        "question": state["question"],
        "running_summary": clamp_text(state["running_summary"], cfg.summary_max_chars),
        "agreements": state["agreements"],
        "unresolved": state["unresolved"],
        "recent_window": state["recent_window"],
        "panel_ids": panel_ids,
        "activity": state.get("activity", {}),
        "min_score_each": cfg.min_score_each,
    }
    out = await force_json_only(
        judge,
        [SystemMessage(content=FINAL_JUDGE_SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
        max_repair=3,
    )

    judge_scores = out.get("scores", {}) if isinstance(out.get("scores", {}), dict) else {}
    activity = state.get("activity", {}) if isinstance(state.get("activity", {}), dict) else {}
    out["scores"] = normalize_scores_with_floor(judge_scores, panel_ids, activity, min_each=cfg.min_score_each)

    rationale = out.get("score_rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    for pid in panel_ids:
        if not str(rationale.get(pid, "")).strip():
            a = activity.get(pid, {})
            rationale[pid] = f"발언 {int(a.get('utterances',0))}회, 상호작용 {int(a.get('edges',0))}회 기여 반영."
    out["score_rationale"] = rationale
    return out


async def debate_stream(session_id: str, question: str):
    try:
        try:
            await asyncio.wait_for(hub.ensure_ready(session_id).wait(), timeout=15.0)
        except asyncio.TimeoutError:
            return

        cfg = SESSION_CFG.get(session_id)
        if not cfg:
            await hub.broadcast(session_id, {"type": "error", "message": "missing session config"})
            return

        ratios = cfg.phase_ratios
        phase_of_round = build_phase_schedule(cfg.total_rounds, ratios)
        panel_ids = list(cfg.panel_models.keys())

        await hub.broadcast(
            session_id,
            {
                "type": "init",
                "session_id": session_id,
                "question": question,
                "host": cfg.host,
                "judge_model": cfg.judge_model,
                "panels": [{"id": pid, "model": cfg.panel_models[pid]} for pid in panel_ids],
                "total_rounds": cfg.total_rounds,
                "phase_ratios": ratios,
                "phase_meta": PHASE_META,
            },
        )

        state = init_state(question, panel_ids)

        for r in range(1, cfg.total_rounds + 1):
            phase = phase_of_round[r]
            role_pack = await assign_roles(cfg, state, r, phase, panel_ids)
            roles = role_pack["roles"]
            announcement = build_announcement(r, phase, cfg.total_rounds, ratios, role_pack["round_goal_brief"])

            await hub.broadcast(
                session_id,
                {
                    "type": "round_start",
                    "round": r,
                    "phase": phase,
                    "phase_name": PHASE_META[phase]["name"],
                    "phase_goal": PHASE_META[phase]["goal"],
                    "roles": roles,
                    "announcement": announcement,
                },
            )

            round_outputs: Dict[str, str] = {}

            for pid in panel_ids:
                role = roles.get(pid, "balanced")
                prompt = build_panel_prompt(pid, role, state, announcement, phase, cfg.summary_max_chars)

                llm = get_ollama_chat(cfg.panel_models[pid], cfg.host, temperature=0.2)
                loop = asyncio.get_running_loop()
                utter = await loop.run_in_executor(None, lambda: llm.invoke([HumanMessage(content=prompt)]).content)
                utter = str(utter)
                round_outputs[pid] = utter

                a = state["activity"].setdefault(pid, {"utterances": 0, "chars": 0, "edges": 0})
                a["utterances"] = int(a.get("utterances", 0)) + 1
                a["chars"] = int(a.get("chars", 0)) + len(utter)

                await hub.broadcast(
                    session_id,
                    {"type": "utterance", "round": r, "phase": phase, "panel_id": pid, "role": role, "text": utter},
                )

                mini = await utterance_mini_summary(cfg, state, pid, role, utter, panel_ids)
                if mini["summary"]:
                    await hub.broadcast(
                        session_id, {"type": "judge_mini", "round": r, "panel_id": pid, "summary": mini["summary"]}
                    )

                new_edges = []
                for e in mini.get("edges", []):
                    frm = e["from"]
                    to = e["to"]
                    if frm in panel_ids and to in panel_ids:
                        new_edges.append(e)

                if new_edges:
                    state["edges"].extend(new_edges)
                    a["edges"] = int(a.get("edges", 0)) + len(new_edges)
                    await hub.broadcast(session_id, {"type": "edges", "round": r, "edges": new_edges})

            await update_state_round(cfg, state, r, phase, announcement, roles, round_outputs)

            await hub.broadcast(
                session_id,
                {
                    "type": "round_end",
                    "round": r,
                    "running_summary": state["running_summary"],
                    "open_issues": [i for i in state["issue_board"] if i.get("status") == "open"],
                    "agreements": state["agreements"],
                    "unresolved": state["unresolved"],
                },
            )

        final = await final_judge(cfg, state, panel_ids)
        await hub.broadcast(session_id, {"type": "final", "final": final, "running_summary": state["running_summary"], "edges": state["edges"]})

    except Exception as e:
        await hub.broadcast(session_id, {"type": "error", "message": f"{type(e).__name__}: {e}"})


# =========================
# FastAPI
# =========================
app = FastAPI()


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


@app.post("/api/ollama/list-models")
async def list_models(payload: Dict[str, Any]):
    try:
        host = validate_ollama_host(str(payload.get("host", "")))
        data = http_get_json(f"{host}/api/tags", timeout=8.0)
        models = sorted([m.get("name") for m in data.get("models", []) if m.get("name")])
        return {"ok": True, "host": host, "models": models}
    except HTTPError as e:
        return JSONResponse({"ok": False, "error": f"HTTPError {e.code}: {e.reason}"}, status_code=400)
    except URLError as e:
        return JSONResponse({"ok": False, "error": f"URLError: {getattr(e, 'reason', str(e))}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/start")
async def start(payload: Dict[str, Any]):
    question = str(payload.get("question", "")).strip()
    if not question:
        return {"ok": False, "error": "question required"}

    try:
        host = validate_ollama_host(str(payload.get("host", "")))
    except Exception as e:
        return {"ok": False, "error": f"invalid host: {e}"}

    judge_model = str(payload.get("judge_model", "")).strip()
    if not judge_model:
        return {"ok": False, "error": "judge_model required"}

    panel_models_arr = payload.get("panel_models", [])
    if not isinstance(panel_models_arr, list) or len(panel_models_arr) < 2:
        return {"ok": False, "error": "panel_models must be a list with >= 2 items"}

    panel_models: Dict[str, str] = {}
    for it in panel_models_arr:
        if not isinstance(it, dict):
            continue
        pid = str(it.get("id", "")).strip()
        mdl = str(it.get("model", "")).strip()
        if pid and mdl:
            panel_models[pid] = mdl
    if len(panel_models) < 2:
        return {"ok": False, "error": "need at least 2 valid panel models"}

    try:
        total_rounds = int(payload.get("total_rounds", 10))
    except Exception:
        total_rounds = 10
    total_rounds = max(3, min(200, total_rounds))

    ratios_raw = str(payload.get("phase_ratios", "0.10,0.60,0.20,0.10"))
    ratios = parse_ratios(ratios_raw)

    try:
        min_each = int(payload.get("min_score_each", 5))
    except Exception:
        min_each = 5
    min_each = max(1, min(20, min_each))

    session_id = str(payload.get("session_id") or uuid.uuid4())

    cfg = SessionConfig(
        host=host,
        judge_model=judge_model,
        panel_models=panel_models,
        total_rounds=total_rounds,
        phase_ratios=ratios,
        min_score_each=min_each,
    )
    SESSION_CFG[session_id] = cfg

    if session_id in hub.tasks and not hub.tasks[session_id].done():
        return {"ok": False, "error": "session already running", "session_id": session_id}

    hub.ensure_ready(session_id)
    hub.tasks[session_id] = asyncio.create_task(debate_stream(session_id, question))
    return {"ok": True, "session_id": session_id}


@app.websocket("/ws/{session_id}")
async def ws(session_id: str, websocket: WebSocket):
    await hub.connect(session_id, websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(session_id, websocket)


# =========================
# Frontend (SVG label collision fix + stats restored)
# =========================
INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Debate Arena (SVG Graph Popup + Stats)</title>
  <style>
    :root{
      --bg:#0b1020; --text:#e9eefc; --muted:#9fb0df; --border:rgba(255,255,255,.08);
      --attack:#ff5c8a; --support:#3ee6a8; --counter:#ffc857; --nudge:#7aa2ff;
    }
    *{box-sizing:border-box}
    body{margin:0;color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,"Apple SD Gothic Neo";
      background:radial-gradient(1200px 600px at 20% 0%, #192657 0%, var(--bg) 55%);}
    header{display:flex;gap:12px;align-items:center;padding:14px 16px;border-bottom:1px solid var(--border);
      position:sticky;top:0;background:rgba(11,16,32,.75);backdrop-filter:blur(10px);z-index:20;}
    .title{font-weight:900}
    .pill{padding:6px 10px;border:1px solid var(--border);border-radius:999px;color:var(--muted);font-size:12px}
    .container{display:grid;grid-template-columns:520px 1fr;gap:14px;padding:14px}
    .card{background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02));border:1px solid var(--border);
      border-radius:16px;padding:12px;box-shadow:0 12px 40px rgba(0,0,0,.35)}
    .stack{display:flex;flex-direction:column;gap:12px}
    .row{display:flex;gap:10px;align-items:center}
    input,select,button{border-radius:12px;border:1px solid var(--border);background:rgba(255,255,255,.03);
      color:var(--text);padding:10px 12px;font-size:14px;outline:none}
    input{flex:1}
    button{cursor:pointer;background:rgba(122,162,255,.15)}
    button:hover{background:rgba(122,162,255,.25)}
    .muted{color:var(--muted)} .h{font-weight:800} .big{font-size:18px} .tiny{font-size:12px}
    .feed{max-height:calc(100vh - 170px);overflow:auto;padding-right:4px}
    .msg{border:1px solid var(--border);border-radius:14px;padding:10px 12px;background:rgba(0,0,0,.14)}
    .msg .meta{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:6px}
    .badge{padding:2px 8px;border-radius:999px;border:1px solid var(--border);font-size:11px}
    .badge.filter{border-color:rgba(255,255,255,.18);color:#dce5ff}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .summaryBox{background:linear-gradient(180deg,rgba(62,230,168,.10),rgba(255,255,255,.02));
      border:1px solid rgba(62,230,168,.22);border-radius:14px;padding:10px 12px}

    .graphWrap{
      height:260px;
      border-radius:14px;
      border:1px solid var(--border);
      background:radial-gradient(800px 360px at 20% 0%, rgba(122,162,255,.18), rgba(0,0,0,0));
      position:relative;
      overflow:hidden;
      cursor:pointer;
    }
    .graphHint{
      position:absolute; right:10px; top:10px;
      font-size:12px; color:var(--muted);
      background:rgba(11,16,32,.6);
      border:1px solid var(--border);
      padding:6px 8px; border-radius:10px;
    }

    /* Stats (restored) */
    .panelStats{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
    .stat{
      border:1px solid var(--border);
      border-radius:14px;
      padding:10px 12px;
      background:rgba(0,0,0,.10);
      min-width:240px;
      flex:1 1 240px;
    }
    .stat .id{font-weight:900;margin-bottom:6px}
    .stat .line{font-size:12px;color:var(--muted);display:flex;justify-content:space-between;gap:10px}
    .stat .line span:last-child{color:#e9eefc}

    /* Modal */
    .modalBack{
      display:none;
      position:fixed; inset:0;
      background:rgba(0,0,0,.55);
      z-index:999;
      align-items:center;
      justify-content:center;
      padding:16px;
    }
    .modal{
      width:min(1100px, 96vw);
      height:min(760px, 92vh);
      background:rgba(11,16,32,.92);
      border:1px solid var(--border);
      border-radius:18px;
      box-shadow:0 18px 80px rgba(0,0,0,.55);
      display:flex;
      flex-direction:column;
      overflow:hidden;
    }
    .modalHeader{
      padding:10px 12px;
      display:flex;
      justify-content:space-between;
      align-items:center;
      border-bottom:1px solid var(--border);
    }
    .modalBody{flex:1; padding:10px 12px;}
    .graphWrapBig{
      width:100%;
      height:100%;
      border-radius:14px;
      border:1px solid var(--border);
      background:radial-gradient(1100px 600px at 20% 0%, rgba(122,162,255,.18), rgba(0,0,0,0));
      overflow:hidden;
    }

    @media (max-width:1100px){
      .container{grid-template-columns:1fr}
      .feed{max-height:none}
    }
  </style>
</head>
<body>
<header>
  <div class="title">⚔️ Debate Arena</div>
  <div class="pill" id="status">idle</div>
  <div class="pill" id="roundInfo">Round -</div>
  <div class="pill" id="phaseInfo">Phase -</div>
  <div class="pill" id="filterPill"><span class="badge filter">filter: none</span></div>
</header>

<div class="container">
  <div class="stack">
    <div class="card">
      <div class="h big">설정</div>

      <div class="h" style="margin-top:10px;">1) Ollama 서버 연결</div>
      <div class="row" style="margin-top:10px;">
        <input id="host" value="http://127.0.0.1:11434" placeholder="http://ip:11434"/>
        <button id="loadBtn">Load</button>
      </div>
      <div class="tiny muted" style="margin-top:6px;">Load → /api/tags</div>

      <div style="margin-top:12px;" class="h">2) 모델 선택</div>
      <div class="grid2" style="margin-top:10px;">
        <div>
          <div class="tiny muted">Judge(심판)</div>
          <select id="judgeModel"></select>
        </div>
        <div>
          <div class="tiny muted">패널 수</div>
          <select id="panelCount">
            <option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option>
          </select>
        </div>
      </div>

      <div id="panelSelectors" style="margin-top:10px;"></div>

      <div style="margin-top:12px;" class="h">3) 토론 설정</div>
      <div class="grid2" style="margin-top:10px;">
        <div>
          <div class="tiny muted">Rounds</div>
          <input id="rounds" value="10"/>
        </div>
        <div>
          <div class="tiny muted">Min score each</div>
          <input id="minEach" value="5"/>
        </div>
      </div>
      <div style="margin-top:10px;">
        <div class="tiny muted">Phase Ratios</div>
        <input id="ratios" value="0.10,0.60,0.20,0.10"/>
      </div>

      <div style="margin-top:12px;" class="h">4) 질문 입력</div>
      <div class="row" style="margin-top:10px;">
        <input id="q" placeholder="질문을 입력하고 START"/>
        <button id="startBtn">START</button>
      </div>

      <div class="row" style="justify-content:space-between; margin-top:10px;">
        <div class="tiny muted">노드 클릭=필터</div>
        <button id="clearFilterBtn" style="padding:6px 10px;">Clear Filter</button>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div class="h">관계 그래프</div>
        <div class="tiny muted">click → popup</div>
      </div>
      <div class="graphWrap" id="graphSmall">
        <div class="graphHint">클릭해서 크게 보기</div>
        <div id="svgSmallHolder" style="width:100%;height:100%;"></div>
      </div>

      <!-- ✅ RESTORED: Panel Stats (left-bottom area) -->
      <div class="panelStats" id="panelStats"></div>

      <div class="summaryBox" style="margin-top:12px;">
        <div class="h">🧑‍⚖️ 심판 요약</div>
        <div class="muted tiny" id="judgeMiniMeta">-</div>
        <div id="judgeMini" style="margin-top:8px; line-height:1.5;">발언이 나올 때마다 요약이 업데이트됩니다.</div>
      </div>
    </div>
  </div>

  <div class="stack">
    <div class="card">
      <div class="h big">📡 Live Feed</div>
      <div class="muted tiny">발언 전문(필터 적용 가능)</div>
    </div>
    <div class="feed stack" id="feed"></div>
  </div>
</div>

<!-- Modal -->
<div class="modalBack" id="modalBack">
  <div class="modal">
    <div class="modalHeader">
      <div class="h">관계 그래프 (확대)</div>
      <div class="row">
        <div class="tiny muted" style="margin-right:8px;">ESC 또는 닫기</div>
        <button id="closeModalBtn" style="padding:6px 10px;">닫기</button>
      </div>
    </div>
    <div class="modalBody">
      <div class="graphWrapBig" id="graphBig">
        <div id="svgBigHolder" style="width:100%;height:100%;"></div>
      </div>
    </div>
  </div>
</div>

<script>
let sessionId=null, ws=null;
let filterPanel=null;
let panelIds=[];
let loadedModels=[];

const statusEl=document.getElementById("status");
const roundEl=document.getElementById("roundInfo");
const phaseEl=document.getElementById("phaseInfo");
const feedEl=document.getElementById("feed");
const judgeMini=document.getElementById("judgeMini");
const judgeMiniMeta=document.getElementById("judgeMiniMeta");
const filterPill=document.getElementById("filterPill");

const hostInput=document.getElementById("host");
const judgeSel=document.getElementById("judgeModel");
const panelCountSel=document.getElementById("panelCount");
const panelSelectors=document.getElementById("panelSelectors");

const panelStatsEl=document.getElementById("panelStats");

const smallHolder=document.getElementById("svgSmallHolder");
const bigHolder=document.getElementById("svgBigHolder");

const modalBack=document.getElementById("modalBack");
const closeModalBtn=document.getElementById("closeModalBtn");

window.onerror = (msg, src, line, col, err) => {
  addMsg({title:"JS Error", metaLeft:"runtime", metaRight:`${line}:${col}`, body:String(msg)});
};

function escapeHtml(s){ return (s||"").replace(/[<>&]/g, ch=>({ "<":"&lt;", ">":"&gt;", "&":"&amp;" }[ch])); }

function updateFilterUI(){
  filterPill.innerHTML=`<span class="badge filter">filter: ${filterPanel||"none"}</span>`;
  Array.from(document.querySelectorAll(".msg")).forEach(m=>{
    const pid=m.dataset.panelId;
    if(!filterPanel){ m.style.opacity=1; return; }
    m.style.opacity=(pid===filterPanel || pid==="")?1:0.25;
  });
}

function addMsg({title, metaLeft, metaRight, body, panelId}){
  if(filterPanel && panelId && panelId!==filterPanel) return;
  const div=document.createElement("div");
  div.className="msg";
  div.dataset.panelId=panelId||"";
  div.innerHTML=`
    <div class="meta"><div>${escapeHtml(metaLeft||"")}</div><div>${escapeHtml(metaRight||"")}</div></div>
    <div class="h">${escapeHtml(title||"")}</div>
    <div style="margin-top:8px; white-space:pre-wrap; line-height:1.45;">${escapeHtml(body||"")}</div>`;
  feedEl.prepend(div);
}

/* ---------------- Stats (RESTORED) ---------------- */
const interactionCounts = {}; // pid -> {out:{},in:{}}
function ensureCounts(pid){
  if(!interactionCounts[pid]){
    interactionCounts[pid]={
      out:{attack:0,support:0,counter:0,nudge:0},
      in:{attack:0,support:0,counter:0,nudge:0}
    };
  }
}
function renderStats(){
  panelStatsEl.innerHTML="";
  if(!panelIds.length){
    panelStatsEl.innerHTML = `<div class="tiny muted">(패널 시작 전)</div>`;
    return;
  }
  panelIds.forEach(pid=>{
    ensureCounts(pid);
    const c = interactionCounts[pid];
    const d=document.createElement("div");
    d.className="stat";
    d.innerHTML = `
      <div class="id">${pid}</div>
      <div class="line"><span class="muted">out attack</span><span>${c.out.attack}</span></div>
      <div class="line"><span class="muted">out support</span><span>${c.out.support}</span></div>
      <div class="line"><span class="muted">out counter</span><span>${c.out.counter}</span></div>
      <div class="line"><span class="muted">out nudge</span><span>${c.out.nudge}</span></div>
      <div class="line"><span class="muted">in attack</span><span>${c.in.attack}</span></div>
      <div class="line"><span class="muted">in support</span><span>${c.in.support}</span></div>
      <div class="line"><span class="muted">in counter</span><span>${c.in.counter}</span></div>
      <div class="line"><span class="muted">in nudge</span><span>${c.in.nudge}</span></div>
    `;
    panelStatsEl.appendChild(d);
  });
}

/* ---------------- SVG Graph ---------------- */
let graphNodes=[]; // [{id}]
let graphEdges=[]; // [{from,to,type,label}]

function colorByType(t){
  const css=getComputedStyle(document.documentElement);
  if(t==="attack") return css.getPropertyValue("--attack").trim();
  if(t==="support") return css.getPropertyValue("--support").trim();
  if(t==="counter") return css.getPropertyValue("--counter").trim();
  return css.getPropertyValue("--nudge").trim();
}

function wrapText(str, maxChars){
  const s=(str||"").trim();
  if(!s) return [];
  const words=s.split(/\s+/);
  const lines=[];
  let cur="";
  for(const w of words){
    if(!cur){ cur=w; continue; }
    if((cur.length + 1 + w.length) <= maxChars){
      cur += " " + w;
    }else{
      lines.push(cur);
      cur = w;
    }
  }
  if(cur) lines.push(cur);
  if(lines.length===0){
    for(let i=0;i<s.length;i+=maxChars) lines.push(s.slice(i,i+maxChars));
  }
  return lines;
}

function rectsOverlap(a,b){
  return !(a.x+a.w < b.x || b.x+b.w < a.x || a.y+a.h < b.y || b.y+b.h < a.y);
}

function renderGraph(holder, width, height){
  holder.innerHTML="";
  const svgNS="http://www.w3.org/2000/svg";
  const svg=document.createElementNS(svgNS,"svg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.display="block";

  const defs=document.createElementNS(svgNS,"defs");
  ["attack","support","counter","nudge"].forEach(t=>{
    const m=document.createElementNS(svgNS,"marker");
    m.setAttribute("id","arrow_"+t);
    m.setAttribute("markerWidth","10");
    m.setAttribute("markerHeight","10");
    m.setAttribute("refX","9");
    m.setAttribute("refY","5");
    m.setAttribute("orient","auto");
    const p=document.createElementNS(svgNS,"path");
    p.setAttribute("d","M 0 0 L 10 5 L 0 10 z");
    p.setAttribute("fill",colorByType(t));
    m.appendChild(p);
    defs.appendChild(m);
  });
  svg.appendChild(defs);

  const n=graphNodes.length || 1;
  const cx=width/2, cy=height/2;
  const R=Math.min(width,height)*0.34;
  const nodePos={};
  graphNodes.forEach((node, i)=>{
    const ang = (Math.PI*2*i)/n - Math.PI/2;
    nodePos[node.id]={x: cx + R*Math.cos(ang), y: cy + R*Math.sin(ang)};
  });

  const edgesToDraw = graphEdges.slice(-120);

  edgesToDraw.forEach((e)=>{
    const a=nodePos[e.from], b=nodePos[e.to];
    if(!a || !b) return;
    const line=document.createElementNS(svgNS,"line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    const col=colorByType(e.type);
    line.setAttribute("stroke", col);
    line.setAttribute("stroke-width", "2.2");
    line.setAttribute("marker-end", `url(#arrow_${e.type||"nudge"})`);
    line.setAttribute("opacity","0.95");
    svg.appendChild(line);
  });

  const placedBboxes=[];
  const labelMaxWidth = Math.min(220, Math.floor(width*0.30));
  const approxCharW = 7;
  const paddingX = 10, paddingY = 8;
  const lineH = 14;

  function placeLabel(e){
    const a=nodePos[e.from], b=nodePos[e.to];
    if(!a || !b) return;

    let label=(e.label||"").trim();
    if(!label) return;

    const maxChars = Math.max(10, Math.floor((labelMaxWidth - paddingX*2)/approxCharW));
    let lines = wrapText(label, maxChars);
    if(lines.length>3){
      lines = lines.slice(0,2).concat([(lines[2].slice(0, Math.max(3, maxChars-1)) + "…")]);
    }

    const textW = Math.min(labelMaxWidth, Math.max(60, Math.max(...lines.map(l=>l.length))*approxCharW + paddingX*2));
    const textH = lines.length*lineH + paddingY*2;

    const mx=(a.x+b.x)/2, my=(a.y+b.y)/2;
    const dx=b.x-a.x, dy=b.y-a.y;
    const len=Math.max(1, Math.hypot(dx,dy));
    const nx=-dy/len, ny=dx/len;
    const baseOffset = 18;

    const attempts = 10;
    const step = 18;
    let best = null;

    for(let k=0;k<attempts;k++){
      const s = (k%2===0) ? 1 : -1;
      const mag = baseOffset + step*Math.floor(k/2);
      const ox = nx*mag*s;
      const oy = ny*mag*s;

      const x = mx + ox - textW/2;
      const y = my + oy - textH/2;

      const bbox = {x,y,w:textW,h:textH};
      if(bbox.x < 4 || bbox.y < 4 || bbox.x+bbox.w > width-4 || bbox.y+bbox.h > height-4){
        continue;
      }

      let coll=false;
      for(const pb of placedBboxes){
        if(rectsOverlap(bbox, pb)){ coll=true; break; }
      }
      if(!coll){ best=bbox; break; }
      if(!best) best=bbox;
    }

    if(!best) return;

    let heavy=false;
    for(const pb of placedBboxes){
      if(rectsOverlap(best, pb)){ heavy=true; break; }
    }
    if(heavy){
      const short = label.slice(0, Math.max(8, Math.floor(maxChars*0.7))) + "…";
      lines = [short];
    }

    const g=document.createElementNS(svgNS,"g");
    const rect=document.createElementNS(svgNS,"rect");
    rect.setAttribute("x", best.x);
    rect.setAttribute("y", best.y);
    rect.setAttribute("width", best.w);
    rect.setAttribute("height", best.h);
    rect.setAttribute("rx","10");
    rect.setAttribute("fill","rgba(11,16,32,0.72)");
    rect.setAttribute("stroke","rgba(255,255,255,0.12)");
    g.appendChild(rect);

    const col=colorByType(e.type);
    const tag=document.createElementNS(svgNS,"rect");
    tag.setAttribute("x", best.x+6);
    tag.setAttribute("y", best.y+6);
    tag.setAttribute("width", 8);
    tag.setAttribute("height", best.h-12);
    tag.setAttribute("rx","6");
    tag.setAttribute("fill", col);
    tag.setAttribute("opacity","0.95");
    g.appendChild(tag);

    for(let i=0;i<lines.length;i++){
      const t=document.createElementNS(svgNS,"text");
      t.setAttribute("x", best.x + 18);
      t.setAttribute("y", best.y + paddingY + lineH*(i+1) - 4);
      t.setAttribute("font-size","12");
      t.setAttribute("fill","#cbd6ff");
      t.textContent = lines[i];
      g.appendChild(t);
    }

    svg.appendChild(g);
    placedBboxes.push(best);
  }

  edgesToDraw.forEach(placeLabel);

  graphNodes.forEach((node)=>{
    const p=nodePos[node.id];
    if(!p) return;
    const g=document.createElementNS(svgNS,"g");
    g.style.cursor="pointer";

    const circle=document.createElementNS(svgNS,"circle");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", 22);
    circle.setAttribute("fill", (filterPanel===node.id) ? "rgba(122,162,255,0.9)" : "rgba(122,162,255,0.65)");
    circle.setAttribute("stroke","rgba(255,255,255,0.12)");
    circle.setAttribute("stroke-width","2");

    const text=document.createElementNS(svgNS,"text");
    text.setAttribute("x", p.x);
    text.setAttribute("y", p.y+5);
    text.setAttribute("text-anchor","middle");
    text.setAttribute("font-size","12");
    text.setAttribute("fill","#e9eefc");
    text.textContent=node.id;

    g.appendChild(circle);
    g.appendChild(text);

    g.addEventListener("click",(ev)=>{
      ev.stopPropagation();
      filterPanel = (filterPanel===node.id) ? null : node.id;
      updateFilterUI();
      redrawGraphs();
    });

    svg.appendChild(g);
  });

  holder.appendChild(svg);
}

function redrawGraphs(){
  const sRect = document.getElementById("graphSmall").getBoundingClientRect();
  const wS = Math.max(220, Math.floor(sRect.width));
  const hS = Math.max(220, Math.floor(sRect.height));
  renderGraph(smallHolder, wS, hS);

  if(modalBack.style.display==="flex"){
    const bRect = document.getElementById("graphBig").getBoundingClientRect();
    const wB = Math.max(600, Math.floor(bRect.width));
    const hB = Math.max(420, Math.floor(bRect.height));
    renderGraph(bigHolder, wB, hB);
  }
}

/* modal */
function openModal(){
  modalBack.style.display="flex";
  setTimeout(()=>redrawGraphs(), 30);
}
function closeModal(){ modalBack.style.display="none"; }
document.getElementById("graphSmall").addEventListener("click", openModal);
closeModalBtn.addEventListener("click", closeModal);
modalBack.addEventListener("click",(e)=>{ if(e.target===modalBack) closeModal(); });
window.addEventListener("keydown",(e)=>{ if(e.key==="Escape") closeModal(); });
window.addEventListener("resize", ()=> setTimeout(redrawGraphs, 80));

/* config */
function fillSelect(sel, items){
  sel.innerHTML="";
  items.forEach(x=>{
    const op=document.createElement("option");
    op.value=x; op.textContent=x;
    sel.appendChild(op);
  });
}
function rebuildPanelSelectors(){
  const n=parseInt(panelCountSel.value,10);
  panelSelectors.innerHTML="";
  for(let i=1;i<=n;i++){
    const pid="P"+i;
    const wrap=document.createElement("div");
    wrap.style.marginTop="8px";
    wrap.innerHTML=`
      <div class="tiny muted">Panel ${pid} (ID 변경 가능)</div>
      <div class="row">
        <input class="pid" value="${pid}" style="max-width:110px"/>
        <select class="pmodel" style="flex:1"></select>
      </div>`;
    panelSelectors.appendChild(wrap);
    const sel=wrap.querySelector(".pmodel");
    fillSelect(sel, loadedModels.length?loadedModels:["(load first)"]);
  }
}
panelCountSel.addEventListener("change", rebuildPanelSelectors);

async function loadModels(){
  const host=hostInput.value.trim();
  statusEl.textContent="loading models…";
  try{
    const res = await fetch("/api/ollama/list-models", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ host })
    });
    const data = await res.json();
    if(!data.ok){
      statusEl.textContent="error";
      addMsg({title:"Load error", metaLeft:"/api/tags", metaRight:host, body:data.error||"unknown"});
      return;
    }
    loadedModels = data.models || [];
    if(!loadedModels.length){
      statusEl.textContent="no models";
      addMsg({title:"No models", metaLeft:data.host, body:"서버에 모델이 없습니다."});
      return;
    }
    statusEl.textContent="models loaded";
    fillSelect(judgeSel, loadedModels);
    judgeSel.value = loadedModels[0];
    rebuildPanelSelectors();
    addMsg({title:"Models loaded", metaLeft:data.host, metaRight:`count ${loadedModels.length}`, body:loadedModels.join("\\n")});
  }catch(err){
    statusEl.textContent="error";
    addMsg({title:"Load exception", metaLeft:"fetch", body:String(err)});
  }
}
document.getElementById("loadBtn").addEventListener("click", loadModels);

document.getElementById("clearFilterBtn").addEventListener("click",()=>{
  filterPanel=null;
  updateFilterUI();
  redrawGraphs();
});

/* START */
async function start(){
  const question=document.getElementById("q").value.trim();
  const host=hostInput.value.trim();
  if(!question){ alert("질문을 입력하세요"); return; }
  if(!loadedModels.length){ alert("먼저 Load"); return; }
  if(!judgeSel.value){ alert("Judge 모델 선택"); return; }

  statusEl.textContent="connecting…";
  feedEl.innerHTML="";
  judgeMini.textContent="심판 요약 준비 중…";
  judgeMiniMeta.textContent="-";
  roundEl.textContent="Round -";
  phaseEl.textContent="Phase -";

  // reset graph + stats
  graphNodes=[]; graphEdges=[]; panelIds=[];
  filterPanel=null; updateFilterUI();
  for(const k of Object.keys(interactionCounts)) delete interactionCounts[k];
  renderStats();
  redrawGraphs();

  const panelModels=[];
  Array.from(panelSelectors.querySelectorAll("div")).forEach(w=>{
    const pid = w.querySelector(".pid")?.value?.trim();
    const mdl = w.querySelector(".pmodel")?.value?.trim();
    if(pid && mdl) panelModels.push({id: pid, model: mdl});
  });
  if(panelModels.length < 2){ alert("패널 최소 2"); return; }

  const totalRounds = parseInt(document.getElementById("rounds").value.trim() || "10", 10);
  const ratios = document.getElementById("ratios").value.trim() || "0.10,0.60,0.20,0.10";
  const minEach = parseInt(document.getElementById("minEach").value.trim() || "5", 10);

  sessionId=crypto.randomUUID();
  const WS_SCHEME = (location.protocol === "https:") ? "wss" : "ws";
  ws = new WebSocket(`${WS_SCHEME}://${location.host}/ws/${sessionId}`);

  ws.onerror = () => {
    statusEl.textContent = "ws error";
    addMsg({title:"WebSocket error", metaLeft:"ws", body:"WS 연결 실패. https이면 wss 필요."});
  };

  ws.onopen = async ()=>{
    statusEl.textContent="connected";
    try{
      const res=await fetch("/api/start",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          session_id: sessionId,
          question,
          host,
          judge_model: judgeSel.value,
          panel_models: panelModels,
          total_rounds: totalRounds,
          phase_ratios: ratios,
          min_score_each: minEach
        })
      });
      const data=await res.json();
      if(!data.ok){
        statusEl.textContent="start error";
        addMsg({title:"Start error", metaLeft:"/api/start", body:data.error||"unknown"});
        ws.close();
      }
    }catch(err){
      statusEl.textContent="start error";
      addMsg({title:"Start request failed", metaLeft:"/api/start", body:String(err)});
      ws.close();
    }
  };

  ws.onmessage=(ev)=>handle(JSON.parse(ev.data));
  ws.onclose=()=>{ if(statusEl.textContent!=="done") statusEl.textContent="closed"; };
}
document.getElementById("startBtn").addEventListener("click", start);

/* Handle */
function handle(msg){
  if(msg.type==="error"){
    statusEl.textContent="error";
    addMsg({title:"Server error", metaLeft:"server", body:msg.message||"unknown"});
    return;
  }

  if(msg.type==="init"){
    statusEl.textContent="running";
    panelIds=(msg.panels||[]).map(p=>p.id);
    graphNodes = panelIds.map(id=>({id}));
    graphEdges = [];

    // init stats
    panelIds.forEach(pid=>ensureCounts(pid));
    renderStats();
    redrawGraphs();

    addMsg({
      title:"Session Init",
      metaLeft:`host ${msg.host}`,
      metaRight:`judge ${msg.judge_model}`,
      body:`Q: ${msg.question}\\nPanels:\\n${(msg.panels||[]).map(p=>`${p.id} = ${p.model}`).join("\\n")}`
    });
    return;
  }

  if(msg.type==="round_start"){
    roundEl.textContent=`Round ${msg.round}`;
    phaseEl.textContent=`Phase ${msg.phase}: ${msg.phase_name}`;
    addMsg({title:`Round ${msg.round} Start`, metaLeft:`Phase ${msg.phase}`, metaRight:msg.phase_name, body:msg.announcement});
    return;
  }

  if(msg.type==="utterance"){
    addMsg({title:`${msg.panel_id}`, metaLeft:`R${msg.round} role=${msg.role}`, metaRight:`Phase ${msg.phase}`, body:msg.text, panelId:msg.panel_id});
    return;
  }

  if(msg.type==="judge_mini"){
    judgeMini.textContent = msg.summary || "";
    judgeMiniMeta.textContent = `R${msg.round} · after ${msg.panel_id}`;
    return;
  }

  if(msg.type==="edges"){
    (msg.edges||[]).forEach(e=>{
      if(!graphNodes.find(n=>n.id===e.from)) graphNodes.push({id:e.from});
      if(!graphNodes.find(n=>n.id===e.to)) graphNodes.push({id:e.to});
      graphEdges.push(e);

      // ✅ update stats
      const t = (e.type || "nudge");
      ensureCounts(e.from); ensureCounts(e.to);
      if(interactionCounts[e.from].out[t] !== undefined) interactionCounts[e.from].out[t] += 1;
      if(interactionCounts[e.to].in[t] !== undefined) interactionCounts[e.to].in[t] += 1;
    });
    renderStats();
    redrawGraphs();
    return;
  }

  if(msg.type==="final"){
    statusEl.textContent="done";
    addMsg({title:"FINAL", metaLeft:"judge", body:JSON.stringify(msg.final,null,2)});
    return;
  }
}
</script>
</body>
</html>
"""


# =========================
# Run
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)