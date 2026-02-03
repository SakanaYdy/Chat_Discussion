import os
import json
import time
import http.client
import streamlit as st
import pandas as pd
from urllib.parse import urlparse

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "providers.json")

def _normalize_base_url(b, default):
    s = (b or "").strip().strip("`").strip('"').strip("'")
    return s or default

def _compose_path(base_path, endpoint):
    bp = (base_path or "")
    if bp and not bp.startswith("/"):
        bp = "/" + bp
    bp = bp.rstrip("/")
    ep = endpoint if endpoint.startswith("/") else "/" + endpoint
    if bp.endswith("/v1") and ep.startswith("/v1/"):
        ep = ep[len("/v1"):]
    if bp.endswith("/api/v1") and ep.startswith("/api/v1/"):
        ep = ep[len("/api/v1"):]
    return (bp + ep) if bp else ep

def sanitize_output(text):
    t = (text or "").strip()
    if not t:
        return t
    banned = ["下一步", "请告诉我", "请回复", "请选择", "告诉我你的选择", "回复数字", "你想", "需要你", "你是否"]
    lines = [ln.strip() for ln in t.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            kept.append(ln)
            continue
        if any(b in ln for b in banned):
            continue
        kept.append(ln)
    out = "\n".join(kept).strip()
    return out or t

def load_providers():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            pass
    default = {
        "OpenAI默认": {"name": "OpenAI默认", "kind": "openai", "base_url": "https://api.openai.com", "api_key": os.environ.get("OPENAI_API_KEY", ""), "model": "gpt-4o-mini"},
        "通义千问默认": {"name": "通义千问默认", "kind": "qwen", "base_url": "https://dashscope.aliyuncs.com", "api_key": os.environ.get("DASHSCOPE_API_KEY", ""), "model": "qwen-plus"},
    }
    save_providers(default)
    return default

def save_providers(data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def https_json_request(host, path, method, headers, payload):
    conn = http.client.HTTPSConnection(host, timeout=60)
    body = json.dumps(payload) if payload is not None else None
    conn.request(method, path, body=body, headers=headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    if 200 <= res.status < 300:
        try:
            return json.loads(data)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON response: {e}\n{data}")
    else:
        raise RuntimeError(f"HTTP {res.status}: {data}")

def call_openai(messages, model, base_url=None, api_key=None):
    k = api_key or os.environ.get("OPENAI_API_KEY")
    if not k:
        raise RuntimeError("缺少 OPENAI_API_KEY")
    b = _normalize_base_url(base_url, "https://api.openai.com")
    u = urlparse(b)
    host = u.netloc or b.replace("https://", "").replace("http://", "")
    base_path = (u.path or "").rstrip("/")
    path = _compose_path(base_path, "/v1/chat/completions")
    payload = {"model": model or "gpt-4o-mini", "messages": messages, "temperature": 0.4}
    headers = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
    data = https_json_request(host, path, "POST", headers, payload)
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")

def call_qwen(messages, model, base_url=None, api_key=None):
    k = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not k:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")
    b = _normalize_base_url(base_url, "https://dashscope.aliyuncs.com")
    u = urlparse(b)
    host = u.netloc or b.replace("https://", "").replace("http://", "")
    base_path = (u.path or "").rstrip("/")
    path = _compose_path(base_path, "/compatible-mode/v1/chat/completions")
    payload = {"model": model or "qwen-plus", "messages": messages, "temperature": 0.4}
    headers = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
    data = https_json_request(host, path, "POST", headers, payload)
    content = ""
    if isinstance(data, dict):
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or data.get("output_text", "")
    return content

def run_discussion(topic, question, participants, rounds, providers):
    base_system = {
        "role": "system",
        "content": f"你是参与方案讨论的专家。请围绕主题进行分析，提出可执行的方案步骤、注意事项与潜在风险。输出请结构化分点说明。主题：{topic}",
    }
    user_msg = {"role": "user", "content": question}
    per_model_rounds = {m: [] for m in participants}
    context_text = ""
    for r in range(rounds):
        for m in participants:
            msgs = [base_system, user_msg]
            if context_text:
                msgs.append({"role": "user", "content": f"上一轮讨论摘要：\n{context_text}\n请在本轮进一步完善并提出改进。"})
            cfg = providers.get(m) or {}
            kind = cfg.get("kind")
            model_name = cfg.get("model")
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            try:
                if kind == "openai":
                    content = call_openai(msgs, model_name, base_url, api_key)
                elif kind == "qwen":
                    content = call_qwen(msgs, model_name, base_url, api_key)
                else:
                    content = f"未知模型类型：{kind}"
                per_model_rounds[m].append({"round": r + 1, "content": content})
            except Exception as e:
                per_model_rounds[m].append({"round": r + 1, "error": str(e), "content": ""})
        aggregate = []
        for m in participants:
            last = per_model_rounds[m][-1]
            aggregate.append(f"【{m}·第{last['round']}轮】\n" + (last.get("content") or f"错误：{last.get('error')}"))
        context_text = "\n\n".join(aggregate)
    return per_model_rounds

def run_summary(topic, question, per_model_rounds, summarizer, providers):
    summary_system = {
        "role": "system",
        "content": "你是首席总结官。请综合多轮多位专家的讨论，找出共识与分歧，制定一个清晰的、可执行的最终方案：包括目标、关键步骤、资源与角色、时间计划、风险与应对、验收标准。输出中文分点，最后给出一个简短执行清单。",
    }
    combined_parts = []
    for m, items in per_model_rounds.items():
        for it in items:
            combined_parts.append(f"【{m}·第{it['round']}轮】\n" + (it.get("content") or f"错误：{it.get('error')}"))
    combined = "\n\n".join(combined_parts)
    user_msg = {"role": "user", "content": f"主题：{topic}\n问题：{question}\n以下是多轮讨论内容，请总结并给出最终方案：\n\n{combined}"}
    msgs = [summary_system, user_msg]
    cfg = providers.get(summarizer) or {}
    kind = cfg.get("kind")
    model_name = cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    if kind == "openai":
        return call_openai(msgs, model_name, base_url, api_key)
    if kind == "qwen":
        return call_qwen(msgs, model_name, base_url, api_key)
    raise RuntimeError("未知总结模型类型")

def run_dialogue(topic, question, participants, rounds, providers, live_container=None, progress=None):
    transcript = []
    total_steps = max(1, rounds * max(1, len(participants)))
    step = 0
    for r in range(rounds):
        for m in participants:
            step += 1
            context_text = "\n\n".join(
                [f"【{t['speaker']}·第{t['round']}轮】\n{t.get('content') or ('错误：' + t.get('error', ''))}" for t in transcript]
            )
            system_msg = {
                "role": "system",
                "content": f"你是参与方案讨论的专家，将以轮次对话的方式与其他专家协作，逐步形成可执行方案。主题：{topic}\n\n硬性约束：\n- 不要向用户提问，不要要求用户选择，不要让用户“回复数字/告诉你选择”。\n- 不要出现“下一步请…/请告诉我…”等追问句式。\n- 直接输出你的分析结论与可执行产出（步骤/清单/命令/配置/伪代码均可）。",
            }
            user_msg = {
                "role": "user",
                "content": f"起始问题：{question}\n当前对话：\n{context_text}\n请以「{m}」身份进行第{r+1}轮发言：补充更具体的实现细节（可落地步骤/参数/接口/示例），并纠正或补强上一轮的不足。禁止提问与征询选择。",
            }
            cfg = providers.get(m) or {}
            kind = cfg.get("kind")
            model_name = cfg.get("model")
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            try:
                if kind == "openai":
                    content = call_openai([system_msg, user_msg], model_name, base_url, api_key)
                elif kind == "qwen":
                    content = call_qwen([system_msg, user_msg], model_name, base_url, api_key)
                else:
                    content = f"未知模型类型：{kind}"
                content = sanitize_output(content)
                transcript.append({"round": r + 1, "speaker": m, "content": content})
                if live_container is not None:
                    with live_container:
                        with st.chat_message("assistant"):
                            st.markdown(f"**{m} · 第{r+1}轮**")
                            st.write(content)
            except Exception as e:
                transcript.append({"round": r + 1, "speaker": m, "error": str(e), "content": ""})
                if live_container is not None:
                    with live_container:
                        with st.chat_message("assistant"):
                            st.markdown(f"**{m} · 第{r+1}轮**")
                            st.error(str(e))
            if progress is not None:
                progress.progress(step / total_steps)
    return transcript

def run_summary_from_transcript(topic, question, transcript, summarizer, providers):
    summary_system = {
        "role": "system",
        "content": "你是首席实施官（资深技术负责人）。目标：基于对话，输出一份“可以直接照着做”的落地实施方案。\n\n硬性约束：\n- 禁止向用户提问、征询选择、让用户回复数字；禁止出现“下一步请…”等引导语。\n- 不要输出空泛规划/套话，不要只有“第一阶段/第二阶段”这种宏观描述。\n- 每一条都必须可执行、可验证，并尽量具体到：命令、配置、参数、接口、数据结构、示例输入输出。\n\n输出必须包含（Markdown 标题）：\n1) 最终方案（1页版）：一句话目标 + 关键决策(>=6) + 取舍理由(>=6)\n2) 详细实施步骤：按执行顺序列出(>=20条)，每条包含：目的/操作/产出物/验收方式\n3) 关键设计：接口与数据结构（至少给出 3 个 JSON 示例：请求/响应/配置）\n4) 关键实现要点：给出可直接抄用的伪代码或关键代码片段（>=3段）\n5) 风险与应对：>=8条（含监控指标与回滚/降级）\n6) 验收清单：>=10条（尽量量化，给出验证方法）\n\n最后：把对话里出现的建议逐条映射到你的步骤或设计里（缺失则补齐），确保与讨论强相关。",
    }
    combined = "\n\n".join(
        [f"【{t['speaker']}·第{t['round']}轮】\n{t.get('content') or ('错误：' + t.get('error', ''))}" for t in transcript]
    )
    user_msg = {"role": "user", "content": f"主题：{topic}\n问题：{question}\n以下是对话内容（按轮次）：\n\n{combined}\n\n请严格按系统要求输出最终方案。"}
    cfg = providers.get(summarizer) or {}
    kind = cfg.get("kind")
    model_name = cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    if kind == "openai":
        return sanitize_output(call_openai([summary_system, user_msg], model_name, base_url, api_key))
    if kind == "qwen":
        return sanitize_output(call_qwen([summary_system, user_msg], model_name, base_url, api_key))
    raise RuntimeError("未知总结模型类型")

def test_provider(name, providers):
    cfg = providers.get(name) or {}
    kind = cfg.get("kind")
    model_name = cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    msgs = [
        {"role": "system", "content": "你是健康检查助手"},
        {"role": "user", "content": "请用一句话确认服务可用"},
    ]
    start = time.time()
    try:
        if kind == "openai":
            content = call_openai(msgs, model_name, base_url, api_key)
        elif kind == "qwen":
            content = call_qwen(msgs, model_name, base_url, api_key)
        else:
            return {"ok": False, "error": f"未知模型类型：{kind}"}
        return {"ok": True, "content": content, "elapsed": time.time() - start}
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - start}

def generate_topic_from_question(question):
    q = (question or "").strip().replace("\n", " ")
    q = " ".join(q.split())
    if not q:
        return "未命名主题"
    short = q[:30]
    return f"自动主题：{short}"

def build_length_metrics(per_model_rounds):
    metrics = []
    for m, items in per_model_rounds.items():
        txt = "\n\n".join([x.get("content", "") for x in items if x.get("content")])
        metrics.append({"model": m, "chars": len(txt)})
    return pd.DataFrame(metrics)

st.set_page_config(page_title="多模型方案讨论系统（Python + Streamlit）", layout="wide")
st.title("多模型方案讨论系统（Python + Streamlit）")
st.caption("配置多个模型参与多轮讨论，并由总结模型给出最终方案，同时可视化讨论产出规模")

if "providers" not in st.session_state:
    st.session_state["providers"] = load_providers()
if "active_dialog" not in st.session_state:
    st.session_state["active_dialog"] = None

@st.dialog("新增模型")
def add_model_dialog():
    with st.form("add_provider_form", clear_on_submit=False):
        add_name = st.text_input("名称", value="")
        add_kind = st.selectbox("类型", ["openai", "qwen"], index=0)
        add_base = st.text_input(
            "BaseURL",
            value="https://api.openai.com" if add_kind == "openai" else "https://dashscope.aliyuncs.com",
        )
        add_key = st.text_input("API Key", value="", type="password")
        add_model = st.selectbox(
            "模型",
            ["gpt-4o-mini", "gpt-4o"] if add_kind == "openai" else ["qwen-plus", "qwen-turbo", "qwen-max"],
            index=0,
        )
        submitted = st.form_submit_button("保存")
        if submitted:
            if not add_name:
                st.error("名称不能为空")
                return
            st.session_state["providers"][add_name] = {
                "name": add_name,
                "kind": add_kind,
                "base_url": add_base,
                "api_key": add_key,
                "model": add_model,
            }
            save_providers(st.session_state["providers"])
            st.session_state["active_dialog"] = None
            st.rerun()

@st.dialog("管理模型（修改 / 连通性测试）")
def manage_model_dialog():
    names = list(st.session_state["providers"].keys())
    if not names:
        st.info("暂无模型配置，请先新增")
        return
    cols = st.columns(2)
    for idx, n in enumerate(names):
        p = st.session_state["providers"].get(n) or {}
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{p.get('name', n)}**")
                st.write(f"类型：{p.get('kind', '')}")
                st.write(f"模型：{p.get('model', '')}")
                st.write(f"BaseURL：{p.get('base_url', '')}")
                if st.button("编辑 / 测试", key=f"open_edit_{n}"):
                    st.session_state["edit_target"] = n
                    st.session_state["active_dialog"] = "edit"
                    st.rerun()

@st.dialog("编辑模型（修改 / 连通性测试）")
def edit_model_dialog():
    sel = st.session_state.get("edit_target")
    if not sel or sel not in st.session_state["providers"]:
        st.info("未选择模型")
        return
    p = st.session_state["providers"].get(sel) or {}
    new_name = st.text_input("名称", value=p.get("name", sel))
    kind = st.selectbox("类型", ["openai", "qwen"], index=0 if p.get("kind") == "openai" else 1)
    base_url = st.text_input("BaseURL", value=p.get("base_url", ""))
    api_key = st.text_input("API Key", value=p.get("api_key", ""), type="password")
    if kind == "openai":
        model = st.selectbox("模型", ["gpt-4o-mini", "gpt-4o"], index=0 if p.get("model") == "gpt-4o-mini" else 1)
    else:
        qwen_models = ["qwen-plus", "qwen-turbo", "qwen-max"]
        model = st.selectbox("模型", qwen_models, index=qwen_models.index(p.get("model")) if p.get("model") in qwen_models else 0)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("保存修改"):
            if not new_name:
                st.error("名称不能为空")
                return
            del st.session_state["providers"][sel]
            st.session_state["providers"][new_name] = {
                "name": new_name,
                "kind": kind,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }
            save_providers(st.session_state["providers"])
            st.session_state["active_dialog"] = None
            st.session_state["edit_target"] = None
            st.rerun()
    with c2:
        if st.button("连通性测试"):
            tmp = {
                new_name: {
                    "name": new_name,
                    "kind": kind,
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                }
            }
            res = test_provider(new_name, tmp)
            if res.get("ok"):
                st.success(f"成功，用时 {res.get('elapsed', 0):.1f}s")
                st.write(res.get("content", ""))
            else:
                st.error(f"失败，用时 {res.get('elapsed', 0):.1f}s")
                st.write(res.get("error", ""))

with st.sidebar:
    st.header("模型管理")
    mgmt_cols = st.columns([1, 1])
    with mgmt_cols[0]:
        if st.button("新增"):
            st.session_state["active_dialog"] = "add"
    with mgmt_cols[1]:
        if st.button("修改/测试"):
            st.session_state["active_dialog"] = "manage"
    st.divider()
    st.header("参与与总结")
    provider_names = list(st.session_state["providers"].keys())
    models = st.multiselect("参与模型", provider_names, default=provider_names)
    summarizer = st.selectbox("总结模型", provider_names, index=0 if provider_names else None)
    rounds = st.slider("讨论轮次", min_value=1, max_value=5, value=2)

topic = st.text_input("主题（可留空自动生成）", value="")
question = st.text_area("具体问题/目标", value="", height=140)

if st.session_state["active_dialog"] == "add":
    add_model_dialog()
if st.session_state["active_dialog"] == "manage":
    manage_model_dialog()
if st.session_state["active_dialog"] == "edit":
    edit_model_dialog()

run = st.button("开始多轮讨论并生成方案")

if run:
    if not question or not models:
        st.error("请填写问题，并至少选择一个参与模型")
    else:
        if not topic:
            topic = generate_topic_from_question(question)
        status = st.status("正在进行讨论与总结", expanded=True)
        try:
            status.update(label="正在进行多轮讨论", state="running")
            live_expander = st.expander("对话过程（实时）", expanded=True)
            with live_expander:
                live_container = st.container()
                with live_container:
                    with st.chat_message("user"):
                        st.write(question)
                progress = st.progress(0.0)
            start = time.time()
            transcript = run_dialogue(topic, question, models, rounds, st.session_state["providers"], live_container=live_container, progress=progress)
            status.update(label="讨论完成，正在生成总结", state="running")
            summary = run_summary_from_transcript(topic, question, transcript, summarizer, st.session_state["providers"])
            elapsed = time.time() - start
            status.update(label=f"完成，用时 {elapsed:.1f}s", state="complete")

            cols = st.columns(2)
            with cols[0]:
                st.subheader("轮次对话")
                for t in transcript:
                    st.markdown(f"**第{t['round']}轮 · {t['speaker']}**")
                    if t.get("error"):
                        st.error(t["error"])
                    else:
                        st.write(t["content"])
            with cols[1]:
                tabs = st.tabs(["最终方案", "可视化"])
                with tabs[0]:
                    st.write(summary)
                with tabs[1]:
                    df = pd.DataFrame([{"model": x["speaker"], "chars": len(x.get("content", ""))} for x in transcript]).groupby("model", as_index=False).sum().set_index("model")
                    st.bar_chart(df)
        except Exception as e:
            status.update(label="执行失败", state="error")
            st.error(str(e))
