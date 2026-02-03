import os
import json
import http.client
from urllib.parse import urlparse

def https_json_request(host, path, headers, payload):
    conn = http.client.HTTPSConnection(host, timeout=60)
    body = json.dumps(payload)
    conn.request("POST", path, body=body, headers=headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    return res.status, data

def load_config():
    cfg = {
        "base_url": "https://dashscope.aliyuncs.com",
        "api_key": os.environ.get("DASHSCOPE_API_KEY") or "",
        "model": "qwen-plus",
    }
    try:
        with open(os.path.join(os.path.dirname(__file__), "providers.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            # Prefer a provider with kind == 'qwen'
            for name, p in data.items():
                if isinstance(p, dict) and p.get("kind") == "qwen":
                    cfg["base_url"] = p.get("base_url") or cfg["base_url"]
                    cfg["api_key"] = p.get("api_key") or cfg["api_key"]
                    cfg["model"] = p.get("model") or cfg["model"]
                    break
    except Exception:
        pass
    return cfg

def normalize_base_url(b):
    s = (b or "").strip().strip("`").strip('"').strip("'")
    return s or "https://dashscope.aliyuncs.com"

def compose_path(base_path, endpoint):
    bp = (base_path or "")
    if bp and not bp.startswith("/"):
        bp = "/" + bp
    bp = bp.rstrip("/")
    ep = endpoint if endpoint.startswith("/") else "/" + endpoint
    if bp.endswith("/api/v1") and ep.startswith("/api/v1/"):
        ep = ep[len("/api/v1"):]
    return (bp + ep) if bp else ep

def main():
    cfg = load_config()
    base = normalize_base_url(cfg["base_url"])
    u = urlparse(base)
    host = u.netloc or base.replace("https://", "").replace("http://", "")
    path = compose_path(u.path, "/compatible-mode/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "你是健康检查助手"},
            {"role": "user", "content": "请用一句话确认服务可用"},
        ],
        "temperature": 0.4
    }
    status, data = https_json_request(host, path, headers, payload)
    print("Status:", status)
    print("Response:", data)

if __name__ == "__main__":
    main()
