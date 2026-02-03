const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const PORT = process.env.PORT ? Number(process.env.PORT) : 8000;

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf-8');
        resolve(raw ? JSON.parse(raw) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
}

function serveStatic(req, res) {
  const u = new URL(req.url, `http://${req.headers.host}`);
  let filePath = path.join(__dirname, u.pathname === '/' ? 'index.html' : u.pathname);
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const type =
      ext === '.html' ? 'text/html; charset=utf-8' :
      ext === '.css' ? 'text/css; charset=utf-8' :
      ext === '.js' ? 'application/javascript; charset=utf-8' :
      'application/octet-stream';
    res.writeHead(200, { 'Content-Type': type });
    res.end(data);
  });
}

function httpsJsonRequest(options, payload) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf-8');
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            reject(new Error(`Invalid JSON response: ${e.message}\n${body}`));
          }
        } else {
          reject(new Error(`HTTP ${res.statusCode}: ${body}`));
        }
      });
    });
    req.on('error', reject);
    if (payload) {
      req.write(JSON.stringify(payload));
    }
    req.end();
  });
}

async function callOpenAI(messages, model) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('缺少 OPENAI_API_KEY 环境变量');
  const payload = {
    model: model || 'gpt-4o-mini',
    messages,
    temperature: 0.4,
  };
  const options = {
    method: 'POST',
    hostname: 'api.openai.com',
    path: '/v1/chat/completions',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  };
  const data = await httpsJsonRequest(options, payload);
  const content = data?.choices?.[0]?.message?.content || '';
  return content;
}

async function callQwen(messages, model) {
  const apiKey = process.env.DASHSCOPE_API_KEY;
  if (!apiKey) throw new Error('缺少 DASHSCOPE_API_KEY 环境变量');
  const payload = {
    model: model || 'qwen-plus',
    input: { messages },
  };
  const options = {
    method: 'POST',
    hostname: 'dashscope.aliyuncs.com',
    path: '/api/v1/chat/completions',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  };
  const data = await httpsJsonRequest(options, payload);
  const content = data?.output?.choices?.[0]?.message?.content || data?.output_text || '';
  return content;
}

async function runDiscussion({ topic, question, models, modelOptions }) {
  const baseSystem = {
    role: 'system',
    content: `你是参与方案讨论的专家。请围绕主题进行分析，提出可执行的方案步骤、注意事项与潜在风险。输出请结构化分点说明。主题：${topic}`,
  };
  const userMsg = { role: 'user', content: question };
  const results = [];
  const tasks = (models || []).map((m) => {
    const msgs = [baseSystem, userMsg];
    const modelName = modelOptions?.[m]?.model;
    if (m === 'openai') {
      return callOpenAI(msgs, modelName).then((content) => results.push({ model: 'openai', content })).catch((e) => results.push({ model: 'openai', error: String(e) }));
    }
    if (m === 'qwen') {
      return callQwen(msgs, modelName).then((content) => results.push({ model: 'qwen', content })).catch((e) => results.push({ model: 'qwen', error: String(e) }));
    }
    results.push({ model: m, error: '未知模型' });
    return Promise.resolve();
  });
  await Promise.all(tasks);
  return results;
}

async function runSummary({ topic, question, discussions, summarizer, modelOptions }) {
  const summarySystem = {
    role: 'system',
    content:
      '你是首席总结官。请综合多位专家的讨论，找出共识与分歧，制定一个清晰的、可执行的最终方案：包括目标、关键步骤、资源与角色、时间计划、风险与应对、验收标准。输出中文分点，最后给出一个简短执行清单。',
  };
  const combined = discussions
    .map((d, idx) => `【#${idx + 1} ${d.model}】\n${d.error ? ('错误：' + d.error) : d.content}`)
    .join('\n\n');
  const userMsg = {
    role: 'user',
    content: `主题：${topic}\n问题：${question}\n以下是讨论内容，请总结并给出最终方案：\n\n${combined}`,
  };
  const msgs = [summarySystem, userMsg];
  const modelName = modelOptions?.[summarizer]?.model;
  if (summarizer === 'openai') {
    return callOpenAI(msgs, modelName);
  }
  if (summarizer === 'qwen') {
    return callQwen(msgs, modelName);
  }
  throw new Error('未知总结模型');
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  if (req.method === 'POST' && u.pathname === '/api/discuss') {
    try {
      const body = await readBody(req);
      const { topic, question, models, summarizer, modelOptions } = body || {};
      if (!topic || !question || !Array.isArray(models) || models.length === 0 || !summarizer) {
        return sendJson(res, 400, { error: '缺少必要参数：topic, question, models[], summarizer' });
      }
      const discussions = await runDiscussion({ topic, question, models, modelOptions });
      let summary = '';
      try {
        summary = await runSummary({ topic, question, discussions, summarizer, modelOptions });
      } catch (e) {
        summary = `总结失败：${String(e)}`;
      }
      return sendJson(res, 200, { discussions, summary });
    } catch (e) {
      return sendJson(res, 500, { error: String(e) });
    }
  }
  serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log('请确保已设置环境变量 OPENAI_API_KEY 和/或 DASHSCOPE_API_KEY');
});

