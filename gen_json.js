// 从 index.html 提取 BUILTIN_DATA + PROVIDERS，按当前汇率生成 llm_prices.json
const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');

function extract(name) {
  const start = html.indexOf(`const ${name} = [`);
  const end = html.indexOf('\n];', start);
  if (start < 0 || end < 0) throw new Error('extract failed: ' + name);
  return new Function('return ' + html.slice(start + `const ${name} = `.length, end + 2))();
}

const PROVIDERS = extract('PROVIDERS');
const DATA = extract('BUILTIN_DATA');
const RATE = 6.78;
const nameMap = Object.fromEntries(PROVIDERS.map(p => [p.id, p.name]));
const isCNY = Object.fromEntries(PROVIDERS.map(p => [p.id, p.nativeCurrency === 'CNY']));

const r4 = x => Math.round(x * 10000) / 10000;
const r6 = x => Math.round(x * 1000000) / 1000000;

const models = [];
const seen = new Set();
for (const d of DATA) {
  const key = d.provider + '|' + d.model;
  if (seen.has(key)) continue;
  seen.add(key);
  const native = isCNY[d.provider];
  const inputCNY = native ? d.pIn : r6(d.pIn * RATE);
  const outputCNY = native ? d.pOut : r6(d.pOut * RATE);
  const inputUSD = native ? r4(d.pIn / RATE) : d.pIn;
  const outputUSD = native ? r4(d.pOut / RATE) : d.pOut;
  let cached = 0;
  if (d.cRatio > 0 && d.cRatio < 1) cached = r6(inputCNY * d.cRatio);
  // DeepSeek 缓存为官方绝对值
  if (d.provider === 'deepseek' && /V4 (Pro|Flash)/.test(d.model)) {
    cached = d.model.includes('Pro') ? 0.15 : 0.05;
  }
  models.push({
    provider: nameMap[d.provider],
    model: d.model,
    input_price: inputCNY,
    output_price: outputCNY,
    context_window: d.ctx || '',
    notes: d.note || '',
    cached_input_price: cached,
    input_usd: inputUSD,
    output_usd: outputUSD,
  });
}

// 按 PROVIDERS 顺序 + 模型名排序
const order = Object.fromEntries(PROVIDERS.map((p, i) => [p.id, i]));
models.sort((a, b) => {
  const i = order[a.provider] - order[b.provider];
  return i !== 0 ? i : a.model.localeCompare(b.model);
});

const now = new Date();
const pad = n => String(n).padStart(2, '0');
const updated = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}.000000`;

const out = {
  updated,
  currency: 'CNY (人民币)',
  unit: '每百万 Token',
  models,
};

fs.writeFileSync('llm_prices.json', JSON.stringify(out, null, 2) + '\n', 'utf8');
console.log(`✅ 生成 llm_prices.json: ${models.length} 个模型 / ${new Set(models.map(m => m.provider)).size} 个厂商 / 汇率 ${RATE}`);
