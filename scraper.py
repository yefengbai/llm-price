"""
LLM Token 价格爬虫 — 抓取主流大语言模型的 API 定价
所有价格统一以人民币 CNY (¥) 显示
数据源: 各厂商官方定价页面 + API 端点
"""

import json
import re
import time
import csv
import sys
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

import requests
from bs4 import BeautifulSoup

# ── 汇率 ────────────────────────────────────────────────
# 默认美元 → 人民币汇率 (实时抓取失败时的兜底值, 2026-08 中价 6.78)
DEFAULT_USD_TO_CNY = 6.78


def fetch_exchange_rate() -> float:
    """获取最新美元兑人民币汇率"""
    try:
        # 免费汇率 API ( exchangerate-api / frankfurter )
        resp = requests.get(
            "https://api.exchangerate-api.com/v4/latest/USD",
            timeout=8
        )
        data = resp.json()
        rate = data["rates"].get("CNY", 0)
        if rate:
            print(f"💱 当前汇率: 1 USD = {rate:.4} CNY")
            return rate
    except Exception:
        pass

    # 兜底方案
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=CNY",
            timeout=8
        )
        data = resp.json()
        rate = data["rates"].get("CNY", 0)
        if rate:
            print(f"💱 当前汇率: 1 USD = {rate:.4} CNY")
            return rate
    except Exception:
        pass

    print(f"⚠ 汇率 API 不可用，使用默认汇率: 1 USD = {DEFAULT_USD_TO_CNY} CNY")
    return DEFAULT_USD_TO_CNY


# ── 数据模型 ────────────────────────────────────────────


@dataclass
class ModelPrice:
    provider: str
    model: str
    input_price: float          # ¥/1M tokens (人民币)
    output_price: float          # ¥/1M tokens (人民币)
    context_window: str = ""
    notes: str = ""
    cached_input_price: float = 0  # ¥/1M tokens (缓存命中价)

    # 保留美元原始值方便对比
    input_usd: float = 0
    output_usd: float = 0


# ── 请求工具 ────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

TIMEOUT = 15


def fetch_html(url: str) -> Optional[str]:
    """抓取页面 HTML，带重试"""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"  ⚠ 尝试 {attempt+1}/3 失败: {url} — {e}")
            time.sleep(2 ** attempt)
    return None


# ── 价格转换工具 ────────────────────────────────────────

def usd_to_cny(usd: float, rate: float) -> float:
    return round(usd * rate, 4)


def make_price(provider: str, model: str, input_usd: float, output_usd: float,
               context: str = "", notes: str = "", rate: float = DEFAULT_USD_TO_CNY,
               cached_input_usd: float = 0) -> ModelPrice:
    """创建 ModelPrice，同时计算人民币价格"""
    return ModelPrice(
        provider=provider,
        model=model,
        input_price=usd_to_cny(input_usd, rate),
        output_price=usd_to_cny(output_usd, rate),
        context_window=context,
        notes=notes,
        cached_input_price=usd_to_cny(cached_input_usd, rate),
        input_usd=input_usd,
        output_usd=output_usd,
    )


# ── 各厂商爬虫 ──────────────────────────────────────────

class BaseScraper:
    provider: str = ""

    def __init__(self, rate: float = DEFAULT_USD_TO_CNY):
        self.rate = rate

    def scrape(self) -> list[ModelPrice]:
        raise NotImplementedError

    def usd_price(self, input_usd: float, output_usd: float,
                  context: str = "", notes: str = "",
                  cached_input_usd: float = 0) -> ModelPrice:
        return make_price(self.provider, "", input_usd, output_usd,
                          context, notes, self.rate, cached_input_usd)

    @staticmethod
    def _clean_price(text: str) -> float:
        """ "$2.50 / 1M tokens" → 2.5 或 "¥1.00" → 1.0 """
        m = re.search(r'[¥$]?\s*([\d.]+)', str(text))
        return float(m.group(1)) if m else 0.0

    @staticmethod
    def _parse_table_rows(html: str, table_selector: str) -> list[list[str]]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one(table_selector)
        if not table:
            return []
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        return rows


class OpenAIScraper(BaseScraper):
    """OpenAI 官方定价"""

    provider = "OpenAI"
    URL = "https://openai.com/api/pricing/"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()

        soup = BeautifulSoup(html, "lxml")
        results = []

        for script in soup.find_all("script"):
            if not script.string:
                continue
            text = script.string
            for pattern in [r'"gpt-[\w.-]+"', r'"o[\w-]*mini"']:
                if re.search(pattern, text, re.IGNORECASE):
                    prices = self._extract_from_json(text)
                    results.extend(prices)

        if not results:
            print(f"[{self.provider}] 页面解析无结果，使用离线数据")
            return self._offline_data()

        return results

    def _extract_from_json(self, text: str) -> list[ModelPrice]:
        results = []
        pattern = r'"([\w.-]+(?:gpt|o\d)[\w.-]*)".*?"input[_\s]?price"[:\s]*([\d.]+).*?"output[_\s]?price"[:\s]*([\d.]+)'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            model = m.group(1)
            inp = float(m.group(2))
            out = float(m.group(3))
            results.append(make_price(self.provider, model, inp, out, rate=self.rate))
        return results

    def _offline_data(self) -> list[ModelPrice]:
        return [
            make_price("OpenAI", "GPT-5.5 Pro",        30.00, 180.00, "1M",   "顶级旗舰",         self.rate, cached_input_usd=3.00),
            make_price("OpenAI", "GPT-5.5",             5.00, 30.00, "1M",    "最新旗舰",        self.rate, cached_input_usd=0.50),
            make_price("OpenAI", "GPT-5.5 Mini",        0.50,  3.00, "1M",    "旗舰轻量",        self.rate),
            make_price("OpenAI", "GPT-5.6 Sol",         5.00, 30.00, "1M",    "最新GA/同5.5",    self.rate, cached_input_usd=0.50),
            make_price("OpenAI", "GPT-5.6 Terra",       2.00, 12.00, "1M",    "最新GA/性价比",   self.rate, cached_input_usd=0.20),
            make_price("OpenAI", "GPT-5.6 Luna",        0.20,  1.20, "1M",    "最新GA/轻量",     self.rate),
            make_price("OpenAI", "GPT-5",               1.25, 10.00, "272K",  "性价比旗舰",      self.rate),
            make_price("OpenAI", "GPT-5.4",             2.50, 15.00, "1M",    "推荐中端",        self.rate, cached_input_usd=0.25),
            make_price("OpenAI", "GPT-5.4 Mini",        0.75,  4.50, "400K",  "轻量",            self.rate),
            make_price("OpenAI", "GPT-5.4 Nano",        0.20,  1.25, "400K",  "入门",            self.rate),
            make_price("OpenAI", "GPT-4.1",             2.00,  8.00, "1M",    "长上下文",         self.rate),
            make_price("OpenAI", "GPT-4.1 Mini",        0.40,  1.60, "1M",    "",                self.rate),
            make_price("OpenAI", "GPT-4.1 Nano",        0.10,  0.40, "1M",    "",                self.rate),
            make_price("OpenAI", "o4",                 10.00, 40.00, "200K",  "推理旗舰",         self.rate),
            make_price("OpenAI", "o4-mini",             1.10,  4.40, "200K",  "推理轻量",         self.rate),
            make_price("OpenAI", "o3-pro",             20.00, 80.00, "200K",  "推理旧旗舰",       self.rate),
            make_price("OpenAI", "o3-mini",             1.10,  4.40, "200K",  "旧推理中端",       self.rate),
        ]


class AnthropicScraper(BaseScraper):
    """Anthropic Claude 定价"""

    provider = "Anthropic"
    URL = "https://docs.anthropic.com/en/docs/about-claude/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()

        rows = self._parse_table_rows(html, "table")
        if not rows:
            rows = self._parse_table_rows(html, ".table-wrapper table")

        results = []
        for row in rows:
            if len(row) < 3:
                continue
            model_name = row[0]
            if not any(kw in model_name.lower() for kw in ["claude", "haiku", "sonnet", "opus"]):
                continue
            inp = self._clean_price(row[1]) if len(row) > 1 else 0
            out = self._clean_price(row[2]) if len(row) > 2 else 0
            results.append(make_price(self.provider, model_name, inp, out, rate=self.rate))
        return results or self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        return [
            make_price("Anthropic", "Claude Fable 5",       5.00, 25.00, "1M",   "最新旗舰",      self.rate, cached_input_usd=0.50),
            make_price("Anthropic", "Claude Opus 4.8",      5.00, 25.00, "1M",   "旗舰",          self.rate, cached_input_usd=0.50),
            make_price("Anthropic", "Claude Opus 4.8 Fast", 10.00, 50.00, "1M",   "快速模式2.5x",  self.rate, cached_input_usd=1.00),
            make_price("Anthropic", "Claude Opus 4.7",      5.00, 25.00, "1M",   "旧旗舰",        self.rate, cached_input_usd=0.50),
            make_price("Anthropic", "Claude Sonnet 4.6",    3.00, 15.00, "1M",   "生产推荐",      self.rate, cached_input_usd=0.30),
            make_price("Anthropic", "Claude Sonnet 4.5",    3.00, 15.00, "200K", "",              self.rate),
            make_price("Anthropic", "Claude Sonnet 4",      3.00, 15.00, "200K", "旧版生产",      self.rate),
            make_price("Anthropic", "Claude Haiku 4.5",     1.00,  5.00, "200K", "轻量",          self.rate),
            make_price("Anthropic", "Claude Haiku 3.5",     0.80,  4.00, "200K", "",              self.rate),
            make_price("Anthropic", "Claude Haiku 3",       0.25,  1.25, "200K", "极致便宜/将退役", self.rate),
            make_price("Anthropic", "Claude 3 Opus",       15.00, 75.00, "200K", "旧旗舰",        self.rate),
            make_price("Anthropic", "Mythos Preview",      25.00, 125.00, "1M",  "研究预览/邀请制", self.rate),
        ]


class GoogleScraper(BaseScraper):
    """Google Gemini 定价"""

    provider = "Google"
    URL = "https://ai.google.dev/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()

        soup = BeautifulSoup(html, "lxml")
        results = []
        text = soup.get_text()

        model_pattern = r'(Gemini\s*[\d.]+\s*\w*(?:\s*(?:Pro|Flash|Lite|Nano))?)'
        for m in re.finditer(model_pattern, text, re.IGNORECASE):
            model = m.group(1).strip()
            end = min(len(text), m.end() + 500)
            snippet = text[m.end():end]
            prices = re.findall(r'[¥$]?\s*([\d.]+)\s*/\s*(?:1M|million|百万)', snippet)
            if len(prices) >= 2:
                results.append(make_price(
                    self.provider, model,
                    float(prices[0]), float(prices[1]),
                    rate=self.rate
                ))
        return results or self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        return [
            make_price("Google", "Gemini 3.5 Pro",            3.50, 21.00, "2M",   "最新旗舰",       self.rate),
            make_price("Google", "Gemini 3.6 Flash",          1.50,  7.50, "1M",   "最新(7月)/输出降17%", self.rate, cached_input_usd=0.15),
            make_price("Google", "Gemini 3.5 Flash",          1.50,  9.00, "1M",   "最新",           self.rate, cached_input_usd=0.15),
            make_price("Google", "Gemini 3.5 Flash-Lite",     0.30,  2.50, "1M",   "最新轻量",       self.rate),
            make_price("Google", "Gemini 3.1 Pro",            2.00, 12.00, "2M",   "旗舰",           self.rate),
            make_price("Google", "Gemini 3 Flash",            0.50,  3.00, "1M",   "性价比",         self.rate),
            make_price("Google", "Gemini 3.1 Flash Lite",     0.25,  1.50, "1M",   "最便宜",         self.rate),
            make_price("Google", "Gemini 2.5 Pro",            1.25, 10.00, "1M",   "",               self.rate),
            make_price("Google", "Gemini 2.5 Flash",          0.30,  2.50, "1M",   "",               self.rate),
            make_price("Google", "Gemini 2.5 Flash-Lite",     0.10,  0.40, "1M",   "旧版/10月退役",   self.rate),
        ]


class DeepSeekScraper(BaseScraper):
    """DeepSeek API 定价 — 官网以人民币标价"""

    provider = "DeepSeek"
    URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()

        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text()

        # DeepSeek 官网以人民币定价，格式类似 "¥1 / 百万 tokens"
        results = []
        model_pattern = r'[Dd]eep[Ss]eek[-\s]*(?:Chat|R1|V\d[\d.]*)[^\n]*'
        for m in re.finditer(model_pattern, text):
            model = m.group(0).strip()[:50]
            end = min(len(text), m.end() + 600)
            snippet = text[m.end():end]
            prices = re.findall(r'[¥$]?\s*([\d.]+)\s*/?\s*(?:百万|1M|million)', snippet)
            if len(prices) >= 2:
                # DeepSeek 本身标的是人民币
                results.append(ModelPrice(
                    provider=self.provider, model=model,
                    input_price=float(prices[0]),
                    output_price=float(prices[1]),
                    input_usd=round(float(prices[0]) / self.rate, 4),
                    output_usd=round(float(prices[1]) / self.rate, 4),
                ))
        return results or self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        # DeepSeek 官网以人民币标价，直接用 CNY 值构造 ModelPrice
        # input_price / output_price 字段存人民币，input_usd / output_usd 存美元等价
        # 2026-08-17 起峰谷分时定价: 下列为空闲时段价, 高峰时段(9-12/14-18点)x2
        rate = self.rate
        return [
            ModelPrice(
                provider="DeepSeek", model="DeepSeek V4 Pro",
                input_price=4.50, output_price=13.50, context_window="1M",
                notes="🔥空闲价(高峰¥9/27)", cached_input_price=0.15,
                input_usd=round(4.50/rate, 4), output_usd=round(13.50/rate, 4),
            ),
            ModelPrice(
                provider="DeepSeek", model="DeepSeek V4 Flash",
                input_price=1.50, output_price=4.50, context_window="1M",
                notes="空闲价(高峰¥3/9)", cached_input_price=0.05,
                input_usd=round(1.50/rate, 4), output_usd=round(4.50/rate, 4),
            ),
            ModelPrice(
                provider="DeepSeek", model="DeepSeek V4 Flash Vision Exp",
                input_price=1.50, output_price=4.50, context_window="1M",
                notes="✨视觉实验版/与V4 Flash同价", cached_input_price=0.05,
                input_usd=round(1.50/rate, 4), output_usd=round(4.50/rate, 4),
            ),
            ModelPrice(
                provider="DeepSeek", model="DeepSeek V3.2",
                input_price=2.00, output_price=3.00, context_window="164K",
                notes="", cached_input_price=0,
                input_usd=round(2.00/rate, 4), output_usd=round(3.00/rate, 4),
            ),
            ModelPrice(
                provider="DeepSeek", model="DeepSeek R1-0528",
                input_price=4.00, output_price=16.00, context_window="164K",
                notes="推理模型", cached_input_price=0,
                input_usd=round(4.00/rate, 4), output_usd=round(16.00/rate, 4),
            ),
            ModelPrice(
                provider="DeepSeek", model="DeepSeek Coder V2",
                input_price=1.00, output_price=2.00, context_window="128K",
                notes="代码专用", cached_input_price=0,
                input_usd=round(1.00/rate, 4), output_usd=round(2.00/rate, 4),
            ),
        ]


class CloudPriceScraper(BaseScraper):
    """聚合站: https://cloudprice.net/models"""

    provider = "CloudPrice"
    URL = "https://cloudprice.net/models"

    def scrape(self) -> list[ModelPrice]:
        print(f"[聚合] 正在抓取 CloudPrice ...")
        html = fetch_html(self.URL)
        if not html:
            return []
        rows = self._parse_table_rows(html, "table")
        results = []
        for row in rows:
            if len(row) < 4:
                continue
            provider = row[0]
            model = row[1] if len(row) > 1 else ""
            inp = self._clean_price(row[2]) if len(row) > 2 else 0
            out = self._clean_price(row[3]) if len(row) > 3 else 0
            results.append(make_price(provider, model, inp, out, rate=self.rate))
        return results


class MistralScraper(BaseScraper):
    """Mistral API 定价 (USD)"""

    provider = "Mistral"
    URL = "https://mistral.ai/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()
        # 官网为动态渲染页面，暂以离线数据为准
        return self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        return [
            make_price("Mistral", "Mistral Large 3",     0.50,  1.50, "256K", "推荐",       self.rate),
            make_price("Mistral", "Mistral Large 2",     2.00,  6.00, "128K", "上代旗舰",   self.rate),
            make_price("Mistral", "Mistral Medium 3.5",  1.50,  7.50, "256K", "最新中端旗舰", self.rate),
            make_price("Mistral", "Mistral Medium 3.1",  0.40,  2.00, "131K", "中端",       self.rate),
            make_price("Mistral", "Mistral Small 4",     0.15,  0.60, "32K",  "小模型",     self.rate),
            make_price("Mistral", "Mistral Small 3.1",   0.10,  0.30, "128K", "性价比轻量", self.rate),
            make_price("Mistral", "Mistral Tiny",        0.15,  0.40, "32K",  "极致便宜",   self.rate),
            make_price("Mistral", "Codestral",           0.30,  0.90, "32K",  "代码",       self.rate),
            make_price("Mistral", "Mistral Nemo",        0.02,  0.03, "131K", "极致便宜/最小", self.rate),
            make_price("Mistral", "Ministral 3 8B",      0.15,  0.15, "262K", "边缘端侧",   self.rate),
        ]


class XAIScraper(BaseScraper):
    """xAI Grok API 定价 (USD)"""

    provider = "xAI"
    URL = "https://docs.x.ai/docs/models"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()
        return self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        return [
            make_price("xAI", "Grok 4.5",      2.00,  6.00, "256K", "最新旗舰", self.rate),
            make_price("xAI", "Grok-4.3",      1.25,  2.50, "1M",   "最新/推荐", self.rate, cached_input_usd=0.20),
            make_price("xAI", "Grok Build 0.1", 1.00, 2.00, "1M",   "新版尝鲜", self.rate, cached_input_usd=0.16),
            make_price("xAI", "Grok 4.1",      0.75,  1.50, "128K", "上代/性价比", self.rate),
        ]


class CohereScraper(BaseScraper):
    """Cohere API 定价 (USD)"""

    provider = "Cohere"
    URL = "https://cohere.com/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()
        return self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        return [
            make_price("Cohere", "Command R+", 1.50, 4.50, "128K", "企业级长文本", self.rate),
        ]


class AlibabaScraper(BaseScraper):
    """阿里云通义千问 API 定价 (CNY)"""

    provider = "阿里云"
    URL = "https://help.aliyun.com/zh/model-studio/models"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()
        return self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        rate = self.rate
        return [ModelPrice(
            provider=self.provider, model=m, input_price=i, output_price=o,
            context_window=ctx, notes=note,
            input_usd=round(i/rate, 4), output_usd=round(o/rate, 4),
        ) for m, i, o, ctx, note in [
            ("Qwen3.7-Max",  6.00, 18.00, "256K", "🔥5折!原¥12/36"),
            ("Qwen3-Max",    2.50, 10.00, "252K", "上代旗舰"),
            ("Qwen3.6-Plus", 2.00, 12.00, "1M",   "最新推荐"),
            ("Qwen3.5-Plus", 0.80,  4.80, "1M",   "推荐(上代)"),
            ("Qwen-Plus",    0.80,  2.00, "131K", "旧推荐"),
            ("Qwen-Turbo",   0.37,  1.47, "1M",   "极低价"),
            ("Qwen-Coder",   1.00,  4.00, "1M",   "代码"),
            ("Qwen-Long",    0.50,  2.00, "10M",  "超长上下文"),
        ]]


class ZhipuScraper(BaseScraper):
    """智谱 GLM API 定价 (CNY)"""

    provider = "智谱AI"
    URL = "https://open.bigmodel.cn/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()
        return self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        rate = self.rate
        return [ModelPrice(
            provider=self.provider, model=m, input_price=i, output_price=o,
            context_window=ctx, notes=note,
            input_usd=round(i/rate, 4), output_usd=round(o/rate, 4),
        ) for m, i, o, ctx, note in [
            ("GLM-5.2",     8.00, 28.00, "1M",   "🔥最新旗舰"),
            ("GLM-5.1",     6.00, 24.00, "200K", "🔥顶级编程"),
            ("GLM-5",       4.00, 18.00, "128K", "旗舰"),
            ("GLM-4-Plus",  5.00,  5.00, "128K", "推荐"),
            ("GLM-4-Air",   0.60,  0.60, "128K", "轻量"),
            ("GLM-4-Flash", 0.00,  0.00, "128K", "免费"),
            ("CodeGeeX",    0.00,  0.00, "128K", "代码/免费"),
        ]]


class MoonshotScraper(BaseScraper):
    """月之暗面 Kimi API 定价 (CNY)"""

    provider = "月之暗面"
    URL = "https://platform.moonshot.cn/docs/pricing"

    def scrape(self) -> list[ModelPrice]:
        print(f"[{self.provider}] 正在抓取 {self.URL} ...")
        html = fetch_html(self.URL)
        if not html:
            print(f"[{self.provider}] 抓取失败，使用离线数据")
            return self._offline_data()
        return self._offline_data()

    def _offline_data(self) -> list[ModelPrice]:
        rate = self.rate
        return [ModelPrice(
            provider=self.provider, model=m, input_price=i, output_price=o,
            context_window=ctx, notes=note, cached_input_price=cached,
            input_usd=round(i/rate, 4), output_usd=round(o/rate, 4),
        ) for m, i, o, ctx, note, cached in [
            ("Kimi-K2.6",       6.50, 27.00, "262K", "🔥最新Agent旗舰", 1.10),
            ("Moonshot-v1-128k", 5.00, 10.00, "128K", "旧版", 0),
        ]]


# ── 输出格式化 ──────────────────────────────────────────

def print_table(prices: list[ModelPrice], rate: float = DEFAULT_USD_TO_CNY):
    """Rich 终端表格 / 纯文本降级"""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🤖 LLM Token 价格总览 (每百万 Token / 人民币 ¥)", header_style="bold cyan")
        table.add_column("厂商", style="magenta", width=12)
        table.add_column("模型", style="green", width=28)
        table.add_column("输入 ¥/1M", justify="right", style="yellow", width=14)
        table.add_column("输出 ¥/1M", justify="right", style="red", width=14)
        table.add_column("≈输入 $", justify="right", style="dim", width=10)
        table.add_column("≈输出 $", justify="right", style="dim", width=10)
        table.add_column("上下文", justify="center", width=10)
        table.add_column("备注", style="dim", width=20)

        for p in sorted(prices, key=lambda x: (x.provider, x.input_price)):
            table.add_row(
                p.provider, p.model,
                f"¥{p.input_price:.4f}", f"¥{p.output_price:.4f}",
                f"${p.input_usd:.4f}" if p.input_usd else "",
                f"${p.output_usd:.4f}" if p.output_usd else "",
                p.context_window, p.notes,
            )
        console.print(table)
    except ImportError:
        print_table_simple(prices, rate)


def print_table_simple(prices: list[ModelPrice], rate: float = DEFAULT_USD_TO_CNY):
    """纯文本降级表格"""
    print()
    header = f"{'厂商':<12} {'模型':<28} {'输入¥/1M':>10} {'输出¥/1M':>10} {'≈输入$':>8} {'≈输出$':>8} {'上下文':>8}  {'备注'}"
    print("=" * len(header))
    print("🤖 LLM Token 价格总览 (每百万 Token / 人民币 ¥)")
    print(f"   汇率: 1 USD ≈ {rate:.2f} CNY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for p in sorted(prices, key=lambda x: (x.provider, x.input_price)):
        print(f"{p.provider:<12} {p.model:<28} ¥{p.input_price:>8.4f}  ¥{p.output_price:>8.4f}"
              f"  ${p.input_usd:>6.4f}  ${p.output_usd:>6.4f}  {p.context_window:>6}  {p.notes}")
    print("=" * len(header))


def to_json(prices: list[ModelPrice], pretty: bool = True) -> str:
    data = {
        "updated": datetime.now().isoformat(),
        "currency": "CNY (人民币)",
        "unit": "每百万 Token",
        "models": [asdict(p) for p in prices],
    }
    return json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)


def to_csv(prices: list[ModelPrice], filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "provider", "model", "input_price_cny", "output_price_cny",
            "input_price_usd", "output_price_usd",
            "context_window", "cached_input_price_cny", "notes"
        ])
        writer.writeheader()
        for p in prices:
            writer.writerow({
                "provider": p.provider,
                "model": p.model,
                "input_price_cny": p.input_price,
                "output_price_cny": p.output_price,
                "input_price_usd": p.input_usd,
                "output_price_usd": p.output_usd,
                "context_window": p.context_window,
                "cached_input_price_cny": p.cached_input_price,
                "notes": p.notes,
            })
    print(f"✅ CSV 已保存到: {filepath}")


# ── 主流程 ──────────────────────────────────────────────

def scrape_all(rate: float) -> list[ModelPrice]:
    all_prices: list[ModelPrice] = []

    scrapers: list[BaseScraper] = [
        OpenAIScraper(rate),
        AnthropicScraper(rate),
        GoogleScraper(rate),
        DeepSeekScraper(rate),
        MistralScraper(rate),
        XAIScraper(rate),
        CohereScraper(rate),
        AlibabaScraper(rate),
        ZhipuScraper(rate),
        MoonshotScraper(rate),
        CloudPriceScraper(rate),
    ]

    for scraper in scrapers:
        try:
            prices = scraper.scrape()
            all_prices.extend(prices)
            print(f"  ✅ {scraper.provider}: 获取 {len(prices)} 个模型价格")
        except Exception as e:
            print(f"  ❌ {scraper.provider}: 异常 — {e}")

    # 去重
    seen = set()
    unique = []
    for p in all_prices:
        key = (p.provider, p.model)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Token 价格爬虫 — 人民币计价")
    parser.add_argument("--offline", action="store_true",
                        help="仅使用离线数据，不发起网络请求")
    parser.add_argument("--json", action="store_true",
                        help="输出 JSON 格式")
    parser.add_argument("--csv", type=str, nargs="?", const="llm_prices.csv",
                        help="导出 CSV (可选指定路径)")
    parser.add_argument("--save-json", type=str, nargs="?", const="llm_prices.json",
                        help="保存 JSON 文件 (可选指定路径)")
    parser.add_argument("--provider", type=str,
                        choices=["openai", "anthropic", "google", "deepseek"],
                        help="只抓取指定厂商")
    parser.add_argument("--rate", type=float, default=0,
                        help="手动指定 USD→CNY 汇率 (默认自动获取)")

    args = parser.parse_args()

    print("🔍 LLM Token 价格爬虫启动 (人民币计价)\n")

    # 获取汇率
    if args.rate > 0:
        rate = args.rate
        print(f"💱 使用指定汇率: 1 USD = {rate} CNY")
    elif args.offline:
        rate = DEFAULT_USD_TO_CNY
        print(f"💱 离线模式，使用默认汇率: 1 USD = {rate} CNY")
    else:
        rate = fetch_exchange_rate()
    print()

    if args.offline:
        print("📴 离线模式 — 仅使用内置数据\n")
        all_prices = []
        for scraper in [OpenAIScraper(rate), AnthropicScraper(rate),
                        GoogleScraper(rate), DeepSeekScraper(rate),
                        MistralScraper(rate), XAIScraper(rate),
                        CohereScraper(rate), AlibabaScraper(rate),
                        ZhipuScraper(rate), MoonshotScraper(rate)]:
            all_prices.extend(scraper._offline_data())
    elif args.provider:
        scraper_map = {
            "openai": OpenAIScraper,
            "anthropic": AnthropicScraper,
            "google": GoogleScraper,
            "deepseek": DeepSeekScraper,
        }
        scraper = scraper_map[args.provider](rate)
        all_prices = scraper.scrape()
    else:
        all_prices = scrape_all(rate)

    if not all_prices:
        print("❌ 未获取到任何价格数据")
        return 1

    # 输出
    if args.json:
        print(to_json(all_prices))
    else:
        print_table(all_prices, rate)

    if args.csv:
        filepath = args.csv if isinstance(args.csv, str) else "llm_prices.csv"
        to_csv(all_prices, filepath)

    if args.save_json:
        filepath = args.save_json if isinstance(args.save_json, str) else "llm_prices.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(to_json(all_prices))
        print(f"✅ JSON 已保存到: {filepath}")

    providers = set(p.provider for p in all_prices)
    print(f"\n📊 共 {len(all_prices)} 个模型 | {len(providers)} 个厂商 | 汇率 1 USD ≈ {rate:.2f} CNY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
