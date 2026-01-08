import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import gspread
from google.oauth2.service_account import Credentials

import holidays
import requests

JST = ZoneInfo("Asia/Tokyo")


# --- 日本語化・表示変換（出力用） -----------------

SECTOR_JP_MAP = {
    "Basic Materials": "素材",
    "Communication Services": "通信サービス",
    "Consumer Cyclical": "一般消費財",
    "Consumer Defensive": "生活必需品",
    "Energy": "エネルギー",
    "Financial Services": "金融",
    "Healthcare": "ヘルスケア",
    "Industrials": "資本財・サービス",
    "Real Estate": "不動産",
    "Technology": "情報技術",
    "Utilities": "公益事業",
}

# Industryは粒度が細かく全網羅は難しいため、代表的なものだけマップし、未定義は原文を出す
INDUSTRY_JP_MAP = {
    "Conglomerates": "複合企業",
    "Oil & Gas E&P": "石油・ガス（探鉱・生産）",
}

REASON_JP_MAP = {
    "SECTOR_UNKNOWN": "セクター取得不可",
    "FINANCIAL_SECTOR_EXCLUDED": "金融セクター除外",
    "NO_MCAP": "時価総額取得不可",
    "SIZE<30億": "時価総額30億未満",
    "NO_ADV3M": "平均日次売買代金（3か月）取得不可",
    "LIQ<1000万": "平均日次売買代金1,000万円未満",
    "NO_OCF2Y": "営業CF（直近2年）取得不可",
    "NO_DIV_YIELD": "配当利回り取得不可",
    "NO_CASH": "現金（Cash）取得不可",
    "NO_DEBT": "有利子負債（Debt）取得不可",
    "NO_ADJNCAV_INPUT": "Adjusted NCAV計算に必要なB/S項目不足",
    "SPLIT_3Y": "過去3年に分割/併合あり（加点無効）",
    "NO_SHARES": "発行済株式数推移（Shares）取得不可",
    "UPGRADED_BY_SHARES_SCORE": "株数減少スコアで△→◯に昇格",
    "EXCEPTION:": "例外発生",
    "CONDITION_NOT_MET": "条件未達",
}


def _to_oku(v):
    """円→億円（表示用）。None/空はそのまま。"""
    if v is None:
        return None
    try:
        # "123,456" 等にも耐える
        if isinstance(v, str):
            v = v.replace(",", "").strip()
            if v == "":
                return None
        return float(v) / 1e8
    except Exception:
        return None


def _sector_jp(sector_raw):
    if sector_raw is None:
        return None
    return SECTOR_JP_MAP.get(sector_raw, sector_raw)


def _industry_jp(industry_raw):
    if industry_raw is None:
        return None
    return INDUSTRY_JP_MAP.get(industry_raw, industry_raw)


def _reason_jp(reason_csv: str) -> str:
    if not reason_csv:
        return ""
    parts = [p.strip() for p in str(reason_csv).split(",") if p.strip()]
    out = []
    for p in parts:
        # EXCEPTION:〇〇 はプレフィクス一致で日本語化
        if p.startswith("EXCEPTION:"):
            out.append(REASON_JP_MAP["EXCEPTION:"] + f"（{p}）")
        else:
            out.append(REASON_JP_MAP.get(p, p))
    # 重複除去（順序保持）
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return "、".join(uniq)


def _has_japanese(s: str) -> bool:
    if not s:
        return False
    return re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", s) is not None


def _fetch_company_name_jp_from_yahoo(ticker: str):
    """
    finance.yahoo.co.jp から日本語社名を引くフォールバック。
    取れない場合は None。
    """
    try:
        url = f"https://finance.yahoo.co.jp/quote/{ticker}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        html = r.text
        # <title>住友商事(8053.T) ...</title> を想定して先頭を抜く
        m = re.search(r"<title>\s*([^<]+?)\s*\(", html, flags=re.IGNORECASE)
        if not m:
            return None
        name = m.group(1).strip()
        if name and _has_japanese(name):
            return name
        return None
    except Exception:
        return None


def get_company_name(t: yf.Ticker):
    """企業名（可能なら日本語名）"""
    try:
        info = t.get_info()

        # まずはyfinanceが返す候補を幅広く見る（日本語があればそれを採用）
        candidates = [
            info.get("longName"),
            info.get("shortName"),
            info.get("displayName"),
            info.get("name"),
        ]
        for c in candidates:
            if c:
                s = str(c).strip()
                if _has_japanese(s):
                    return s

        # 次に英語名（fallback）
        en = None
        for c in candidates:
            if c:
                en = str(c).strip()
                if en:
                    break

        # 最後に finance.yahoo.co.jp から日本語名を拾う（取れれば上書き）
        try:
            sym = info.get("symbol") or getattr(t, "ticker", None)
            if sym:
                jp = _fetch_company_name_jp_from_yahoo(str(sym).strip())
                if jp:
                    return jp
        except Exception:
            pass

        return en
    except Exception:
        return None


# ----------------------------
# 0) 実行ガード（週末・祝日・年末年始）
# ----------------------------
def should_skip_run(now_jst: datetime) -> tuple[bool, str]:
    d = now_jst.date()

    # 週末
    if d.weekday() >= 5:
        return True, "WEEKEND"

    # 年末年始（12/31〜1/3）
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (1, 2, 3)):
        return True, "YEAR_END_HOLIDAY"

    # 日本の祝日
    jp_holidays = holidays.country_holidays("JP")
    if d in jp_holidays:
        return True, f"JP_HOLIDAY:{jp_holidays.get(d)}"

    return False, ""


# ----------------------------
# 1) Secrets（1つ）読み込み：JSON直貼り版
# ----------------------------
@dataclass
class AppConfig:
    id: str       # spreadsheet id
    tab: str      # worksheet name
    sa: dict      # service account json


def load_config_from_env() -> AppConfig:
    raw = (os.environ.get("APP_CFG_JSON") or "").strip()
    if not raw:
        raise RuntimeError("Missing env APP_CFG_JSON (GitHub Secret)")

    try:
        cfg = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"APP_CFG_JSON is not valid JSON: {e}")

    for k in ("id", "tab", "sa"):
        if k not in cfg:
            raise RuntimeError(f"APP_CFG_JSON missing key: {k}")

    return AppConfig(
        id=str(cfg["id"]).strip(),
        tab=str(cfg["tab"]).strip(),
        sa=cfg["sa"],
    )


# ----------------------------
# 2) Google Sheets I/O
# ----------------------------
def open_worksheet(cfg: AppConfig):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(cfg.sa, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(cfg.id)
    ws = sh.worksheet(cfg.tab)
    return ws


def read_codes_from_col_a(ws) -> list[str]:
    col = ws.col_values(1)
    codes = []
    for v in col:
        s = str(v).strip()
        if not s:
            continue
        if s.lower() in ("code", "ticker", "銘柄コード", "銘柄", "銘柄code"):
            continue
        codes.append(s)

    # 重複除去（順序保持）
    seen = set()
    out = []
    for c in codes:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def write_table(ws, headers: list[str], rows: list[list]):
    values = [headers] + rows
    ws.update(values, "A1")


# ----------------------------
# 3) yfinance ユーティリティ
# ----------------------------
def normalize_key(s: str) -> str:
    return str(s).strip().lower()


def pick_bs_value(bs: pd.DataFrame, candidates: list[str]):
    if bs is None or bs.empty:
        return None
    latest_col = bs.columns[0]
    idx_map = {normalize_key(i): i for i in bs.index}

    for name in candidates:
        k = normalize_key(name)
        if k in idx_map:
            v = bs.loc[idx_map[k], latest_col]
            try:
                if pd.isna(v):
                    return None
                return float(v)
            except Exception:
                return None
    return None


def get_latest_balance_sheet(t: yf.Ticker) -> pd.DataFrame:
    bs = t.balance_sheet
    if bs is None or bs.empty:
        bs = t.quarterly_balance_sheet
    return bs


def get_cashflow_annual(t: yf.Ticker) -> pd.DataFrame:
    cf = t.cashflow
    if cf is None or cf.empty:
        cf = t.quarterly_cashflow
    return cf


def to_ticker(code: str) -> str:
    s = str(code).strip()
    if not s:
        return ""
    if "." in s:
        return s
    return f"{s}.T"


def get_market_cap(t: yf.Ticker):
    try:
        fi = getattr(t, "fast_info", {}) or {}
        mc = fi.get("market_cap")
        if mc is not None:
            return float(mc)
    except Exception:
        pass

    try:
        info = t.get_info()
        mc = info.get("marketCap")
        if mc is not None:
            return float(mc)
    except Exception:
        pass

    return None


def get_sector_industry(t: yf.Ticker):
    try:
        info = t.get_info()
    except Exception:
        return None, None
    sector = info.get("sector")
    industry = info.get("industry")
    return (str(sector).strip() if sector else None), (str(industry).strip() if industry else None)


def avg_daily_value_3m(t: yf.Ticker):
    try:
        hist = t.history(period="3mo", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        if "Close" not in hist.columns or "Volume" not in hist.columns:
            return None
        dv = (hist["Close"] * hist["Volume"]).dropna()
        if dv.empty:
            return None
        return float(dv.mean())
    except Exception:
        return None


def last_close_price(t: yf.Ticker):
    try:
        hist = t.history(period="5d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        close = hist["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


def dividend_yield_trailing(t: yf.Ticker):
    try:
        info = t.get_info()
        dy = info.get("dividendYield")
        if dy is not None:
            dyf = float(dy)
            if 0 <= dyf <= 1:
                return dyf
    except Exception:
        pass

    try:
        div = t.dividends
        if div is None or div.empty:
            return 0.0
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
        div_1y = div[div.index >= cutoff]
        total = float(div_1y.sum()) if not div_1y.empty else 0.0
        price = last_close_price(t)
        if price is None or price <= 0:
            return None
        return total / price
    except Exception:
        return None


def operating_cf_two_years(t: yf.Ticker):
    cf = get_cashflow_annual(t)
    if cf is None or cf.empty:
        return None, None

    candidates = [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "Net Cash Provided By Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ]

    idx_map = {normalize_key(i): i for i in cf.index}
    row = None
    for name in candidates:
        k = normalize_key(name)
        if k in idx_map:
            row = cf.loc[idx_map[k], :]
            break
    if row is None:
        return None, None

    cols = list(cf.columns)
    try:
        v1 = row[cols[0]] if len(cols) >= 1 else None
        v2 = row[cols[1]] if len(cols) >= 2 else None
        v1 = None if v1 is None or pd.isna(v1) else float(v1)
        v2 = None if v2 is None or pd.isna(v2) else float(v2)
        return v1, v2
    except Exception:
        return None, None


def split_flag_3y(t: yf.Ticker, now_jst: datetime) -> tuple[bool, str]:
    try:
        splits = t.splits
        if splits is None or len(splits) == 0:
            return False, ""
        cutoff = (now_jst.date() - timedelta(days=365 * 3))
        for ts, ratio in splits.items():
            try:
                if ts.date() >= cutoff:
                    return True, "SPLIT_3Y"
            except Exception:
                continue
        return False, ""
    except Exception:
        return False, ""


def shares_reduction_score_3y(t: yf.Ticker, now_jst: datetime) -> tuple[object, str]:
    try:
        start = (now_jst.date() - timedelta(days=365 * 4)).isoformat()
        df = t.get_shares_full(start=start)
        if df is None or df.empty:
            return None, "NO_SHARES"

        if "Shares" in df.columns:
            s = df["Shares"].copy()
        elif df.shape[1] == 1:
            s = df.iloc[:, 0].copy()
        else:
            return None, "NO_SHARES"

        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.dropna().sort_index()
        if s.empty:
            return None, "NO_SHARES"

        def nearest_value(target_date: date):
            tts = pd.Timestamp(target_date)
            after = s[s.index >= tts]
            if not after.empty:
                return float(after.iloc[0])
            before = s[s.index <= tts]
            if not before.empty:
                return float(before.iloc[-1])
            return None

        d0 = now_jst.date()
        d1 = d0 - timedelta(days=365)
        d2 = d0 - timedelta(days=365 * 2)
        d3 = d0 - timedelta(days=365 * 3)

        p0 = nearest_value(d0)
        p1 = nearest_value(d1)
        p2 = nearest_value(d2)
        p3 = nearest_value(d3)

        if any(v is None for v in (p0, p1, p2, p3)):
            return None, "NO_SHARES"

        score = 0
        if p0 < p1:
            score = 1
            if p1 < p2:
                score = 2
                if p2 < p3:
                    score = 3

        return score, ""
    except Exception:
        return None, "NO_SHARES"


# ----------------------------
# 4) 1銘柄解析（仕様どおり）
# ----------------------------
def analyze_one(code: str, now_jst: datetime) -> dict:
    out = {"code": code}
    reasons = []

    ticker = to_ticker(code)
    out["ticker"] = ticker

    t = yf.Ticker(ticker)

    # 企業名（追加：可能なら日本語）
    company = get_company_name(t)
    out["CompanyName"] = company

    # セクター/業種（判定は英語の生値で行い、表示は日本語化した値を別列に出す）
    sector_raw, industry_raw = get_sector_industry(t)
    out["SectorRaw"] = sector_raw
    out["IndustryRaw"] = industry_raw
    out["SectorJP"] = _sector_jp(sector_raw)
    out["IndustryJP"] = _industry_jp(industry_raw)

    sector_ok = (sector_raw is not None) and (sector_raw != "Financial Services")
    if sector_raw is None:
        reasons.append("SECTOR_UNKNOWN")
    elif sector_raw == "Financial Services":
        reasons.append("FINANCIAL_SECTOR_EXCLUDED")

    mcap = get_market_cap(t)
    adv3m = avg_daily_value_3m(t)
    out["MarketCap"] = mcap
    out["AvgDailyValue3M"] = adv3m

    size_ok = (mcap is not None) and (mcap >= 3_000_000_000)  # 30億
    if mcap is None:
        reasons.append("NO_MCAP")
    elif mcap < 3_000_000_000:
        reasons.append("SIZE<30億")

    liq_ok = (adv3m is not None) and (adv3m >= 10_000_000)  # 1000万
    if adv3m is None:
        reasons.append("NO_ADV3M")
    elif adv3m < 10_000_000:
        reasons.append("LIQ<1000万")

    bs = get_latest_balance_sheet(t)

    cash = pick_bs_value(bs, ["Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments", "Cash"])
    sti = pick_bs_value(bs, ["Short Term Investments", "Short-term Investments", "Short Term Investments And Other Short Term Investments"])
    receivables = pick_bs_value(bs, ["Net Receivables", "Accounts Receivable", "Receivables"])
    inventory = pick_bs_value(bs, ["Inventory"])
    other_ca = pick_bs_value(bs, ["Other Current Assets", "Other current assets"])
    total_ca = pick_bs_value(bs, ["Total Current Assets", "Current Assets"])

    if other_ca is None and total_ca is not None:
        parts = [cash or 0.0, sti or 0.0, receivables or 0.0, inventory or 0.0]
        other_ca = total_ca - sum(parts)

    tl_gross = pick_bs_value(bs, ["Total Liab", "Total Liabilities", "Total liabilities"])
    tl_net_mi = pick_bs_value(bs, ["Total Liabilities Net Minority Interest", "Total liabilities net minority interest"])

    minority = pick_bs_value(bs, ["Minority Interest", "Non Controlling Interests", "Non-controlling interests", "Minority Interests"])
    preferred = pick_bs_value(bs, ["Preferred Stock", "Preferred Stock And Other Adjustments", "Preferred Equity"]) or 0.0

    total_debt = pick_bs_value(bs, ["Total Debt"])
    if total_debt is None:
        lt_debt = pick_bs_value(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        st_debt = pick_bs_value(bs, ["Short Long Term Debt", "Short Term Debt", "Current Debt"])
        if lt_debt is not None or st_debt is not None:
            total_debt = (lt_debt or 0.0) + (st_debt or 0.0)

    lease_liab = pick_bs_value(bs, ["Lease Liabilities", "Lease Liabilities Non Current", "Lease Liabilities Current"])

    # 二重控除回避ルール
    if tl_gross is not None:
        used_tl = tl_gross
        minority_used = minority
    elif tl_net_mi is not None:
        used_tl = tl_net_mi
        minority_used = 0.0
    else:
        used_tl = None
        minority_used = minority

    out["Cash"] = cash
    out["ShortTermInvestments"] = sti
    out["Receivables"] = receivables
    out["Inventory"] = inventory
    out["OtherCurrentAssets"] = other_ca
    out["TotalCurrentAssets"] = total_ca

    out["TotalLiabilities"] = used_tl
    out["TotalDebt"] = total_debt
    out["LeaseLiabilities"] = lease_liab
    out["PreferredEquity"] = preferred
    out["MinorityInterest"] = minority_used

    ocf1, ocf2 = operating_cf_two_years(t)
    out["OCF_FY1"] = ocf1
    out["OCF_FY2"] = ocf2
    ocf_2y_ok = (ocf1 is not None) and (ocf2 is not None) and (ocf1 > 0) and (ocf2 > 0)
    if ocf1 is None or ocf2 is None:
        reasons.append("NO_OCF2Y")

    dy = dividend_yield_trailing(t)
    out["DividendYield"] = dy
    dividend_ok = (dy is not None) and (dy >= 0.02)
    if dy is None:
        reasons.append("NO_DIV_YIELD")

    net_cash = None
    if cash is not None and total_debt is not None:
        net_cash = (cash + (sti or 0.0)) - total_debt
    out["NetCash"] = net_cash
    netcash_ok = (net_cash is not None) and (net_cash > 0)
    out["NetCash_OK"] = bool(netcash_ok) if net_cash is not None else None
    if cash is None:
        reasons.append("NO_CASH")
    if total_debt is None:
        reasons.append("NO_DEBT")

    adj_ncav = None
    if cash is None or receivables is None or inventory is None or other_ca is None or used_tl is None or minority_used is None:
        reasons.append("NO_ADJNCAV_INPUT")
    else:
        adj_ncav = (
            (cash + (sti or 0.0)) * 1.00
            + receivables * 0.75
            + inventory * 0.50
            + other_ca * 0.25
            - used_tl
            - preferred
            - (minority_used or 0.0)
        )
    out["AdjustedNCAV"] = adj_ncav

    adj_ratio = None
    if adj_ncav is not None and mcap not in (None, 0):
        adj_ratio = adj_ncav / mcap
    out["AdjNCAV_div_MCap"] = adj_ratio

    adj_netnet_ok = (adj_ncav is not None) and (mcap is not None) and (mcap < adj_ncav)
    graham_2_3_ok = (adj_ncav is not None) and (mcap is not None) and (mcap < adj_ncav * (2.0 / 3.0))

    split3y, split_reason = split_flag_3y(t, now_jst)
    out["Split_3Y_Flag"] = bool(split3y)
    if split_reason:
        reasons.append(split_reason)

    score, score_reason = shares_reduction_score_3y(t, now_jst)
    if split3y:
        out["Shares_Reduction_Score"] = 0
    else:
        out["Shares_Reduction_Score"] = score
        if score_reason:
            reasons.append(score_reason)

    out["Size_OK"] = bool(size_ok) if mcap is not None else None
    out["Liquidity_OK"] = bool(liq_ok) if adv3m is not None else None
    out["Sector_OK"] = bool(sector_ok) if sector_raw is not None else None

    out["AdjNetNet_OK"] = bool(adj_netnet_ok) if adj_ncav is not None and mcap is not None else None
    out["Graham_2_3_OK"] = bool(graham_2_3_ok) if adj_ncav is not None and mcap is not None else None
    out["OCF_2Y_OK"] = bool(ocf_2y_ok) if (ocf1 is not None and ocf2 is not None) else None
    out["Dividend_OK"] = bool(dividend_ok) if dy is not None else None

    # ハード制約
    hard_ok = True
    if not size_ok or not liq_ok or not sector_ok:
        hard_ok = False
    if mcap is None or adv3m is None or sector_raw is None or adj_ncav is None or dy is None or net_cash is None or ocf1 is None or ocf2 is None:
        hard_ok = False

    final = "✘"
    if not hard_ok:
        final = "✘"
    else:
        if graham_2_3_ok and ocf_2y_ok and dividend_ok and netcash_ok:
            final = "◎"
        elif adj_netnet_ok and ocf_2y_ok and dividend_ok and netcash_ok:
            final = "◯"
        else:
            if adj_netnet_ok:
                unmet = 0
                unmet += 0 if ocf_2y_ok else 1
                unmet += 0 if dividend_ok else 1
                unmet += 0 if netcash_ok else 1
                final = "△" if unmet == 1 else "✘"
            else:
                if adj_ratio is not None and 0.90 <= adj_ratio < 1.00 and ocf_2y_ok and dividend_ok and netcash_ok:
                    final = "△"
                else:
                    final = "✘"

    # 加点で△→◯（◎には影響なし / split3yなら無効）
    if final == "△" and (not split3y):
        score_val = out.get("Shares_Reduction_Score")
        if isinstance(score_val, (int, float)) and score_val >= 2 and netcash_ok:
            if adj_ratio is not None and 0.90 <= adj_ratio < 1.00 and ocf_2y_ok and dividend_ok and netcash_ok:
                final = "◯"
                reasons.append("UPGRADED_BY_SHARES_SCORE")
            elif adj_netnet_ok:
                unmet = 0
                unmet += 0 if ocf_2y_ok else 1
                unmet += 0 if dividend_ok else 1
                unmet += 0 if netcash_ok else 1
                if unmet == 1 and netcash_ok:
                    if (not ocf_2y_ok) or (not dividend_ok):
                        final = "◯"
                        reasons.append("UPGRADED_BY_SHARES_SCORE")

    out["Final"] = final
    out["Reason"] = ",".join(sorted(set(reasons))) if reasons else ""

    return out


# ----------------------------
# 5) 実行本体
# ----------------------------
def main():
    now_jst = datetime.now(JST)
    skip, reason = should_skip_run(now_jst)
    if skip:
        print(f"[SKIP] {now_jst.isoformat()} reason={reason}")
        return

    cfg = load_config_from_env()
    ws = open_worksheet(cfg)

    codes = read_codes_from_col_a(ws)
    if not codes:
        print("No codes found in column A.")
        return

    results = []
    for c in codes:
        try:
            results.append(analyze_one(c, now_jst))
        except Exception as e:
            results.append({
                "code": c,
                "ticker": to_ticker(c),
                "Final": "✘",
                "Reason": f"EXCEPTION:{type(e).__name__}",
            })

    df = pd.DataFrame(results)

    # 出力用：日本語ヘッダー＋表示変換（億円・Reason日本語化）
    # ※計算はすべて円で行い、スプシ表示だけ億円にする
    jp_headers = [
        "銘柄コード",
        "ティッカー",
        "企業名",
        "セクター",
        "業種",
        "時価総額(億円)",
        "平均日次売買代金3か月(億円)",
        "時価総額OK",
        "流動性OK",
        "セクターOK",
        "現金(億円)",
        "短期投資(億円)",
        "売掛金等(億円)",
        "棚卸資産(億円)",
        "その他流動資産(億円)",
        "流動資産合計(億円)",
        "負債合計(億円)",
        "有利子負債(億円)",
        "リース負債(億円)",
        "優先株等(億円)",
        "少数株主持分(億円)",
        "ネットキャッシュ(億円)",
        "ネットキャッシュOK",
        "Adjusted NCAV(億円)",
        "AdjNCAV/時価総額",
        "等倍割れOK",
        "2/3割れOK",
        "営業CF FY1(億円)",
        "営業CF FY2(億円)",
        "営業CF2年OK",
        "配当利回り",
        "配当2%OK",
        "過去3年分割/併合",
        "株数減少スコア(0-3)",
        "最終判定",
        "理由",
    ]

    # 欠損列を補完（既存ロジックを壊さない）
    needed = [
        "code",
        "ticker",
        "CompanyName",
        "SectorJP",
        "IndustryJP",
        "MarketCap",
        "AvgDailyValue3M",
        "Size_OK",
        "Liquidity_OK",
        "Sector_OK",
        "Cash",
        "ShortTermInvestments",
        "Receivables",
        "Inventory",
        "OtherCurrentAssets",
        "TotalCurrentAssets",
        "TotalLiabilities",
        "TotalDebt",
        "LeaseLiabilities",
        "PreferredEquity",
        "MinorityInterest",
        "NetCash",
        "NetCash_OK",
        "AdjustedNCAV",
        "AdjNCAV_div_MCap",
        "AdjNetNet_OK",
        "Graham_2_3_OK",
        "OCF_FY1",
        "OCF_FY2",
        "OCF_2Y_OK",
        "DividendYield",
        "Dividend_OK",
        "Split_3Y_Flag",
        "Shares_Reduction_Score",
        "Final",
        "Reason",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    d = df.copy()

    # 億円変換（表示用）
    oku_cols = [
        "MarketCap",
        "AvgDailyValue3M",
        "Cash",
        "ShortTermInvestments",
        "Receivables",
        "Inventory",
        "OtherCurrentAssets",
        "TotalCurrentAssets",
        "TotalLiabilities",
        "TotalDebt",
        "LeaseLiabilities",
        "PreferredEquity",
        "MinorityInterest",
        "NetCash",
        "AdjustedNCAV",
        "OCF_FY1",
        "OCF_FY2",
    ]
    for c in oku_cols:
        # ここが今回の修正点：文字列混入でも取りこぼさないよう、先に数値化してから億円へ
        d[c] = pd.to_numeric(d[c], errors="coerce").apply(_to_oku)

    # Reason 日本語化
    d["ReasonJP"] = d["Reason"].apply(_reason_jp)

    # Sharesスコアは None のとき空欄に
    shares_score_disp = d["Shares_Reduction_Score"].where(pd.notna(d["Shares_Reduction_Score"]), "")

    # 出力順（日本語ヘッダー順に揃える）
    out_df = pd.DataFrame({
        "銘柄コード": d["code"],
        "ティッカー": d["ticker"],
        "企業名": d["CompanyName"],
        "セクター": d["SectorJP"],
        "業種": d["IndustryJP"],
        "時価総額(億円)": d["MarketCap"],
        "平均日次売買代金3か月(億円)": d["AvgDailyValue3M"],
        "時価総額OK": d["Size_OK"],
        "流動性OK": d["Liquidity_OK"],
        "セクターOK": d["Sector_OK"],
        "現金(億円)": d["Cash"],
        "短期投資(億円)": d["ShortTermInvestments"],
        "売掛金等(億円)": d["Receivables"],
        "棚卸資産(億円)": d["Inventory"],
        "その他流動資産(億円)": d["OtherCurrentAssets"],
        "流動資産合計(億円)": d["TotalCurrentAssets"],
        "負債合計(億円)": d["TotalLiabilities"],
        "有利子負債(億円)": d["TotalDebt"],
        "リース負債(億円)": d["LeaseLiabilities"],
        "優先株等(億円)": d["PreferredEquity"],
        "少数株主持分(億円)": d["MinorityInterest"],
        "ネットキャッシュ(億円)": d["NetCash"],
        "ネットキャッシュOK": d["NetCash_OK"],
        "Adjusted NCAV(億円)": d["AdjustedNCAV"],
        "AdjNCAV/時価総額": d["AdjNCAV_div_MCap"],
        "等倍割れOK": d["AdjNetNet_OK"],
        "2/3割れOK": d["Graham_2_3_OK"],
        "営業CF FY1(億円)": d["OCF_FY1"],
        "営業CF FY2(億円)": d["OCF_FY2"],
        "営業CF2年OK": d["OCF_2Y_OK"],
        "配当利回り": d["DividendYield"],
        "配当2%OK": d["Dividend_OK"],
        "過去3年分割/併合": d["Split_3Y_Flag"],
        "株数減少スコア(0-3)": shares_score_disp,
        "最終判定": d["Final"],
        "理由": d["ReasonJP"],
    }).replace({np.nan: None})

    ws.update([jp_headers] + out_df.values.tolist(), "A1")
    print(f"Updated tab='{cfg.tab}' rows={len(out_df)} at {now_jst.isoformat()}")


if __name__ == "__main__":
    main()
