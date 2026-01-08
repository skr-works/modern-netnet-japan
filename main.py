import json
import os
import re
import time
import random
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

import gspread
from google.oauth2.service_account import Credentials

import holidays
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JST = ZoneInfo("Asia/Tokyo")

# --- 日本語化・表示変換（出力用） -----------------

# 東証33業種リスト（スクレイピング時のマッチング用）
TSE_SECTORS = [
    "水産・農林業", "鉱業", "建設業", "食料品", "繊維製品", "パルプ・紙", "化学",
    "医薬品", "石油・石炭製品", "ゴム製品", "ガラス・土石製品", "鉄鋼", "非鉄金属",
    "金属製品", "機械", "電気機器", "輸送用機器", "精密機器", "その他製品",
    "電気・ガス業", "陸運業", "海運業", "空運業", "倉庫・運輸関連業", "情報・通信業",
    "卸売業", "小売業", "銀行業", "証券、商品先物取引業", "保険業",
    "その他金融業", "不動産業", "サービス業"
]

REASON_JP_MAP = {
    "SECTOR_UNKNOWN": "業種取得不可",
    "FINANCIAL_SECTOR_EXCLUDED": "金融業種除外",
    "NO_MCAP": "時価総額取得不可",
    "SIZE<30億": "時価総額30億未満",
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
    "NO_JP_NAME": "企業名（日本語）取得不可",
}


def _to_oku(v):
    """円→億円（表示用）。None/空はそのまま。"""
    if v is None:
        return None
    try:
        return float(v) / 1e8
    except Exception:
        return None


def _reason_jp(reason_csv: str) -> str:
    if not reason_csv:
        return ""
    parts = [p.strip() for p in str(reason_csv).split(",") if p.strip()]
    out = []
    for p in parts:
        if p.startswith("EXCEPTION:"):
            out.append(REASON_JP_MAP["EXCEPTION:"] + f"（{p}）")
        else:
            out.append(REASON_JP_MAP.get(p, p))
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return "、".join(uniq)


# --- HTTPセッション ---
_HTTP_SESSION = requests.Session()
_retry = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
)
_HTTP_SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
_HTTP_SESSION.mount("http://", HTTPAdapter(max_retries=_retry))


def fetch_yahoo_jp_data(ticker_code: str, existing_name: str | None) -> tuple[str | None, str | None]:
    """
    Yahoo!ファイナンス(日本)から企業名と業種(セクター)を取得する。
    """
    try:
        s = str(ticker_code).strip()
        if not s:
            return None, None

        if "." in s:
            base = s.split(".", 1)[0]
        else:
            base = s

        code_t = f"{base}.T"
        url = f"https://finance.yahoo.co.jp/quote/{code_t}"

        time.sleep(random.uniform(0.05, 0.1))

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = _HTTP_SESSION.get(url, headers=headers, timeout=5)
        res.encoding = res.apparent_encoding
        html = res.text

        # 1. 企業名
        name = existing_name
        if not name:
            m_title = re.search(r"<title>\s*(.*?)\s*(?:【|\|)", html, flags=re.IGNORECASE | re.DOTALL)
            if m_title:
                name = re.sub(r"\s+", " ", m_title.group(1)).strip()

        # 2. 業種 (HTMLから東証33業種を探す)
        sector = None
        for candidate in TSE_SECTORS:
            # リンクテキスト等になっているケースを想定
            if f">{candidate}<" in html or f"\"{candidate}\"" in html:
                sector = candidate
                break
        
        if not sector:
            for candidate in TSE_SECTORS:
                if candidate in html:
                    sector = candidate
                    break

        return name, sector

    except Exception:
        return existing_name, None


def get_market_cap_with_retry(t: yf.Ticker, max_retries=3) -> float | None:
    """
    時価総額をリトライ付きで取得する関数。
    アクセス拒否(401/429)対策として、失敗時にWaitを入れて再試行する。
    """
    for i in range(max_retries):
        try:
            mcap = None
            if hasattr(t, "fast_info"):
                # fast_infoはdictではないため[]アクセスで取得
                try:
                    mcap = t.fast_info["marketCap"]
                except KeyError:
                    # バージョン揺れ対応
                    if "market_cap" in t.fast_info:
                        mcap = t.fast_info["market_cap"]
            
            # 取得できていれば数値を返す
            if mcap is not None:
                return float(mcap)
        
        except Exception:
            # エラー時はスルーしてリトライへ
            pass
        
        # 失敗した場合、少し待機時間を増やしてリトライ (1秒, 2秒, 3秒...)
        time.sleep(1.0 + i)
        
    return None


# ----------------------------
# 0) 実行ガード
# ----------------------------
def should_skip_run(now_jst: datetime) -> tuple[bool, str]:
    d = now_jst.date()
    if d.weekday() >= 5:
        return True, "WEEKEND"
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (1, 2, 3)):
        return True, "YEAR_END_HOLIDAY"
    jp_holidays = holidays.country_holidays("JP")
    if d in jp_holidays:
        return True, f"JP_HOLIDAY:{jp_holidays.get(d)}"
    return False, ""


# ----------------------------
# 1) Secrets読み込み
# ----------------------------
@dataclass
class AppConfig:
    id: str
    tab: str
    sa: dict


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


def read_sheet_data(ws) -> tuple[list[str], dict[str, str]]:
    rows = ws.get_all_values()
    if not rows:
        return [], {}
    data_rows = rows[1:]
    codes = []
    name_map = {}
    seen = set()
    for row in data_rows:
        if len(row) < 1:
            continue
        c = str(row[0]).strip()
        if not c:
            continue
        if c.lower() in ("code", "ticker", "銘柄コード", "銘柄", "銘柄code"):
            continue
        n = str(row[1]).strip() if len(row) > 1 else ""
        if c not in seen:
            codes.append(c)
            seen.add(c)
            name_map[c] = n
    return codes, name_map


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
    try:
        bs = t.balance_sheet
        if bs is None or bs.empty:
            bs = t.quarterly_balance_sheet
        return bs
    except Exception:
        return None


def get_cashflow_annual(t: yf.Ticker) -> pd.DataFrame:
    try:
        cf = t.cashflow
        if cf is None or cf.empty:
            cf = t.quarterly_cashflow
        return cf
    except Exception:
        return None


def to_ticker(code: str) -> str:
    s = str(code).strip()
    if not s:
        return ""
    if "." in s:
        return s
    return f"{s}.T"


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
# 4) 1銘柄解析（超高速化・列ズレ修正・順序修正・リトライ版）
# ----------------------------
def analyze_one(code: str, now_jst: datetime, name_cache: dict) -> dict:
    time.sleep(random.uniform(0.5, 1.5))
    
    out = {"code": code}
    reasons = []

    ticker = to_ticker(code)
    out["ticker"] = ticker

    t = yf.Ticker(ticker)

    # 1. 企業名と業種(JP)の取得
    existing_name = name_cache.get(code)
    company_jp, sector_jp = fetch_yahoo_jp_data(ticker, existing_name)
    
    if company_jp is None:
        reasons.append("NO_JP_NAME")
    out["CompanyName"] = company_jp
    
    out["SectorJP"] = sector_jp 

    sector_ok = (sector_jp is not None)
    if sector_jp and any(x in sector_jp for x in ["銀行", "証券", "保険", "金融"]):
        sector_ok = False
        reasons.append("FINANCIAL_SECTOR_EXCLUDED")
    if sector_jp is None:
        reasons.append("SECTOR_UNKNOWN")

    # 2. 時価総額の門前払い（リトライ機能付き）
    # 大量アクセスでNoneが返りやすいため、リトライで粘る
    mcap = get_market_cap_with_retry(t, max_retries=3)
    
    if mcap is not None and mcap < 3_000_000_000:
        out["MarketCap"] = mcap
        out["Size_OK"] = False
        out["Final"] = "✘"
        out["Reason"] = "SIZE<30億"
        return out
    
    out["MarketCap"] = mcap
    size_ok = (mcap is not None) and (mcap >= 3_000_000_000)
    if mcap is None:
        reasons.append("NO_MCAP")
    elif not size_ok:
        reasons.append("SIZE<30億")

    # 3. 財務データの取得
    if not size_ok:
        out["Size_OK"] = False
        out["Final"] = "✘"
        out["Reason"] = ",".join(reasons)
        return out

    bs = get_latest_balance_sheet(t)

    cash = pick_bs_value(bs, ["Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments", "Cash"])
    
    sti = 0.0 
    
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

    try:
        last_price = None
        if hasattr(t, "fast_info"):
            try:
                last_price = t.fast_info["lastPrice"]
            except KeyError:
                if "last_price" in t.fast_info:
                    last_price = t.fast_info["last_price"]

        if last_price and last_price > 0:
            div_series = t.dividends
            if div_series is not None and not div_series.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                div_1y = div_series[div_series.index >= cutoff]
                div_total = float(div_1y.sum())
                dy = div_total / last_price
            else:
                dy = 0.0
        else:
            dy = None
    except Exception:
        dy = None

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

    out["Size_OK"] = bool(size_ok) if mcap is not None else None
    out["Sector_OK"] = bool(sector_ok) if sector_jp is not None else None
    out["AdjNetNet_OK"] = bool(adj_netnet_ok) if adj_ncav is not None and mcap is not None else None
    out["Graham_2_3_OK"] = bool(graham_2_3_ok) if adj_ncav is not None and mcap is not None else None
    out["OCF_2Y_OK"] = bool(ocf_2y_ok) if (ocf1 is not None and ocf2 is not None) else None
    out["Dividend_OK"] = bool(dividend_ok) if dy is not None else None

    # 重い取得はデフォルトOFF
    out["Split_3Y_Flag"] = False
    out["Shares_Reduction_Score"] = None

    hard_ok = True
    if not size_ok or not sector_ok:
        hard_ok = False
    if mcap is None or sector_jp is None or adj_ncav is None or dy is None or net_cash is None or ocf1 is None or ocf2 is None:
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

    if final == "△" and hard_ok:
        split3y, split_reason = split_flag_3y(t, now_jst)
        out["Split_3Y_Flag"] = bool(split3y)
        if split_reason:
            reasons.append(split_reason)

        if split3y:
            out["Shares_Reduction_Score"] = 0
        else:
            score, score_reason = shares_reduction_score_3y(t, now_jst)
            out["Shares_Reduction_Score"] = score
            if score_reason:
                reasons.append(score_reason)

        if (not split3y):
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

    codes, name_cache = read_sheet_data(ws)
    if not codes:
        print("No codes found in column A.")
        return
    
    print(f"Start processing {len(codes)} stocks at {now_jst.isoformat()}")

    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_code = {
            executor.submit(analyze_one, c, now_jst, name_cache): c 
            for c in codes
        }
        
        for i, future in enumerate(as_completed(future_to_code)):
            code = future_to_code[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                results.append({
                    "code": code,
                    "ticker": to_ticker(code),
                    "Final": "✘",
                    "Reason": f"EXCEPTION:{type(e).__name__}",
                })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(codes)}...")

    code_order = {c: i for i, c in enumerate(codes)}
    results.sort(key=lambda x: code_order.get(x["code"], 99999))

    df = pd.DataFrame(results)

    jp_headers = [
        "銘柄コード",
        "企業名",
        "業種",
        "最終判定",
        "時価総額(億円)",
        "時価総額OK",
        "セクターOK",
        "現金(億円)",
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
        "理由",
    ]

    needed = [
        "code",
        "CompanyName",
        "SectorJP",
        "Final",
        "MarketCap",
        "Size_OK",
        "Sector_OK",
        "Cash",
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
        "Reason",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    d = df.copy()

    oku_cols = [
        "MarketCap",
        "Cash",
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
        d[c] = d[c].apply(_to_oku)

    d["ReasonJP"] = d["Reason"].apply(_reason_jp)

    out_df = pd.DataFrame({
        "銘柄コード": d["code"],
        "企業名": d["CompanyName"],
        "業種": d["SectorJP"],
        "最終判定": d["Final"],
        "時価総額(億円)": d["MarketCap"],
        "時価総額OK": d["Size_OK"],
        "セクターOK": d["Sector_OK"],
        "現金(億円)": d["Cash"],
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
        "株数減少スコア(0-3)": d["Shares_Reduction_Score"],
        "理由": d["ReasonJP"],
    }).replace({np.nan: None})

    ws.update([jp_headers] + out_df.values.tolist(), "A1")
    print(f"Updated tab='{cfg.tab}' rows={len(out_df)} at {now_jst.isoformat()}")


if __name__ == "__main__":
    main()
