"""Gold-relevant world news from Forex Factory calendar + Yahoo Finance."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# Countries / hubs that move gold: demand, supply, USD, policy, refining, bullion, geopolitics.
GOLD_HUBS: dict[str, dict[str, Any]] = {
    "US": {
        "name": "United States",
        "lat": 38.9072,
        "lng": -77.0369,
        "role": "Fed policy, USD, CPI & jobs data",
        "currencies": ["USD"],
        "keywords": ["fed", "fomc", "cpi", "payroll", "nfp", "dollar", "treasury", "powell", "usd"],
    },
    "CN": {
        "name": "China",
        "lat": 39.9042,
        "lng": 116.4074,
        "role": "World’s largest gold buyer",
        "currencies": ["CNY", "CNH"],
        "keywords": ["china", "pboc", "yuan", "shanghai gold", "chinese"],
    },
    "IN": {
        "name": "India",
        "lat": 28.6139,
        "lng": 77.2090,
        "role": "Jewellery & festival demand",
        "currencies": ["INR"],
        "keywords": ["india", "rupee", "jewellery", "jewelry", "rbi", "diwali"],
    },
    "AU": {
        "name": "Australia",
        "lat": -35.2809,
        "lng": 149.1300,
        "role": "Major gold miner",
        "currencies": ["AUD"],
        "keywords": ["australia", "rba", "aud", "perth mint"],
    },
    "ZA": {
        "name": "South Africa",
        "lat": -25.7479,
        "lng": 28.2293,
        "role": "Historic mining powerhouse",
        "currencies": ["ZAR"],
        "keywords": ["south africa", "rand", "johannesburg", "mining"],
    },
    "RU": {
        "name": "Russia",
        "lat": 55.7558,
        "lng": 37.6173,
        "role": "Reserves & sanctions flows",
        "currencies": ["RUB"],
        "keywords": ["russia", "ruble", "russian", "central bank gold", "moscow", "kremlin"],
    },
    "CH": {
        "name": "Switzerland",
        "lat": 46.9480,
        "lng": 7.4474,
        "role": "Refining & vault hub",
        "currencies": ["CHF"],
        "keywords": ["switzerland", "swiss", "refinery", "zurich", "snb"],
    },
    "GB": {
        "name": "United Kingdom",
        "lat": 51.5074,
        "lng": -0.1278,
        "role": "LBMA London bullion market",
        "currencies": ["GBP"],
        "keywords": ["britain", "uk", "boe", "lbma", "london gold", "sterling"],
    },
    "EU": {
        "name": "Eurozone",
        "lat": 50.8503,
        "lng": 4.3517,
        "role": "ECB rates & euro risk",
        "currencies": ["EUR"],
        "keywords": ["ecb", "euro", "eurozone", "lagarde", "germany", "france"],
    },
    "JP": {
        "name": "Japan",
        "lat": 35.6762,
        "lng": 139.6503,
        "role": "Yen & BoJ policy spillover",
        "currencies": ["JPY"],
        "keywords": ["japan", "boj", "yen", "tokyo"],
    },
    "CA": {
        "name": "Canada",
        "lat": 45.4215,
        "lng": -75.6972,
        "role": "Mining & CAD risk appetite",
        "currencies": ["CAD"],
        "keywords": ["canada", "boc", "cad", "loonie"],
    },
    "TR": {
        "name": "Turkey",
        "lat": 39.9334,
        "lng": 32.8597,
        "role": "Retail gold demand hotspot",
        "currencies": ["TRY"],
        "keywords": ["turkey", "lira", "istanbul", "turkish"],
    },
    "AE": {
        "name": "United Arab Emirates",
        "lat": 25.2048,
        "lng": 55.2708,
        "role": "Dubai bullion trading hub",
        "currencies": ["AED"],
        "keywords": ["dubai", "uae", "emirates", "dmcc"],
    },
    "BR": {
        "name": "Brazil",
        "lat": -15.8267,
        "lng": -47.9218,
        "role": "EM risk & commodity flows",
        "currencies": ["BRL"],
        "keywords": ["brazil", "real", "bcb", "brazilian"],
    },
    "NZ": {
        "name": "New Zealand",
        "lat": -41.2865,
        "lng": 174.7762,
        "role": "RBNZ & risk-sensitive FX",
        "currencies": ["NZD"],
        "keywords": ["new zealand", "rbnz", "kiwi"],
    },
    # —— Expanded gold influencers ——
    "IR": {
        "name": "Iran",
        "lat": 35.6892,
        "lng": 51.3890,
        "role": "Geopolitics, sanctions & oil-risk premium",
        "currencies": ["IRR"],
        "keywords": [
            "iran", "tehran", "iranian", "hormuz", "irgc", "nuclear deal",
            "jcpoa", "persian gulf",
        ],
    },
    "SA": {
        "name": "Saudi Arabia",
        "lat": 24.7136,
        "lng": 46.6753,
        "role": "OPEC oil, CB buying & Gulf demand",
        "currencies": ["SAR"],
        "keywords": ["saudi", "riyadh", "aramco", "opec", "saudi arabia"],
    },
    "GH": {
        "name": "Ghana",
        "lat": 5.6037,
        "lng": -0.1870,
        "role": "Africa’s top gold producer",
        "currencies": ["GHS"],
        "keywords": ["ghana", "accra", "ghanaian", "west africa gold"],
    },
    "PE": {
        "name": "Peru",
        "lat": -12.0464,
        "lng": -77.0428,
        "role": "Major LatAm mine supply",
        "currencies": ["PEN"],
        "keywords": ["peru", "peruvian", "lima", "andes mining"],
    },
    "MX": {
        "name": "Mexico",
        "lat": 19.4326,
        "lng": -99.1332,
        "role": "Silver/gold mining & Banxico",
        "currencies": ["MXN"],
        "keywords": ["mexico", "mexican", "banxico", "peso"],
    },
    "KZ": {
        "name": "Kazakhstan",
        "lat": 51.1694,
        "lng": 71.4491,
        "role": "Large producer & reserve buyer",
        "currencies": ["KZT"],
        "keywords": ["kazakhstan", "kazakh", "astana", "national bank of kazakhstan"],
    },
    "UZ": {
        "name": "Uzbekistan",
        "lat": 41.2995,
        "lng": 69.2401,
        "role": "Central Asian mine & CB gold",
        "currencies": ["UZS"],
        "keywords": ["uzbekistan", "uzbek", "tashkent"],
    },
    "SG": {
        "name": "Singapore",
        "lat": 1.3521,
        "lng": 103.8198,
        "role": "Asian bullion & vault hub",
        "currencies": ["SGD"],
        "keywords": ["singapore", "mas ", "raffles", "sgx"],
    },
    "HK": {
        "name": "Hong Kong",
        "lat": 22.3193,
        "lng": 114.1694,
        "role": "Physical premiums & China gateway",
        "currencies": ["HKD"],
        "keywords": ["hong kong", "hkma", "hongkong"],
    },
    "PL": {
        "name": "Poland",
        "lat": 52.2297,
        "lng": 21.0122,
        "role": "Aggressive central-bank gold buyer",
        "currencies": ["PLN"],
        "keywords": ["poland", "polish", "nbp", "warsaw", "zloty"],
    },
    "KR": {
        "name": "South Korea",
        "lat": 37.5665,
        "lng": 126.9780,
        "role": "East Asian investment demand",
        "currencies": ["KRW"],
        "keywords": ["south korea", "korea", "korean", "bok", "won", "seoul"],
    },
    "TH": {
        "name": "Thailand",
        "lat": 13.7563,
        "lng": 100.5018,
        "role": "SE Asia jewellery demand",
        "currencies": ["THB"],
        "keywords": ["thailand", "thai", "bangkok", "baht"],
    },
    "VN": {
        "name": "Vietnam",
        "lat": 21.0285,
        "lng": 105.8542,
        "role": "Retail gold & FX controls",
        "currencies": ["VND"],
        "keywords": ["vietnam", "vietnamese", "hanoi", "dong"],
    },
    "EG": {
        "name": "Egypt",
        "lat": 30.0444,
        "lng": 31.2357,
        "role": "MENA retail gold & pound stress",
        "currencies": ["EGP"],
        "keywords": ["egypt", "egyptian", "cairo", "pound"],
    },
}

_CURRENCY_TO_HUB = {
    code: hub_id
    for hub_id, hub in GOLD_HUBS.items()
    for code in hub["currencies"]
}

_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_CACHE_TTL = 300  # 5 minutes

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _get(url: str, timeout: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _short(text: str, limit: int = 88) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return (cut or text[: limit - 1]) + "…"


def _match_hub(text: str) -> Optional[str]:
    """Map headline text → hub; prefer the longest keyword hit (avoids weak overlaps)."""
    low = (text or "").lower()
    best_id: Optional[str] = None
    best_len = 0
    for hub_id, hub in GOLD_HUBS.items():
        for kw in hub["keywords"]:
            if kw and kw in low and len(kw) > best_len:
                best_id = hub_id
                best_len = len(kw)
    if best_id:
        return best_id
    if "gold" in low or "xau" in low or "bullion" in low:
        return "US"  # default gold macro → USD complex
    return None


def _fetch_forex_factory() -> list[dict[str, Any]]:
    """This-week economic calendar (Faireconomy mirror of Forex Factory)."""
    urls = (
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://www.forexfactory.com/calendar?week=this",  # HTML fallback unused; kept for note
    )
    raw = None
    try:
        raw = json.loads(_get(urls[0]).decode("utf-8", errors="replace"))
        if isinstance(raw, list) and raw:
            _write_ff_disk_cache(raw)
    except Exception as exc:
        logger.warning("Forex Factory calendar fetch failed: %s", exc)
        raw = _read_ff_disk_cache()
        if raw:
            logger.info("Using cached Forex Factory calendar (%s rows)", len(raw))

    if not isinstance(raw, list) or not raw:
        return []

    items: list[dict[str, Any]] = []
    for row in raw:
        country = str(row.get("country") or "").upper().strip()
        hub_id = _CURRENCY_TO_HUB.get(country)
        if not hub_id:
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        impact = str(row.get("impact") or "Low").title()
        # Keep Low for major hubs — they still matter for gold scoring
        if impact not in ("High", "Medium", "Low", "Holiday") and hub_id not in ("US", "CN", "EU", "GB"):
            continue
        forecast = row.get("forecast") or "—"
        previous = row.get("previous") or "—"
        date_s = str(row.get("date") or "")
        headline = _short(f"{title} · {impact}")
        detail = _short(f"Forecast {forecast} · Prior {previous}", 100)
        items.append(
            {
                "id": f"ff-{hub_id}-{title}-{date_s}",
                "source": "Forex Factory",
                "hub": hub_id,
                "headline": headline,
                "detail": detail,
                "impact": impact,
                "when": date_s,
                "url": "https://www.forexfactory.com/calendar",
            }
        )
    return items[:120]


def _fetch_yahoo_rss(symbol: str, label: str) -> list[dict[str, Any]]:
    url = f"https://finance.yahoo.com/rss/headline?s={urllib.parse.quote(symbol)}"
    try:
        xml = _get(url).decode("utf-8", errors="replace")
        root = ET.fromstring(xml)
    except Exception as exc:
        logger.warning("Yahoo RSS %s failed: %s", symbol, exc)
        return []

    items: list[dict[str, Any]] = []
    channel = root.find("channel")
    if channel is None:
        return []
    for node in channel.findall("item")[:25]:
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        pub = (node.findtext("pubDate") or "").strip()
        desc = re.sub(r"<[^>]+>", "", node.findtext("description") or "").strip()
        blob = f"{title} {desc}"
        hub_id = _match_hub(blob)
        if hub_id is None and symbol in ("GC=F", "XAUUSD=X", "GLD"):
            hub_id = "US"
        if hub_id is None:
            continue
        items.append(
            {
                "id": f"yh-{symbol}-{hash(title) & 0xFFFF}",
                "source": f"Yahoo · {label}",
                "hub": hub_id,
                "headline": _short(title),
                "detail": _short(desc or title, 100),
                "impact": "News",
                "when": pub,
                "url": link or "https://finance.yahoo.com/quote/GC%3DF",
            }
        )
    return items


def _fetch_yahoo_search(query: str) -> list[dict[str, Any]]:
    q = urllib.parse.quote(query)
    url = (
        f"https://query2.finance.yahoo.com/v1/finance/search?"
        f"q={q}&lang=en-US&region=US&quotesCount=0&newsCount=20&listsCount=0"
    )
    try:
        payload = json.loads(_get(url).decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("Yahoo search failed (%s): %s", query, exc)
        return []

    items: list[dict[str, Any]] = []
    for row in payload.get("news") or []:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        link = str(row.get("link") or "")
        pub = row.get("providerPublishTime")
        when = ""
        if isinstance(pub, (int, float)):
            when = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()
        hub_id = _match_hub(title) or "US"
        items.append(
            {
                "id": f"ys-{row.get('uuid') or hash(title) & 0xFFFF}",
                "source": "Yahoo Finance",
                "hub": hub_id,
                "headline": _short(title),
                "detail": _short(str(row.get("publisher") or "Market news"), 80),
                "impact": "News",
                "when": when,
                "url": link or "https://finance.yahoo.com",
            }
        )
    return items


def _fallback_seed() -> list[dict[str, Any]]:
    """Static context so the map stays useful if feeds are down."""
    now = datetime.now(timezone.utc).isoformat()
    seeds = [
        ("US", "Watch Fed speakers & USD — gold’s main driver", "Macro"),
        ("CN", "China physical demand sets the tone for dips", "Demand"),
        ("IN", "India buying often lifts prices into festivals", "Demand"),
        ("AU", "Aussie mine supply & RBA keep AUD–gold linked", "Supply"),
        ("GB", "London LBMA sets the global bullion benchmark", "Market"),
        ("CH", "Swiss refiners move bars between East & West", "Flow"),
        ("EU", "ECB path vs Fed spreads moves the euro & gold", "Macro"),
        ("ZA", "SA mining headlines can nudge sentiment", "Supply"),
        ("RU", "Reserve buying & sanctions reshape flows", "Flow"),
        ("AE", "Dubai is a key physical trading hub", "Market"),
        ("JP", "BoJ & yen swings spill into gold via USD crosses", "Macro"),
        ("CA", "Canadian miners & CAD track risk appetite", "Supply"),
        ("TR", "Turkish households often rush into gold on lira stress", "Demand"),
        ("BR", "Brazil risk tone feeds EM commodity flows", "Macro"),
        ("NZ", "RBNZ surprises can jolt risk FX and gold", "Macro"),
        ("IR", "Iran tension & Hormuz risk lift safe-haven gold bids", "Geopolitics"),
        ("SA", "Saudi / OPEC oil shocks & Gulf CB buying matter", "Macro"),
        ("GH", "Ghana mine output headlines can shift supply tone", "Supply"),
        ("PE", "Peru is a key LatAm gold producer", "Supply"),
        ("MX", "Mexico mining & peso risk feed metals", "Supply"),
        ("KZ", "Kazakhstan produces and accumulates gold reserves", "Flow"),
        ("UZ", "Uzbekistan is a growing Central Asian gold source", "Supply"),
        ("SG", "Singapore vaults & bullion trade set Asian liquidity", "Market"),
        ("HK", "Hong Kong premiums flag China physical demand", "Demand"),
        ("PL", "Poland’s NBP is among the most active CB buyers", "Demand"),
        ("KR", "Korea investment flows add East Asian demand", "Demand"),
        ("TH", "Thai jewellery demand supports Asian offtake", "Demand"),
        ("VN", "Vietnam retail gold rises when the dong is stressed", "Demand"),
        ("EG", "Egypt pound stress often drives household gold buying", "Demand"),
    ]
    out = []
    for hub, headline, impact in seeds:
        out.append(
            {
                "id": f"seed-{hub}",
                "source": "Zenith context",
                "hub": hub,
                "headline": headline,
                "detail": GOLD_HUBS[hub]["role"],
                "impact": impact,
                "when": now,
                "url": None,
            }
        )
    return out


# Phrases that typically lift gold (safe-haven / demand / weaker USD)
_BULL_PATTERNS = [
    (r"\b(dovish|rate cut|cuts rates|easing|qe\b|stimulus|pause hikes)\b", 2.0),
    (r"\b(inflation|cpi|ppi|pce).{0,28}(hot|higher|surge|beat|above|sticky)\b", 1.8),
    (r"\b(weak(er)?\s+dollar|usd\s+fall|dollar\s+drop|dxy\s+(fall|drop|slide))\b", 2.2),
    (r"\b(safe[\s-]?haven|risk[\s-]?off|geopolit|war|conflict|sanctions|tension|missile|strike)\b", 1.8),
    (r"\b(strait of hormuz|hormuz|oil\s+shock|supply\s+shock)\b", 2.0),
    (r"\b(gold\s+(rally|rises?|surge|jump|climb|gain|record|high|bid))\b", 2.4),
    (r"\b(buy|buying|demand|import|jewellery|jewelry|physical|etf\s+inflow)\b", 1.6),
    (r"\b(central\s+bank|reserve).{0,24}(buy|buying|purchase|add)\b", 2.0),
    (r"\b(recession|slowdown|layoff|unemployment\s+rise|jobless\s+rise)\b", 1.3),
    (r"\b(lira|currency\s+crisis|devalu|fx\s+stress)\b", 1.5),
    (r"\b(festival|diwali|wedding\s+season)\b", 1.2),
]

# Phrases that typically pressure gold
_BEAR_PATTERNS = [
    (r"\b(hawkish|rate hike|hikes rates|tighten|higher for longer|qt\b)\b", 2.0),
    (r"\b(strong(er)?\s+dollar|usd\s+rise|dollar\s+rally|dxy\s+(rise|rally))\b", 2.2),
    (r"\b(yields?\s+rise|treasury\s+yield|real\s+yields?\s+up)\b", 1.8),
    (r"\b(gold\s+(fall|drop|slide|slump|sink|decline|sell|pressure))\b", 2.4),
    (r"\b(risk[\s-]?on|equity\s+rally|stocks?\s+rally)\b", 1.2),
    (r"\b(etf\s+outflow|sell[\s-]?off|profit[\s-]?taking)\b", 1.5),
    (r"\b(inflation|cpi|ppi|pce).{0,28}(cool|soft|miss|below|fall|ease)\b", 1.5),
    (r"\b(mine|mining|production|output|supply).{0,24}(surge|boost|record|glut)\b", 1.1),
]

# Structural lean for each hub when headlines are thin (always applied, mild)
_HUB_PRIORS = {
    "US": 0.0,
    "CN": 1.15,   # physical demand
    "IN": 1.25,   # jewellery demand
    "TR": 1.1,    # retail gold in FX stress
    "RU": 0.95,   # reserve / sanctions flows
    "IR": 1.2,    # geopolitics → safe-haven bid
    "SA": 0.7,    # oil risk + CB demand
    "AE": 0.55,   # bullion hub / demand
    "PL": 0.85,   # active CB buyer
    "HK": 0.6,    # China gateway premiums
    "SG": 0.5,    # Asian bullion hub
    "KR": 0.45,
    "TH": 0.5,
    "VN": 0.55,
    "EG": 0.7,    # FX stress → household gold
    "CH": 0.45,   # refining / vault flows
    "GB": 0.35,   # LBMA benchmark liquidity
    "EU": 0.15,
    "JP": 0.25,
    "BR": 0.3,
    "NZ": 0.15,
    "AU": -0.55,  # mining supply
    "ZA": -0.5,
    "CA": -0.45,
    "GH": -0.55,  # major African producer
    "PE": -0.5,
    "MX": -0.4,
    "KZ": -0.25,  # supply + some CB buying
    "UZ": -0.35,
}

_IMPACT_WEIGHT = {
    "High": 1.45,
    "Medium": 1.15,
    "News": 1.05,
    "Demand": 1.25,
    "Macro": 1.2,
    "Supply": 1.0,
    "Flow": 1.05,
    "Market": 1.0,
    "Holiday": 0.4,
}

_FF_DISK_CACHE = Path(__file__).resolve().parent / ".cache_ff_calendar.json"


def _read_ff_disk_cache() -> list[dict[str, Any]]:
    try:
        if not _FF_DISK_CACHE.exists():
            return []
        raw = json.loads(_FF_DISK_CACHE.read_text())
        if isinstance(raw, list) and raw:
            return raw
    except Exception:
        pass
    return []


def _write_ff_disk_cache(rows: list) -> None:
    try:
        _FF_DISK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _FF_DISK_CACHE.write_text(json.dumps(rows)[:2_000_000])
    except Exception as exc:
        logger.debug("FF disk cache write failed: %s", exc)


def _parse_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace("%", "").replace(",", "")
    if not s or s in ("—", "-", "N/A", "n/a", "All Day"):
        return None
    # "54.2K" / "1.2M" style
    mult = 1.0
    if s[-1:].upper() == "K":
        mult = 1_000.0
        s = s[:-1]
    elif s[-1:].upper() == "M":
        mult = 1_000_000.0
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _parse_forecast_prior(detail: str) -> tuple[Optional[float], Optional[float]]:
    text = detail or ""
    m = re.search(
        r"Forecast\s+(.+?)\s*·\s*Prior\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None
    return _parse_num(m.group(1)), _parse_num(m.group(2))


def _score_calendar_event(headline: str, detail: str, impact: str) -> tuple[float, str]:
    """
    Score Forex Factory-style events for gold.
    Returns (score, reason_tag). Positive = gold-friendly.
    """
    title = (headline or "").lower()
    # strip " · High" suffix from our short headlines
    title = re.sub(r"\s·\s(high|medium|low|holiday|news|macro|demand|supply|flow|market)\s*$", "", title)
    forecast, prior = _parse_forecast_prior(detail or "")
    impact_w = _IMPACT_WEIGHT.get(str(impact or "News"), 1.0)
    delta = None
    if forecast is not None and prior is not None and prior != 0:
        delta = (forecast - prior) / abs(prior) if abs(prior) > 1e-9 else forecast - prior
    elif forecast is not None and prior is not None:
        delta = forecast - prior

    # Inflation prints: hotter → bullish gold
    if re.search(r"\b(cpi|ppi|pce|inflation|core\s+inflation)\b", title):
        if delta is not None:
            score = 2.2 * (1 if delta > 0 else -1 if delta < 0 else 0) * impact_w
            return score, "inflation"
        return 0.55 * impact_w, "inflation-watch"

    # Labour strength: stronger jobs → firmer USD → bearish gold (esp. US-style prints)
    if re.search(r"\b(non-?farm|payroll|employment\s+change|employment\s+level)\b", title):
        if delta is not None:
            score = -2.0 * (1 if delta > 0 else -1 if delta < 0 else 0) * impact_w
            return score, "jobs"
        return -0.35 * impact_w, "jobs-watch"

    if re.search(r"\b(unemployment|jobless\s+claims|claimant)\b", title):
        if delta is not None:
            # higher unemployment → bullish gold
            score = 1.8 * (1 if delta > 0 else -1 if delta < 0 else 0) * impact_w
            return score, "unemployment"
        return 0.3 * impact_w, "unemployment-watch"

    # Policy / rates: higher rate path → bearish gold
    if re.search(r"\b(interest\s+rate|cash\s+rate|fed\s+funds|bank\s+rate|repo\s+rate|fomc|rate\s+decision)\b", title):
        if delta is not None:
            score = -2.4 * (1 if delta > 0 else -1 if delta < 0 else 0) * impact_w
            return score, "rates"
        if re.search(r"\b(cut|dovish)\b", title):
            return 1.8 * impact_w, "rates-dovish"
        if re.search(r"\b(hike|hawkish)\b", title):
            return -1.8 * impact_w, "rates-hawkish"
        return -0.25 * impact_w, "rates-watch"

    if re.search(r"\b(fomc|fed\s+chair|powell|lagarde|bailey|ueda)\b", title):
        return 0.45 * impact_w, "policy-speak"  # event risk often supports gold

    # Growth: stronger growth → mild bearish for gold
    if re.search(r"\b(gdp|retail\s+sales|ism|pmi|industrial\s+production)\b", title):
        if delta is not None:
            score = -1.2 * (1 if delta > 0 else -1 if delta < 0 else 0) * impact_w
            return score, "growth"
        return -0.2 * impact_w, "growth-watch"

    # Consumer confidence / housing — mild
    if re.search(r"\b(consumer\s+confidence|housing|home\s+sales)\b", title):
        if delta is not None:
            score = -0.8 * (1 if delta > 0 else -1 if delta < 0 else 0) * impact_w
            return score, "activity"
        return 0.0, "activity-watch"

    # High-impact unknown → slight safe-haven bid into the print
    if str(impact).lower() == "high":
        return 0.4 * impact_w, "high-impact"

    return 0.0, "other"


def _score_text_for_gold(text: str) -> float:
    low = (text or "").lower()
    score = 0.0
    for pat, w in _BULL_PATTERNS:
        if re.search(pat, low):
            score += w
    for pat, w in _BEAR_PATTERNS:
        if re.search(pat, low):
            score -= w
    return score


def _score_item(item: dict[str, Any]) -> float:
    headline = str(item.get("headline") or "")
    detail = str(item.get("detail") or "")
    impact = str(item.get("impact") or "News")
    source = str(item.get("source") or "")

    text_score = _score_text_for_gold(f"{headline} {detail}")
    cal_score = 0.0
    if "Forex Factory" in source or impact in ("High", "Medium", "Low", "Holiday"):
        cal_score, _ = _score_calendar_event(headline, detail, impact)

    # Prefer the stronger structured calendar read when present
    if abs(cal_score) >= 0.2:
        return cal_score + 0.35 * text_score
    return text_score * _IMPACT_WEIGHT.get(impact, 1.0)


def conclude_hub_bias(hub_id: str, headlines: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Plain-English bullish / bearish / neutral call for gold from that country's updates.
    Uses calendar math (forecast vs prior) + headline keywords + hub priors.
    """
    hub = GOLD_HUBS.get(hub_id, {})
    name = hub.get("name", hub_id)
    prior = float(_HUB_PRIORS.get(hub_id, 0.0))

    if not headlines:
        avg = prior
        conf = 28
    else:
        total = sum(_score_item(item) for item in headlines)
        # Don't let long lists wash out the signal too hard
        n = max(len(headlines), 1)
        avg = total / (0.65 + 0.35 * math.sqrt(n)) + prior
        conf = min(94, int(42 + abs(avg) * 16 + min(n, 8) * 3))

    # Decisive thresholds — avoid "everything neutral"
    if avg >= 0.45:
        bias, label = "BULLISH", "Bullish"
        summary = (
            f"{name} looks bullish for gold — demand, softer policy/USD tone, or hotter "
            f"inflation/safe-haven signals in the latest updates."
        )
    elif avg <= -0.45:
        bias, label = "BEARISH", "Bearish"
        summary = (
            f"{name} looks bearish for gold — firmer rates/USD, cooler inflation, stronger "
            f"growth/jobs, or heavier mine-supply tone in the latest updates."
        )
    else:
        bias, label = "NEUTRAL", "Neutral"
        summary = (
            f"{name} is near balance for gold — bullish and bearish cues are close; wait for "
            f"the next high-impact print to tip it."
        )

    return {
        "bias": bias,
        "label": label,
        "score": round(avg, 2),
        "confidence": conf,
        "summary": summary,
    }


def build_gold_world_news(force: bool = False) -> dict[str, Any]:
    """Multi-market world desk (metals, FX, crypto) — gold path kept for API compat."""
    return build_world_desk_news(force=force)


# Yahoo symbols → Zenith instruments
_MARKET_FEEDS: list[tuple[str, str, list[str], str]] = [
    ("GC=F", "Gold", ["GOLD"], "metals"),
    ("SI=F", "Silver", ["SILVER", "GOLD"], "metals"),
    ("CL=F", "Crude", ["USOIL", "GOLD"], "energy"),
    ("^DJI", "Dow", ["US30"], "indices"),
    ("BTC-USD", "Bitcoin", ["BTCUSD"], "crypto"),
    ("ETH-USD", "Ethereum", ["BTCUSD"], "crypto"),
    ("EURUSD=X", "EURUSD", ["EURUSD"], "fx"),
    ("GBPUSD=X", "GBPUSD", ["GBPUSD"], "fx"),
    ("USDJPY=X", "USDJPY", ["USDJPY"], "fx"),
]

_ASSET_HINTS: dict[str, list[str]] = {
    "GOLD": [r"\bgold\b", r"\bxau\b", r"\bbullion\b", r"\bglod\b"],
    "SILVER": [r"\bsilver\b", r"\bxag\b"],
    "EURUSD": [r"\beurusd\b", r"\beuro\b", r"\beur\b", r"\becb\b"],
    "GBPUSD": [r"\bgbp\b", r"\bstirling\b", r"\bpound\b", r"\bboe\b", r"\bgbpusd\b"],
    "USDJPY": [r"\busdjpy\b", r"\byen\b", r"\bboj\b", r"\bjpy\b"],
    "BTCUSD": [r"\bbitcoin\b", r"\bbtc\b", r"\bcrypto\b", r"\bethereum\b", r"\beth\b"],
    "USOIL": [r"\boil\b", r"\bcrude\b", r"\bopec\b", r"\bbrent\b", r"\bwti\b"],
    "US30": [r"\bus30\b", r"\bdow\b", r"\bdjia\b", r"\bwall\s*street\b"],
}

_FX_BULL_FOR_PAIR = [
    # Weaker USD often lifts EUR/GBP/XAU/BTC
    (r"\b(weak(er)?\s+dollar|usd\s+fall|dollar\s+drop|dxy\s+(fall|drop|slide))\b", 2.0),
    (r"\b(dovish|rate cut|easing)\b", 1.5),
]
_FX_BEAR_FOR_PAIR = [
    (r"\b(strong(er)?\s+dollar|usd\s+rise|dollar\s+rally|dxy\s+(rise|rally))\b", 2.0),
    (r"\b(hawkish|rate hike|tighten)\b", 1.5),
]
_CRYPTO_BULL = [
    (r"\b(bitcoin|btc).{0,20}(rally|surge|high|jump|etf\s+inflow)\b", 2.4),
    (r"\b(risk[\s-]?on|crypto\s+rally)\b", 1.4),
]
_CRYPTO_BEAR = [
    (r"\b(bitcoin|btc).{0,20}(fall|drop|crash|sell)\b", 2.4),
    (r"\b(risk[\s-]?off|crypto\s+sell)\b", 1.4),
]


def _tag_assets(item: dict[str, Any], forced: Optional[list[str]] = None) -> list[str]:
    text = f"{item.get('headline') or ''} {item.get('detail') or ''}".lower()
    found: list[str] = list(forced or [])
    for asset, pats in _ASSET_HINTS.items():
        if asset in found:
            continue
        if any(re.search(p, text) for p in pats):
            found.append(asset)
    if not found:
        # Calendar currencies → map to majors we trade
        hub = item.get("hub")
        if hub == "EU":
            found.append("EURUSD")
        elif hub == "GB":
            found.append("GBPUSD")
        elif hub == "JP":
            found.append("USDJPY")
        elif hub in ("US", "CN", "IN", "CH", "AE"):
            found.append("GOLD")
        elif hub in ("SG", "KR", "HK"):
            found.extend(["BTCUSD", "GOLD"])
    # Always keep gold as soft link for macro hubs when empty
    if not found:
        found.append("GOLD")
    # unique preserve order
    out: list[str] = []
    for a in found:
        if a not in out:
            out.append(a)
    return out


def _score_text_for_asset(asset: str, text: str) -> float:
    """Directional lean for a specific Zenith symbol (+ = bullish that asset)."""
    low = (text or "").lower()
    score = _score_text_for_gold(low)  # macro (USD/Fed) baseline

    if asset in ("EURUSD", "GBPUSD"):
        # Weaker USD → bullish for EUR/GBP
        for pat, w in _FX_BULL_FOR_PAIR:
            if re.search(pat, low):
                score += w
        for pat, w in _FX_BEAR_FOR_PAIR:
            if re.search(pat, low):
                score -= w
        if asset == "EURUSD" and re.search(r"\beur(o)?\s+(rally|rise|gain|surge)\b", low):
            score += 1.8
        if asset == "EURUSD" and re.search(r"\beur(o)?\s+(fall|drop|slide|weak)\b", low):
            score -= 1.8
        if asset == "GBPUSD" and re.search(r"\b(sterling|pound|gbp)\s+(rally|rise|gain)\b", low):
            score += 1.8
        if asset == "GBPUSD" and re.search(r"\b(sterling|pound|gbp)\s+(fall|drop|slide)\b", low):
            score -= 1.8
    elif asset == "USDJPY":
        # Stronger USD / hawkish → often lifts USDJPY; risk-off can crush yen shorts too — keep simple
        for pat, w in _FX_BEAR_FOR_PAIR:  # strong USD
            if re.search(pat, low):
                score += w  # bullish USDJPY
        for pat, w in _FX_BULL_FOR_PAIR:
            if re.search(pat, low):
                score -= w
        if re.search(r"\byen\s+(rally|rise|strength)\b", low):
            score -= 1.8
        if re.search(r"\byen\s+(fall|weak|slide)\b", low):
            score += 1.8
    elif asset == "BTCUSD":
        score *= 0.35  # mute pure gold safe-haven for crypto
        for pat, w in _CRYPTO_BULL:
            if re.search(pat, low):
                score += w
        for pat, w in _CRYPTO_BEAR:
            if re.search(pat, low):
                score -= w
    elif asset == "SILVER":
        if re.search(r"\bsilver\s+(rally|rise|surge)\b", low):
            score += 2.0
        if re.search(r"\bsilver\s+(fall|drop|slide)\b", low):
            score -= 2.0
    elif asset == "USOIL":
        score = 0.0
        if re.search(r"\b(oil|crude|brent|wti).{0,16}(rally|surge|jump|high)\b", low):
            score += 2.2
        if re.search(r"\b(oil|crude|brent|wti).{0,16}(fall|drop|slide|glut)\b", low):
            score -= 2.2
        if re.search(r"\bopec.{0,20}(cut|extend)\b", low):
            score += 1.5
    # GOLD keeps baseline gold score
    return score


def _conclude_asset_bias(symbol: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "symbol": symbol,
            "bias": "NEUTRAL",
            "label": "Neutral",
            "score": 0.0,
            "confidence": 30,
            "summary": f"No fresh headlines tagged to {symbol} yet.",
            "count": 0,
        }
    total = 0.0
    for it in items:
        text = f"{it.get('headline') or ''} {it.get('detail') or ''}"
        total += _score_text_for_asset(symbol, text)
    n = max(len(items), 1)
    avg = total / (0.65 + 0.35 * math.sqrt(n))
    conf = min(92, int(40 + abs(avg) * 14 + min(n, 8) * 3))
    if avg >= 0.45:
        bias, label = "BULLISH", "Bullish"
        summary = f"News lean is bullish for {symbol} from the latest desk headlines."
    elif avg <= -0.45:
        bias, label = "BEARISH", "Bearish"
        summary = f"News lean is bearish for {symbol} from the latest desk headlines."
    else:
        bias, label = "NEUTRAL", "Neutral"
        summary = f"{symbol} news lean is mixed — wait for the next high-impact print."
    return {
        "symbol": symbol,
        "bias": bias,
        "label": label,
        "score": round(avg, 2),
        "confidence": conf,
        "summary": summary,
        "count": len(items),
    }


def build_world_desk_news(force: bool = False) -> dict[str, Any]:
    """
    Multi-asset world desk: metals + FX majors + crypto (+ oil colour).
    Same hub map; each hub summarizes for gold/macro, plus a markets board.
    """
    now = time.time()
    if not force and _CACHE["payload"] and now - float(_CACHE["ts"]) < _CACHE_TTL:
        return _CACHE["payload"]

    seed_items = _fallback_seed()
    collected: list[dict[str, Any]] = []
    collected.extend(_fetch_forex_factory())

    for yahoo_sym, label, assets, _cls in _MARKET_FEEDS:
        rows = _fetch_yahoo_rss(yahoo_sym, label)
        for r in rows:
            r["assets"] = list(assets)
            r["market_class"] = _cls
        collected.extend(rows)

    for q in (
        "gold price federal reserve",
        "euro dollar ECB",
        "pound dollar Bank of England",
        "yen Bank of Japan USDJPY",
        "bitcoin crypto ETF",
        "oil OPEC crude",
        "silver price",
    ):
        collected.extend(_fetch_yahoo_search(q))

    collected.extend(seed_items)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in collected:
        key = str(item.get("headline") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        item = dict(item)
        item["assets"] = _tag_assets(item, item.get("assets"))
        unique.append(item)

    by_hub: dict[str, list[dict[str, Any]]] = {hid: [] for hid in GOLD_HUBS}
    for item in unique:
        hub = item.get("hub")
        if hub in by_hub:
            by_hub[hub].append(item)

    seeds_by_hub = {s["hub"]: s for s in _fallback_seed()}
    for hub_id, headlines in by_hub.items():
        if not headlines and hub_id in seeds_by_hub:
            seed = dict(seeds_by_hub[hub_id])
            seed["assets"] = _tag_assets(seed)
            headlines.append(seed)

    markers = []
    for hub_id, hub in GOLD_HUBS.items():
        headlines = by_hub.get(hub_id) or []
        headlines.sort(
            key=lambda x: (
                0 if x.get("impact") == "High" else 1 if x.get("impact") == "Medium" else 2,
                x.get("when") or "",
            )
        )
        top = headlines[:5]
        conclusion = conclude_hub_bias(hub_id, headlines)
        # Multi-asset colour for hover — only the focused trading book
        try:
            from config import INSTRUMENT_UNIVERSE as _universe
        except Exception:
            _universe = ("GOLD", "USOIL")
        asset_counts: dict[str, int] = {}
        for h in headlines:
            for a in h.get("assets") or []:
                if a in _universe:
                    asset_counts[a] = asset_counts.get(a, 0) + 1
        top_assets = sorted(asset_counts, key=asset_counts.get, reverse=True)[:4]
        summary = conclusion["summary"]
        if top_assets:
            summary = summary.replace(" for gold", f" for markets ({', '.join(top_assets)})")
            if " for gold" in conclusion["summary"] and " for markets" not in summary:
                summary = f"{conclusion['summary']} Touches: {', '.join(top_assets)}."
        markers.append(
            {
                "id": hub_id,
                "name": hub["name"],
                "lat": hub["lat"],
                "lng": hub["lng"],
                "role": hub["role"],
                "count": len(headlines),
                "headlines": top,
                "assets": top_assets,
                "bias": conclusion["bias"],
                "bias_label": conclusion["label"],
                "bias_score": conclusion["score"],
                "bias_confidence": conclusion["confidence"],
                "bias_summary": summary,
            }
        )

    bias_by_hub = {m["id"]: m for m in markers}
    feed = []
    for item in unique[:48]:
        hub_meta = bias_by_hub.get(item.get("hub") or "")
        row = dict(item)
        if hub_meta:
            row["bias"] = hub_meta["bias"]
            row["bias_label"] = hub_meta["bias_label"]
            row["bias_summary"] = hub_meta["bias_summary"]
            row["bias_confidence"] = hub_meta["bias_confidence"]
            row["country"] = hub_meta["name"]
        feed.append(row)

    # Markets board — focused book only (no colour instruments off the desk)
    try:
        from config import INSTRUMENT_UNIVERSE

        board_syms = list(INSTRUMENT_UNIVERSE)
    except Exception:
        board_syms = ["GOLD", "USOIL"]

    markets = []
    for sym in board_syms:
        tagged = [it for it in unique if sym in (it.get("assets") or [])]
        markets.append(_conclude_asset_bias(sym, tagged))

    world = _aggregate_from_markers(markers)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sources": ["Forex Factory", "Yahoo Finance"],
        "total_headlines": len(unique),
        "markers": markers,
        "feed": feed,
        "markets": markets,
        "world": world,
        "desk": "multi-market",
    }
    _CACHE["ts"] = now
    _CACHE["payload"] = payload
    return payload


# Hub importance for gold when averaging country desks into one world variable
_WORLD_HUB_WEIGHT = {
    "US": 1.45,
    "CN": 1.35,
    "IN": 1.25,
    "EU": 1.2,
    "GB": 1.15,
    "IR": 1.2,    # geopolitics punches above calendar coverage
    "SA": 1.1,
    "CH": 1.05,
    "JP": 1.05,
    "TR": 1.1,
    "RU": 1.05,
    "AE": 1.0,
    "PL": 1.05,
    "HK": 1.0,
    "SG": 0.95,
    "AU": 0.95,
    "GH": 0.95,
    "CA": 0.9,
    "ZA": 0.9,
    "PE": 0.9,
    "KZ": 0.9,
    "KR": 0.85,
    "MX": 0.85,
    "BR": 0.85,
    "TH": 0.85,
    "VN": 0.85,
    "EG": 0.85,
    "UZ": 0.8,
    "NZ": 0.75,
}


def _aggregate_from_markers(markers: list[dict[str, Any]]) -> dict[str, Any]:
    """Weighted average of country gold leans → one world-news variable."""
    if not markers:
        return {
            "direction": "NEUTRAL",
            "label": "Neutral",
            "confidence": 50.0,
            "score": 0.0,
            "available": False,
            "bullish_hubs": 0,
            "bearish_hubs": 0,
            "neutral_hubs": 0,
            "notes": ["World news desks unavailable"],
            "hub_count": 0,
        }

    weighted = 0.0
    weight_sum = 0.0
    bullish = bearish = neutral = 0
    notes: list[str] = []

    ranked = sorted(
        markers,
        key=lambda m: abs(float(m.get("bias_score") or 0)) * _WORLD_HUB_WEIGHT.get(m["id"], 1.0),
        reverse=True,
    )

    for m in markers:
        score = float(m.get("bias_score") or 0.0)
        hub_w = _WORLD_HUB_WEIGHT.get(m["id"], 1.0)
        conf_w = 0.6 + 0.4 * (float(m.get("bias_confidence") or 50) / 100.0)
        count_w = 1.0 + min(int(m.get("count") or 0), 10) * 0.05
        w = hub_w * conf_w * count_w
        weighted += score * w
        weight_sum += w
        bias = m.get("bias") or "NEUTRAL"
        if bias == "BULLISH":
            bullish += 1
        elif bias == "BEARISH":
            bearish += 1
        else:
            neutral += 1

    avg = weighted / weight_sum if weight_sum else 0.0
    # Map score → confidence (50–92)
    confidence = round(min(92.0, 55.0 + abs(avg) * 12.0), 1)

    if avg >= 0.45:
        direction, label = "BULLISH", "Bullish"
    elif avg <= -0.45:
        direction, label = "BEARISH", "Bearish"
    else:
        direction, label = "NEUTRAL", "Neutral"
        confidence = max(52.0, min(68.0, confidence))

    for m in ranked[:3]:
        notes.append(
            f"{m['name']}: {m.get('bias_label') or m.get('bias')} "
            f"({float(m.get('bias_score') or 0):+.1f})"
        )

    return {
        "direction": direction,
        "label": label,
        "confidence": confidence,
        "score": round(avg, 3),
        "available": True,
        "bullish_hubs": bullish,
        "bearish_hubs": bearish,
        "neutral_hubs": neutral,
        "notes": notes,
        "hub_count": len(markers),
        "summary": (
            f"World news average {label.lower()} for gold "
            f"({bullish} bull / {bearish} bear / {neutral} flat desks)."
        ),
    }


def aggregate_world_news_bias(force: bool = False) -> dict[str, Any]:
    """
    Public entry: cached world-news variable for bias / ML / soft MC.
    Safe to call from signal loop — uses gold_news TTL cache.
    """
    try:
        payload = build_gold_world_news(force=force)
        world = payload.get("world")
        if world:
            return world
        return _aggregate_from_markers(payload.get("markers") or [])
    except Exception as exc:
        logger.warning("World news aggregate failed: %s", exc)
        return {
            "direction": "NEUTRAL",
            "label": "Neutral",
            "confidence": 50.0,
            "score": 0.0,
            "available": False,
            "bullish_hubs": 0,
            "bearish_hubs": 0,
            "neutral_hubs": 0,
            "notes": [f"News layer offline: {exc}"],
            "hub_count": 0,
            "summary": "World news layer unavailable — physics bias only.",
        }
