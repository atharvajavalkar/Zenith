"""Reconcile local trade records with live XM / MetaAPI positions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trade_record import TradeRecord
from strategy.trailing_stop import sl_trail_label

logger = logging.getLogger(__name__)

OPEN_STATUSES = ("open", "partial", "partial2", "partial3")


def _direction(pos: dict) -> str:
    ptype = str(pos.get("type", "")).upper()
    if "BUY" in ptype:
        return "BUY"
    return "SELL"


def _pos_id(pos: dict) -> str:
    return str(pos.get("id") or pos.get("positionId") or "")


def _is_hyper_pos(pos: dict) -> bool:
    """Hyper-scalp positions carry a 'zenith-algo-HS<digits>_SYM' order comment.

    They belong to the separate Hyper engine (own state + own RL), so Zenith
    must never import them as TradeRecords or learn from them.
    """
    import re

    c = str(pos.get("comment") or pos.get("brokerComment") or "")
    return bool(re.search(r"HS\d", c))


def _is_live_row(row: TradeRecord) -> bool:
    if row.trade_mode in ("live", "demo"):
        return True
    msg = (row.message or "").lower()
    return "live" in msg or "demo" in msg or "synced from xm" in msg


def _xm_closed_pnl(broker_client: Any, position_id: str) -> Optional[dict[str, Any]]:
    if hasattr(broker_client, "position_closed_pnl"):
        try:
            # Desk epoch can be >14d — use wide window so closes always resolve
            return broker_client.position_closed_pnl(position_id, days=90)
        except Exception as exc:
            logger.debug("position_closed_pnl error: %s", exc)
    return None


def _stamp_xm_close_message(row: TradeRecord, xm_pnl: float, status: str) -> None:
    pos = row.oanda_transaction_id or "?"
    pnl_bit = f" · XM P&L {xm_pnl:+.2f}"
    if status == "closed_sl":
        row.message = f"Stopped out on XM (pos {pos}){pnl_bit}"
        row.sl_trail_status = row.sl_trail_status or "SL hit"
    elif status == "closed_tp":
        row.message = f"Take-profit on XM (pos {pos}){pnl_bit}"
        row.sl_trail_status = row.sl_trail_status or "TP hit"
    else:
        row.message = f"Closed on XM (pos {pos}){pnl_bit}"


def _apply_xm_deal_to_closed_row(row: TradeRecord, deal: dict[str, Any]) -> bool:
    """Overwrite local fantasy PnL with XM deal cash. Returns True if changed."""
    if deal.get("pnl") is None:
        return False
    xm_pnl = round(float(deal["pnl"]), 2)
    local = round(float(row.pnl or 0), 2)
    changed = abs(local - xm_pnl) >= 0.01
    row.pnl = xm_pnl
    if deal.get("exit_price") is not None:
        row.current_price = float(deal["exit_price"])
    from analysis.trade_outcome import classify_broker_exit

    status = classify_broker_exit(
        row.signal,
        row.entry,
        row.sl,
        row.tp,
        row.current_price,
        symbol=row.symbol or "GOLD",
        tp1=row.tp1,
        tp2=row.tp2,
        tp3=row.tp3,
    )
    row.pnl_status = status
    row.status = "closed"
    _stamp_xm_close_message(row, xm_pnl, status)
    # Mark features so UI/stats never "repair" with leg_pnl
    try:
        feats = (
            json.loads(row.features)
            if isinstance(row.features, str)
            else (row.features or {})
        )
        if not isinstance(feats, dict):
            feats = {}
        feats["xm_deal_pnl"] = xm_pnl
        feats["pnl_source"] = "xm_deal"
        row.features = json.dumps(feats)
    except Exception:
        pass
    return changed


_LAST_PNL_RECONCILE_AT = 0.0
_PNL_RECONCILE_MIN_SEC = 45.0


async def reconcile_closed_pnl_to_xm(
    session: AsyncSession, *, force: bool = False
) -> dict[str, Any]:
    """Force Zenith closed-book PnL = XM deal cash (profit+commission+swap).

    Bulk-loads deals since desk epoch and patches every mismatched closed row.
    """
    import time

    import runtime_config
    from data.broker import broker_client

    global _LAST_PNL_RECONCILE_AT

    if not runtime_config.is_broker_mode() or not broker_client.configured:
        return {"fixed": 0, "checked": 0, "reason": "not broker mode"}

    now = time.time()
    if not force and now - _LAST_PNL_RECONCILE_AT < _PNL_RECONCILE_MIN_SEC:
        return {
            "fixed": 0,
            "checked": 0,
            "throttled": True,
            "age_sec": round(now - _LAST_PNL_RECONCILE_AT, 1),
        }
    _LAST_PNL_RECONCILE_AT = now

    since = None
    try:
        epoch_since, _bal = broker_client._load_desk_epoch()  # noqa: SLF001
        since = epoch_since
    except Exception:
        since = None

    if hasattr(broker_client, "closed_pnl_map_since"):
        deal_map = broker_client.closed_pnl_map_since(since, days=90)
    else:
        deal_map = {}

    rows = (
        await session.execute(
            select(TradeRecord).where(
                TradeRecord.status == "closed",
                TradeRecord.oanda_transaction_id.is_not(None),
            )
        )
    ).scalars().all()

    # Any local row (open or closed) blocks import of that XM position id
    known_pids: set[str] = {
        str(pid).strip()
        for pid in (
            await session.execute(
                select(TradeRecord.oanda_transaction_id).where(
                    TradeRecord.oanda_transaction_id.is_not(None)
                )
            )
        ).scalars().all()
        if str(pid or "").strip()
    }

    fixed = 0
    checked = 0
    for row in rows:
        if row.signal not in ("BUY", "SELL"):
            continue
        pid = str(row.oanda_transaction_id or "").strip()
        if not pid:
            continue
        checked += 1
        deal = deal_map.get(pid)
        if deal is None:
            deal = _xm_closed_pnl(broker_client, pid)
        if not deal:
            continue
        if _apply_xm_deal_to_closed_row(row, deal):
            fixed += 1

    # Import XM closes that never landed in the Zenith log (gap killers)
    imported = 0
    try:
        from strategy.instrument_meta import desk_symbol_from_broker
    except Exception:
        desk_symbol_from_broker = lambda s: str(s or "GOLD").upper()  # noqa: E731

    try:
        import config as _cfg

        desk_universe = {
            str(s).upper() for s in getattr(_cfg, "INSTRUMENT_UNIVERSE", ()) or ()
        }
    except Exception:
        desk_universe = set()

    for pid, deal in deal_map.items():
        if pid in known_pids:
            continue
        side = deal.get("side")
        entry = deal.get("entry_price")
        exit_px = deal.get("exit_price")
        if side not in ("BUY", "SELL") or entry is None:
            continue
        if deal.get("exit_price") is None:
            continue
        broker_sym = str(deal.get("symbol") or "GOLD")
        desk_sym = desk_symbol_from_broker(broker_sym)
        if desk_universe and desk_sym.upper() not in desk_universe:
            continue
        xm_pnl = float(deal["pnl"])
        row = TradeRecord(
            candle_time=str(deal.get("time") or datetime.now(timezone.utc).isoformat())[:32],
            log_time=f"xm-recon-{pid}",
            signal=side,
            symbol=desk_sym,
            entry=float(entry),
            sl=None,
            tp=None,
            current_price=float(exit_px) if exit_px is not None else float(entry),
            lot_size=float(deal["volume"]) if deal.get("volume") is not None else None,
            pnl=xm_pnl,
            partial_pnl=0.0,
            pnl_status="closed_broker",
            status="closed",
            trade_mode=runtime_config.get_trade_mode(),
            oanda_transaction_id=pid,
            message=f"Imported from XM (pos {pid}) · XM P&L {xm_pnl:+.2f}",
            features=json.dumps(
                {
                    "pnl_source": "xm_deal",
                    "xm_deal_pnl": xm_pnl,
                    "symbol": desk_sym,
                    "imported_from_xm": True,
                }
            ),
        )
        session.add(row)
        _apply_xm_deal_to_closed_row(row, deal)
        known_pids.add(pid)
        imported += 1
        fixed += 1

    if fixed or imported:
        await session.commit()
        logger.info(
            "Reconciled closed PnL to XM: fixed=%d imported=%d checked=%d",
            fixed,
            imported,
            checked,
        )
    return {
        "fixed": fixed,
        "imported": imported,
        "checked": checked,
        "deals_mapped": len(deal_map),
    }


async def sync_broker_positions(session: AsyncSession) -> dict[str, Any]:
    """
    XM is source of truth in demo/live.

    - One Zenith row per XM position id (no ghost re-imports)
    - If XM is flat (0 positions), EVERY local open row is closed — including
      orphans with no broker id (stops daily ghost-open glitches)
    - Closed P&L comes from MetaAPI deals when available
    """
    import runtime_config
    from data.broker import broker_client, broker_name

    if not runtime_config.is_broker_mode() or not broker_client.configured:
        return {"synced": False, "reason": "not broker mode or broker not configured"}

    try:
        # Always bypass cache so dashboard never re-opens from a stale list
        if hasattr(broker_client, "get_open_positions"):
            try:
                broker_positions = broker_client.get_open_positions(force=True)
            except TypeError:
                broker_positions = broker_client.get_open_positions()
        else:
            broker_positions = []
    except Exception as exc:
        logger.warning("broker sync fetch failed: %s", exc)
        return {"synced": False, "reason": str(exc)}

    broker_by_id = {_pos_id(p): p for p in broker_positions if _pos_id(p)}

    # ALL open rows — broker-tagged OR orphan — XM flat ⇒ local flat
    all_open_rows = (
        await session.execute(
            select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
        )
    ).scalars().all()

    all_broker_rows = (
        await session.execute(
            select(TradeRecord).where(TradeRecord.oanda_transaction_id.is_not(None))
        )
    ).scalars().all()

    matched = 0
    closed = 0
    imported = 0
    reopened = 0
    orphans_closed = 0
    known_open_ids: set[str] = set()

    # Prefer the open row for a broker id; else newest row
    by_broker_id: dict[str, TradeRecord] = {}
    for row in sorted(all_broker_rows, key=lambda r: r.id or 0):
        bid = str(row.oanda_transaction_id or "")
        if not bid:
            continue
        prev = by_broker_id.get(bid)
        if prev is None:
            by_broker_id[bid] = row
            continue
        prev_open = (prev.pnl_status or "") in OPEN_STATUSES
        row_open = (row.pnl_status or "") in OPEN_STATUSES
        if row_open and not prev_open:
            by_broker_id[bid] = row
        elif row_open == prev_open and (row.id or 0) >= (prev.id or 0):
            by_broker_id[bid] = row

    # ── XM flat: wipe every local open immediately ──────────────────────────
    if not broker_by_id:
        for row in all_open_rows:
            if _should_force_close_local(row):
                _close_row_from_broker(row, broker_client, broker_name)
                if not row.oanda_transaction_id:
                    orphans_closed += 1
                else:
                    closed += 1
        if closed or orphans_closed:
            await session.commit()
            logger.info(
                "XM flat — closed %d broker + %d orphan local opens",
                closed,
                orphans_closed,
            )
        return {
            "synced": True,
            "broker_open": 0,
            "matched": 0,
            "closed_stale": closed,
            "orphans_closed": orphans_closed,
            "imported": 0,
            "reopened": 0,
            "xm_flat": True,
        }

    for row in all_open_rows:
        bid = str(row.oanda_transaction_id or "")

        if bid and bid in broker_by_id:
            pos = broker_by_id[bid]
            known_open_ids.add(bid)
            managed = row.trade_mode in ("live", "demo") and bid
            _apply_pos_to_row(row, pos, managed=bool(managed), broker_name=broker_name)
            matched += 1
            continue

        # Not on XM anymore (missing id, or id gone) → close
        if _should_force_close_local(row):
            _close_row_from_broker(row, broker_client, broker_name)
            if bid:
                closed += 1
            else:
                orphans_closed += 1

    now = datetime.now(timezone.utc).isoformat()
    current_mode = runtime_config.get_trade_mode()
    for bid, pos in broker_by_id.items():
        if bid in known_open_ids:
            continue

        # Hyper-scalp positions are managed by the separate Hyper engine — never
        # import them into Zenith's ledger or feed them to Zenith learning.
        if _is_hyper_pos(pos):
            continue

        existing = by_broker_id.get(bid)
        if existing is not None:
            # Already have a row for this XM id — reopen if needed, never create another
            if (existing.pnl_status or "") not in OPEN_STATUSES or existing.status == "closed":
                existing.status = "executed"
                existing.pnl_status = "open"
                existing.message = (
                    f"Reopened — still open on {broker_name.upper()} (pos {bid})"
                )
                _apply_pos_to_row(existing, pos, managed=True, broker_name=broker_name)
                if not existing.symbol:
                    existing.symbol = str(pos.get("symbol") or "GOLD")
                reopened += 1
            else:
                _apply_pos_to_row(existing, pos, managed=True, broker_name=broker_name)
            known_open_ids.add(bid)
            continue

        # Race guard: native fill may have committed same pid after our initial load
        raced = (
            await session.execute(
                select(TradeRecord).where(TradeRecord.oanda_transaction_id == bid)
            )
        ).scalars().all()
        if raced:
            keeper = sorted(
                raced,
                key=lambda r: (
                    1 if (r.pnl_status or "") in OPEN_STATUSES else 0,
                    0 if "Synced from XM" in str(r.message or "") else 1,
                    r.id or 0,
                ),
                reverse=True,
            )[0]
            for drop in raced:
                if drop.id != keeper.id:
                    await session.delete(drop)
            if (keeper.pnl_status or "") not in OPEN_STATUSES or keeper.status == "closed":
                keeper.status = "executed"
                keeper.pnl_status = "open"
                reopened += 1
            _apply_pos_to_row(keeper, pos, managed=True, broker_name=broker_name)
            by_broker_id[bid] = keeper
            known_open_ids.add(bid)
            continue

        direction = _direction(pos)
        entry = float(pos.get("openPrice") or 0)
        raw_sym = str(pos.get("symbol") or getattr(broker_client, "symbol", "GOLD") or "GOLD")
        try:
            from strategy.instrument_meta import desk_symbol_from_broker

            pos_symbol = desk_symbol_from_broker(raw_sym)
        except Exception:
            pos_symbol = raw_sym
        vol0 = float(pos.get("volume") or 0.1)
        feats = json.dumps(
            {
                "symbol": pos_symbol,
                "SYMBOL": pos_symbol,
                "broker_symbol": raw_sym,
                "source": "xm_sync",
                "lot_size": vol0,
                "entry_lot": vol0,
            }
        )
        session.add(
            TradeRecord(
                candle_time=now,
                log_time=f"xm-sync-{bid}-{now[:19]}",
                signal=direction,
                symbol=pos_symbol,
                entry=entry,
                sl=float(pos["stopLoss"]) if pos.get("stopLoss") else None,
                tp=float(pos["takeProfit"]) if pos.get("takeProfit") else None,
                current_price=float(pos.get("currentPrice") or entry),
                pnl=round(float(pos.get("profit") or 0), 2),
                partial_pnl=0.0,
                lot_size=vol0,
                sl_trail_status=sl_trail_label(
                    direction, entry, float(pos.get("stopLoss") or entry), "open"
                ),
                pnl_status="open",
                status="executed",
                message=f"Synced from XM {pos_symbol} position {bid}",
                oanda_transaction_id=bid,
                trade_mode=current_mode if current_mode in ("demo", "live") else "demo",
                features=feats,
            )
        )
        imported += 1

    await session.commit()

    return {
        "synced": True,
        "broker_open": len(broker_positions),
        "matched": matched,
        "closed_stale": closed,
        "orphans_closed": orphans_closed,
        "imported": imported,
        "reopened": reopened,
        "xm_flat": False,
    }


def _should_force_close_local(row: TradeRecord) -> bool:
    """In demo/live, any lingering open must yield to XM flat / missing position."""
    if row.trade_mode in ("live", "demo"):
        return True
    if _is_live_row(row):
        return True
    # Unlabeled but has XM id — treat as broker-managed
    if row.oanda_transaction_id:
        return True
    return False


def _ensure_entry_lot(row: TradeRecord, volume: float) -> None:
    """Keep original entry lot in features so partials size off original, not remaining."""
    try:
        feats = (
            json.loads(row.features)
            if isinstance(row.features, str)
            else (row.features or {})
        )
        if not isinstance(feats, dict):
            feats = {}
        if not feats.get("entry_lot"):
            # Prefer current lot_size if larger (pre-partial), else broker volume
            base = float(row.lot_size or 0) or float(volume)
            feats["entry_lot"] = max(float(volume), base)
            row.features = json.dumps(feats)
    except Exception:
        pass


def _apply_pos_to_row(
    row: TradeRecord, pos: dict, *, managed: bool, broker_name: str
) -> None:
    row.current_price = float(pos.get("currentPrice") or row.current_price or 0)
    broker_runner_pnl = round(float(pos.get("profit") or 0), 2)
    booked = float(row.partial_pnl or 0.0)
    # Only add booked partials when remaining volume is below entry lot (real scale-out).
    # Otherwise XM profit alone is truth — avoids ghost floating inflation.
    entry_lot = None
    try:
        feats = (
            json.loads(row.features)
            if isinstance(row.features, str)
            else (row.features or {})
        )
        if isinstance(feats, dict) and feats.get("entry_lot"):
            entry_lot = float(feats["entry_lot"])
    except Exception:
        entry_lot = None
    rem = float(pos.get("volume") or row.lot_size or 0)
    # Ignore tiny XM volume rounding (e.g. 0.91 → 0.90); real partials cut much more
    scaled_out = bool(
        entry_lot and rem > 0 and (entry_lot - rem) >= max(0.02, entry_lot * 0.05)
    )
    if managed:
        row.pnl = round((booked + broker_runner_pnl) if scaled_out else broker_runner_pnl, 2)
        if not scaled_out:
            row.partial_pnl = 0.0
        if pos.get("stopLoss"):
            row.sl = float(pos["stopLoss"])
        if pos.get("takeProfit"):
            row.tp = float(pos["takeProfit"])
        if pos.get("volume"):
            vol = float(pos["volume"])
            _ensure_entry_lot(row, vol)
            row.lot_size = vol  # remaining on broker
    else:
        row.pnl = broker_runner_pnl
        if pos.get("stopLoss"):
            row.sl = float(pos["stopLoss"])
        if pos.get("takeProfit"):
            row.tp = float(pos["takeProfit"])
        row.message = f"Live {row.signal} on {broker_name.upper()} (pos {_pos_id(pos)})"
    if not row.symbol:
        try:
            from strategy.instrument_meta import desk_symbol_from_broker

            row.symbol = desk_symbol_from_broker(
                str(pos.get("symbol") or row.symbol or "GOLD")
            )
        except Exception:
            row.symbol = str(pos.get("symbol") or row.symbol or "GOLD")


def _close_row_from_broker(row: TradeRecord, broker_client: Any, broker_name: str) -> None:
    bid = str(row.oanda_transaction_id or "")
    deal = _xm_closed_pnl(broker_client, bid) if bid else None
    if deal and deal.get("pnl") is not None:
        _apply_xm_deal_to_closed_row(row, deal)
        return

    # No deal yet (MetaAPI lag) — do NOT invent leg_pnl; keep last XM float if any
    from analysis.trade_outcome import classify_broker_exit

    exit_px = float(row.current_price) if row.current_price is not None else None
    status = classify_broker_exit(
        row.signal,
        row.entry,
        row.sl,
        row.tp,
        exit_px,
        symbol=row.symbol or "GOLD",
        tp1=row.tp1,
        tp2=row.tp2,
        tp3=row.tp3,
    )
    row.pnl_status = status
    row.status = "closed"
    broker_u = broker_name.upper()
    pos_bit = f"pos {bid}" if bid else "synced"
    pnl_bit = f" · pending XM deal · last ${float(row.pnl or 0):+.2f}"
    row.message = f"Closed on {broker_u} ({pos_bit}){pnl_bit}"
    # Reconcile will stamp true XM P&L on next pass


async def dedupe_broker_history(session: AsyncSession) -> dict[str, Any]:
    """
    Keep one row per XM position id.
    Preference: open > closed with real tiny |pnl| from XM > newest.
    Backfill closed P&L from MetaAPI deals when |pnl| looks like fantasy (>> broker).
    """
    import runtime_config
    from data.broker import broker_client

    if not runtime_config.is_broker_mode() or not broker_client.configured:
        return {"deduped": 0, "pnl_fixed": 0, "reason": "not broker mode"}

    rows = (
        await session.execute(
            select(TradeRecord).where(TradeRecord.oanda_transaction_id.is_not(None))
        )
    ).scalars().all()

    groups: dict[str, list[TradeRecord]] = {}
    for r in rows:
        bid = str(r.oanda_transaction_id or "")
        if bid:
            groups.setdefault(bid, []).append(r)

    deleted = 0
    pnl_fixed = 0

    for bid, group in groups.items():
        if len(group) == 1:
            keeper = group[0]
        else:
            def _score(r: TradeRecord) -> tuple:
                is_open = 1 if (r.pnl_status or "") in OPEN_STATUSES else 0
                is_native = 0 if str(r.log_time or "").startswith("xm-sync-") else 1
                return (is_open, is_native, r.id or 0)

            group_sorted = sorted(group, key=_score, reverse=True)
            keeper = group_sorted[0]
            for drop in group_sorted[1:]:
                await session.delete(drop)
                deleted += 1

        if (keeper.pnl_status or "") not in OPEN_STATUSES and keeper.oanda_transaction_id:
            deal = _xm_closed_pnl(broker_client, str(keeper.oanda_transaction_id))
            if deal and deal.get("pnl") is not None:
                if _apply_xm_deal_to_closed_row(keeper, deal):
                    pnl_fixed += 1

    await session.commit()
    return {"deduped": deleted, "pnl_fixed": pnl_fixed, "groups": len(groups)}


async def reset_trade_history(session: AsyncSession, sync_after: bool = True) -> dict[str, Any]:
    """Delete all trade history and optionally import current XM open positions."""
    result = await session.execute(delete(TradeRecord))
    await session.commit()
    cleared = result.rowcount or 0

    out: dict[str, Any] = {"cleared": cleared, "synced": None}
    if sync_after:
        out["synced"] = await sync_broker_positions(session)
    return out


# Throttle dashboard-driven syncs so every poll stays XM-true without hammering MetaAPI
_LAST_DASH_SYNC_AT = 0.0
_DASH_SYNC_MIN_SEC = 8.0


async def sync_broker_throttled(
    session: AsyncSession, *, min_interval: float = _DASH_SYNC_MIN_SEC
) -> dict[str, Any]:
    """Used by /trades/summary — keep open board aligned with XM on a short leash."""
    import time

    global _LAST_DASH_SYNC_AT
    now = time.time()

    # Never throttle while local opens disagree with a fresh XM flat check
    local_opens = (
        await session.execute(
            select(TradeRecord).where(TradeRecord.pnl_status.in_(OPEN_STATUSES))
        )
    ).scalars().all()
    force = bool(local_opens)

    if not force and now - _LAST_DASH_SYNC_AT < min_interval:
        # Still reconcile closed PnL on a longer leash so Zenith book tracks XM
        recon = await reconcile_closed_pnl_to_xm(session, force=False)
        return {
            "synced": False,
            "throttled": True,
            "age_sec": round(now - _LAST_DASH_SYNC_AT, 1),
            "pnl_reconcile": recon,
        }
    _LAST_DASH_SYNC_AT = now
    result = await sync_broker_positions(session)
    recon = await reconcile_closed_pnl_to_xm(session, force=False)
    result["pnl_reconcile"] = recon
    return result
