from typing import Any, Optional

from pydantic import BaseModel


class SignalResponse(BaseModel):
    LIVE_TIME: str
    CANDLE_TIME: str
    LIVE_PRICE: float
    SIGNAL: str
    DECISION: Optional[str] = None
    ACTION: Optional[str] = None
    REASON: Optional[str] = None
    SESSION: Optional[str] = None
    SETUP_DIRECTION: Optional[str] = None
    BUY_SCORE: Optional[float] = None
    SELL_SCORE: Optional[float] = None
    BIAS_1H: Optional[str] = None
    BIAS_4H: Optional[str] = None
    CONTEXT_4H: Optional[str] = None
    STATE: Optional[str] = None
    SUCCESS_PERCENT: Optional[float] = None
    RR: Optional[str] = None
    ENTRY: Optional[float] = None
    SL: Optional[float] = None
    TP: Optional[float] = None
    TP1: Optional[float] = None
    TP2: Optional[float] = None
    TP3: Optional[float] = None
    DEMAND_TF: Optional[str] = None
    SUPPLY_TF: Optional[str] = None
    SMC_SCORE: Optional[float] = None
    SMC_READ: Optional[str] = None
    SMC_SWEEP: Optional[bool] = None
    SMC_MSS: Optional[bool] = None
    SMC_CHOCH: Optional[bool] = None
    SMC_DISPLACEMENT: Optional[bool] = None
    SMC_FVG: Optional[str] = None
    EXIT: Optional[str] = None
    DATA_SOURCE: Optional[str] = None
    SYMBOL: Optional[str] = None
    LOG_PRICE_1M: Optional[float] = None
    LOG_TIME_1M: Optional[str] = None
    LOG_TIME_5M: Optional[str] = None
    MOVEMENT_5M: Optional[str] = None
    # PA / lab brains often omit classic zone fields — keep optional
    ENTRY_BRAIN: Optional[str] = None
    ZONE_TAG: Optional[str] = None
    BRAIN_TAG: Optional[str] = None


class CandleBar(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


class CandlesResponse(BaseModel):
    timeframe: str
    candles: list[CandleBar]
    data_source: Optional[str] = None
    symbol: Optional[str] = None


class SignalHistoryItem(BaseModel):
    id: int
    candle_time: str
    live_time: str
    live_price: float
    signal: str
    bias_1h: str
    bias_4h: str
    context_4h: str
    state: str
    success_percent: Optional[float] = None
    rr: Optional[str] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    demand_tf: str
    supply_tf: str
    exit_rule: str
    created_at: Optional[str] = None


class SignalHistoryResponse(BaseModel):
    items: list[SignalHistoryItem]
    total: int
    offset: int
    limit: int


class WsSignalMessage(BaseModel):
    type: str = "signal_update"
    signal: dict[str, Any]
