
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

import gspread
from fastapi import FastAPI
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest


HEADERS = [
    "symbol",
    "qty",
    "gain_pct",
    "peak_gain_pct",
    "armed",
    "drop_from_peak_pct",
    "action",
    "order_id",
    "notes",
]

PROTECTIVE_LAST_ACTIONS = {"SELL_SUBMITTED", "SELL_PROTECTED"}
LAST_STATUS: Dict[str, Any] = {"state": "starting"}


@dataclass(frozen=True)
class Config:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool
    google_sheet_id: str
    google_service_account_json: str

    seller_tab: str
    sell_arm_gain_pct: Decimal
    sell_trail_drop_pct: Decimal
    sell_min_market_value: Decimal
    sell_dry_run: bool
    sell_only_market_hours: bool
    sell_order_lookback_minutes: int

    poll_seconds: int
    bot_auto_start: bool


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_decimal(name: str, default: str) -> Decimal:
    value = os.getenv(name, default)
    try:
        return Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise RuntimeError(f"Invalid decimal env var {name}={value!r}") from exc


def load_config() -> Config:
    return Config(
        alpaca_api_key=env_required("ALPACA_API_KEY"),
        alpaca_secret_key=env_required("ALPACA_SECRET_KEY"),
        alpaca_paper=env_bool("ALPACA_PAPER", True),
        google_sheet_id=env_required("GOOGLE_SHEET_ID"),
        google_service_account_json=env_required("GOOGLE_SERVICE_ACCOUNT_JSON"),
        seller_tab=os.getenv("SELLER_TAB", "Seller").strip() or "Seller",
        sell_arm_gain_pct=env_decimal("SELL_ARM_GAIN_PCT", "12"),
        sell_trail_drop_pct=env_decimal("SELL_TRAIL_DROP_PCT", "4"),
        sell_min_market_value=env_decimal("SELL_MIN_MARKET_VALUE", "0"),
        sell_dry_run=env_bool("SELL_DRY_RUN", True),
        sell_only_market_hours=env_bool("SELL_ONLY_MARKET_HOURS", True),
        sell_order_lookback_minutes=env_int("SELL_ORDER_LOOKBACK_MINUTES", 60),
        poll_seconds=env_int("POLL_SECONDS", 60),
        bot_auto_start=env_bool("BOT_AUTO_START", True),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def decimal_to_sheet(value: Optional[Decimal], places: str = "0.0000") -> str:
    if value is None:
        return ""
    return str(value.quantize(Decimal(places)))


def str_to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_side_text(value: Any) -> str:
    # Handles strings like "long" and enums like PositionSide.LONG.
    return str(value or "").strip().lower()


def safe_order_id(order: Any) -> str:
    order_id = field(order, "id", "")
    return str(order_id) if order_id else ""


def current_gain_pct_from_position(pos: Any) -> Optional[Decimal]:
    # Alpaca unrealized_plpc is usually a fraction, e.g. 0.20 means +20%.
    plpc = to_decimal(field(pos, "unrealized_plpc"))
    if plpc is not None:
        return plpc * Decimal("100")

    market_value = to_decimal(field(pos, "market_value"))
    cost_basis = to_decimal(field(pos, "cost_basis"))
    if market_value is None or cost_basis is None or cost_basis == 0:
        return None

    return ((market_value - cost_basis) / cost_basis) * Decimal("100")


def load_google_credentials(raw_json: str) -> Dict[str, Any]:
    raw_json = raw_json.strip()

    if raw_json.startswith("{"):
        info = json.loads(raw_json)
    else:
        # Optional convenience: allow GOOGLE_SERVICE_ACCOUNT_JSON to point to a mounted file path.
        with open(raw_json, "r", encoding="utf-8") as f:
            info = json.load(f)

    # Railway env vars commonly preserve escaped newlines.
    if "private_key" in info and isinstance(info["private_key"], str):
        info["private_key"] = info["private_key"].replace("\\n", "\n")

    return info


def init_trading_client(config: Config) -> TradingClient:
    return TradingClient(
        config.alpaca_api_key,
        config.alpaca_secret_key,
        paper=config.alpaca_paper,
    )


def init_worksheet(config: Config):
    creds = load_google_credentials(config.google_service_account_json)
    gc = gspread.service_account_from_dict(creds)
    spreadsheet = gc.open_by_key(config.google_sheet_id)

    try:
        worksheet = spreadsheet.worksheet(config.seller_tab)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=config.seller_tab,
            rows=1000,
            cols=len(HEADERS),
        )

    return worksheet


def ensure_grid_size(worksheet: Any, needed_rows: int, needed_cols: int) -> None:
    if worksheet.row_count < needed_rows:
        worksheet.add_rows(needed_rows - worksheet.row_count)
    if worksheet.col_count < needed_cols:
        worksheet.add_cols(needed_cols - worksheet.col_count)


def read_seller_state(worksheet: Any) -> Dict[str, Dict[str, str]]:
    values = worksheet.get_all_values()
    if not values:
        return {}

    headers = [h.strip() for h in values[0]]
    if "symbol" not in headers:
        return {}

    state: Dict[str, Dict[str, str]] = {}
    for row in values[1:]:
        if not any(cell.strip() for cell in row):
            continue

        record = {
            headers[i]: row[i].strip() if i < len(row) else ""
            for i in range(len(headers))
        }

        symbol = record.get("symbol", "").strip().upper()
        if symbol:
            state[symbol] = record

    return state


def write_seller_rows(worksheet: Any, rows: List[List[Any]]) -> None:
    payload = [HEADERS] + rows
    ensure_grid_size(worksheet, max(1000, len(payload) + 10), len(HEADERS))

    # This only clears the Seller tab, never the Screener tab.
    worksheet.clear()
    worksheet.update(
        range_name="A1",
        values=payload,
        value_input_option="USER_ENTERED",
    )


def get_recent_sell_order_symbols(
    trading_client: TradingClient,
    lookback_minutes: int,
) -> Set[str]:
    after = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        side=OrderSide.SELL,
        after=after,
        limit=500,
    )

    orders = trading_client.get_orders(filter=request)
    symbols: Set[str] = set()

    for order in orders:
        symbol = str(field(order, "symbol", "")).strip().upper()
        if symbol:
            symbols.add(symbol)

    return symbols


def can_sell_now(
    trading_client: TradingClient,
    config: Config,
) -> Tuple[bool, str]:
    if not config.sell_only_market_hours:
        return True, "market-hours check disabled"

    clock = trading_client.get_clock()
    if bool(field(clock, "is_open", False)):
        return True, "market open"

    return False, "market closed"


def submit_market_sell(
    trading_client: TradingClient,
    symbol: str,
    qty: Decimal,
) -> Any:
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=float(qty),
        side=OrderSide.SELL,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    return trading_client.submit_order(order_data=order_request)


def build_row(
    symbol: str,
    qty: Optional[Decimal],
    current_gain_pct: Optional[Decimal],
    peak_gain_pct: Optional[Decimal],
    armed: bool,
    drawdown_from_peak_pct: Optional[Decimal],
    last_action: str,
    last_order_id: str,
    notes: str,
) -> List[Any]:
    return [
        symbol,
        decimal_to_sheet(qty, "0.000000000"),
        decimal_to_sheet(current_gain_pct),
        decimal_to_sheet(peak_gain_pct),
        "TRUE" if armed else "FALSE",
        decimal_to_sheet(drawdown_from_peak_pct),
        last_action,
        last_order_id,
        notes,
    ]


def run_cycle(
    config: Config,
    trading_client: TradingClient,
    worksheet: Any,
) -> Dict[str, Any]:
    now = utc_now_iso()

    state = read_seller_state(worksheet)
    positions = trading_client.get_all_positions()

    sell_allowed, sell_allowed_reason = can_sell_now(trading_client, config)

    order_protection_error = ""
    protected_sell_symbols: Set[str] = set()
    try:
        protected_sell_symbols = get_recent_sell_order_symbols(
            trading_client,
            config.sell_order_lookback_minutes,
        )
    except Exception as exc:
        # Safer to block all sells if we cannot inspect recent/open sell orders.
        order_protection_error = str(exc)

    rows: List[List[Any]] = []
    submitted_sells: List[Dict[str, str]] = []
    dry_run_signals: List[str] = []
    blocked_signals: List[Dict[str, str]] = []

    sorted_positions = sorted(
        positions,
        key=lambda p: str(field(p, "symbol", "")).upper(),
    )

    for pos in sorted_positions:
        symbol = str(field(pos, "symbol", "")).strip().upper()
        if not symbol:
            continue

        prior = state.get(symbol, {})

        qty = to_decimal(field(pos, "qty"))
        avg_entry_price = to_decimal(field(pos, "avg_entry_price"))
        current_price = to_decimal(field(pos, "current_price"))
        market_value = to_decimal(field(pos, "market_value"))
        cost_basis = to_decimal(field(pos, "cost_basis"))
        current_gain_pct = current_gain_pct_from_position(pos)

        prior_peak = to_decimal(prior.get("peak_gain_pct"))
        if current_gain_pct is None:
            peak_gain_pct = prior_peak
        elif prior_peak is None:
            peak_gain_pct = current_gain_pct
        else:
            peak_gain_pct = max(prior_peak, current_gain_pct)

        prior_armed = str_to_bool(prior.get("armed"))
        armed = prior_armed or (
            peak_gain_pct is not None
            and peak_gain_pct >= config.sell_arm_gain_pct
        )

        drawdown_from_peak_pct = None
        if peak_gain_pct is not None and current_gain_pct is not None:
            drawdown_from_peak_pct = peak_gain_pct - current_gain_pct

        sell_signal = bool(
            armed
            and drawdown_from_peak_pct is not None
            and drawdown_from_peak_pct >= config.sell_trail_drop_pct
        )

        # Backward compatible with older Seller tabs that used last_action/last_order_id.
        last_action = prior.get("action") or prior.get("last_action", "")
        last_order_id = prior.get("order_id") or prior.get("last_order_id", "")
        notes = ""

        side_text = safe_side_text(field(pos, "side", "long"))
        is_long_position = "long" in side_text or side_text == ""

        has_protective_prior_action = last_action in PROTECTIVE_LAST_ACTIONS
        has_recent_sell_order = symbol in protected_sell_symbols

        if market_value is not None and market_value < config.sell_min_market_value:
            notes = "Below SELL_MIN_MARKET_VALUE; sell blocked."
            sell_signal = False

        if not is_long_position:
            notes = "Non-long position; skipped by v1 seller."
            sell_signal = False

        if sell_signal:
            if qty is None or qty <= 0:
                last_action = "SELL_SIGNAL_BLOCKED"
                notes = "Sell signal, but qty is missing or <= 0."
                blocked_signals.append({"symbol": symbol, "reason": notes})
            elif order_protection_error:
                last_action = "SELL_SIGNAL_BLOCKED"
                notes = f"Order lookup failed; blocked sell for safety: {order_protection_error}"
                blocked_signals.append({"symbol": symbol, "reason": notes})
            elif has_protective_prior_action:
                notes = "Prior SELL_SUBMITTED/SELL_PROTECTED state; skipped duplicate sell."
                blocked_signals.append({"symbol": symbol, "reason": notes})
            elif has_recent_sell_order:
                last_action = "SELL_PROTECTED"
                notes = (
                    "Recent/open sell order found inside "
                    f"{config.sell_order_lookback_minutes} minute lookback; skipped duplicate."
                )
                blocked_signals.append({"symbol": symbol, "reason": notes})
            elif config.sell_dry_run:
                last_action = "DRY_RUN_SELL_SIGNAL"
                notes = "Dry run: would submit market sell for full position."
                dry_run_signals.append(symbol)
            elif not sell_allowed:
                last_action = "SELL_SIGNAL_MARKET_CLOSED"
                notes = f"Sell signal, but sell blocked: {sell_allowed_reason}."
                blocked_signals.append({"symbol": symbol, "reason": notes})
            else:
                order = submit_market_sell(trading_client, symbol, qty)
                last_action = "SELL_SUBMITTED"
                last_order_id = safe_order_id(order)
                notes = "Submitted market sell for full position."
                submitted_sells.append({"symbol": symbol, "order_id": last_order_id})
        elif not notes:
            if armed:
                last_action = "ARMED"
            else:
                last_action = "MONITORING"

        rows.append(
            build_row(
                symbol=symbol,
                qty=qty,
                current_gain_pct=current_gain_pct,
                peak_gain_pct=peak_gain_pct,
                armed=armed,
                drawdown_from_peak_pct=drawdown_from_peak_pct,
                last_action=last_action,
                last_order_id=last_order_id,
                notes=notes,
            )
        )

    write_seller_rows(worksheet, rows)

    result = {
        "state": "ok",
        "updated_at": now,
        "positions": len(rows),
        "dry_run": config.sell_dry_run,
        "sell_allowed": sell_allowed,
        "sell_allowed_reason": sell_allowed_reason,
        "protected_sell_symbols": sorted(protected_sell_symbols),
        "submitted_sells": submitted_sells,
        "dry_run_signals": dry_run_signals,
        "blocked_signals": blocked_signals,
    }

    global LAST_STATUS
    LAST_STATUS = result
    return result


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

CONFIG = load_config()
TRADING_CLIENT = init_trading_client(CONFIG)
WORKSHEET = init_worksheet(CONFIG)

app = FastAPI(title="Seller Bot")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "tab": CONFIG.seller_tab,
        "dry_run": CONFIG.sell_dry_run,
        "auto_start": CONFIG.bot_auto_start,
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    return LAST_STATUS


@app.post("/run")
async def run_once() -> Dict[str, Any]:
    return await asyncio.to_thread(run_cycle, CONFIG, TRADING_CLIENT, WORKSHEET)


async def background_loop() -> None:
    while True:
        started = time.time()
        try:
            result = await asyncio.to_thread(run_cycle, CONFIG, TRADING_CLIENT, WORKSHEET)
            logging.info("Cycle complete: %s", result)
        except Exception as exc:
            logging.exception("Cycle failed")
            global LAST_STATUS
            LAST_STATUS = {
                "state": "error",
                "updated_at": utc_now_iso(),
                "error": str(exc),
            }

        elapsed = time.time() - started
        sleep_for = max(1, CONFIG.poll_seconds - int(elapsed))
        await asyncio.sleep(sleep_for)


@app.on_event("startup")
async def startup_event() -> None:
    if CONFIG.bot_auto_start:
        asyncio.create_task(background_loop())
