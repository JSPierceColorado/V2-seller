import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Set, Tuple

import gspread
from fastapi import FastAPI
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest


# Seller tab visible columns.
# Keep these at exactly A:F. Old columns G:Z are cleared once on startup/write.
HEADERS = [
    "symbol",
    "gain_pct",
    "peak_gain_pct",
    "armed",
    "drop_from_peak_pct",
    "action",
]

# Action states that prevent repeat sell submissions while Alpaca still shows the position.
PROTECTIVE_LAST_ACTIONS = {"SELL_SUBMITTED", "SELL_PROTECTED"}

TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "stopped",
    "done_for_day",
}

LAST_STATUS: Dict[str, Any] = {"state": "starting"}
STATE_CACHE: Dict[str, Dict[str, str]] = {}
SHEET_CLEANED_ONCE = False


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

    sell_extended_hours: bool
    sell_extended_step_pct: Decimal
    sell_extended_step_seconds: int
    sell_extended_total_timeout_seconds: int
    sell_extended_leave_final_order: bool
    sell_extended_time_in_force: str
    order_poll_interval_seconds: int

    sell_order_lookback_minutes: int
    poll_seconds: int
    error_backoff_seconds: int
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

        sell_extended_hours=env_bool("SELL_EXTENDED_HOURS", False),
        sell_extended_step_pct=env_decimal("SELL_EXTENDED_STEP_PCT", "25"),
        sell_extended_step_seconds=env_int("SELL_EXTENDED_STEP_SECONDS", 5),
        sell_extended_total_timeout_seconds=env_int("SELL_EXTENDED_TOTAL_TIMEOUT_SECONDS", 60),
        sell_extended_leave_final_order=env_bool("SELL_EXTENDED_LEAVE_FINAL_ORDER", False),
        sell_extended_time_in_force=os.getenv("SELL_EXTENDED_TIME_IN_FORCE", "day").strip().lower() or "day",
        order_poll_interval_seconds=env_int("ORDER_POLL_INTERVAL_SECONDS", 1),

        sell_order_lookback_minutes=env_int("SELL_ORDER_LOOKBACK_MINUTES", 60),
        poll_seconds=env_int("POLL_SECONDS", 10),
        error_backoff_seconds=env_int("ERROR_BACKOFF_SECONDS", 90),
        bot_auto_start=env_bool("BOT_AUTO_START", True),
    )


def load_google_credentials(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Railway/env vars sometimes store service account JSON with escaped newlines.
        return json.loads(raw.replace("\\n", "\n"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def decimal_to_sheet(value: Optional[Decimal], quant: str = "0.0001") -> str:
    if value is None:
        return ""
    try:
        q = Decimal(quant)
        return str(value.quantize(q))
    except Exception:
        return str(value)


def str_to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def safe_side_text(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw


def safe_order_status(order: Any) -> str:
    value = field(order, "status", "")
    raw = str(value).strip().lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw


def safe_order_id(order: Any) -> str:
    value = field(order, "id", "")
    return str(value) if value else ""


def safe_client_order_id(order: Any) -> str:
    value = field(order, "client_order_id", "")
    return str(value) if value else ""


def current_gain_pct_from_position(pos: Any) -> Optional[Decimal]:
    # Alpaca usually provides unrealized_plpc as a decimal ratio, e.g. 0.1234 for +12.34%.
    plpc = to_decimal(field(pos, "unrealized_plpc"))
    if plpc is not None:
        return plpc * Decimal("100")

    market_value = to_decimal(field(pos, "market_value"))
    cost_basis = to_decimal(field(pos, "cost_basis"))
    if market_value is None or cost_basis is None or cost_basis == 0:
        return None

    return ((market_value - cost_basis) / cost_basis) * Decimal("100")


def init_clients(config: Config) -> Tuple[TradingClient, StockHistoricalDataClient]:
    trading_client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.alpaca_paper,
    )

    data_client = StockHistoricalDataClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )

    return trading_client, data_client


def init_worksheet(config: Config):
    creds = load_google_credentials(config.google_service_account_json)
    gc = gspread.service_account_from_dict(creds)
    spreadsheet = gc.open_by_key(config.google_sheet_id)

    try:
        worksheet = spreadsheet.worksheet(config.seller_tab)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=config.seller_tab,
            rows=20,
            cols=len(HEADERS),
        )

    return worksheet


def column_letter(col: int) -> str:
    result = ""
    while col:
        col, rem = divmod(col - 1, 26)
        result = chr(65 + rem) + result
    return result


def read_seller_state(worksheet: Any) -> Dict[str, Dict[str, str]]:
    values = worksheet.get_all_values()
    if not values:
        return {}

    headers = [h.strip() for h in values[0]]
    if "symbol" not in headers:
        return {}

    state: Dict[str, Dict[str, str]] = {}
    for row in values[1:]:
        if not any(str(cell).strip() for cell in row):
            continue

        record = {
            headers[i]: str(row[i]).strip() if i < len(row) else ""
            for i in range(len(headers))
        }

        symbol = record.get("symbol", "").strip().upper()
        if symbol:
            state[symbol] = record

    return state


def row_to_record(row: List[Any]) -> Dict[str, str]:
    return {HEADERS[i]: str(row[i]) if i < len(row) else "" for i in range(len(HEADERS))}


def cleanup_old_visible_columns(worksheet: Any) -> None:
    """
    Remove stale headers/values from previous wider Seller layouts.
    This is intentionally separate from the normal A:F update so the sheet becomes visually clean.
    """
    global SHEET_CLEANED_ONCE

    if SHEET_CLEANED_ONCE:
        return

    target_cols = len(HEADERS)

    # Resize removes visible extra columns when possible.
    try:
        if worksheet.col_count != target_cols:
            worksheet.resize(cols=target_cols)
    except Exception as exc:
        logging.warning("Could not resize Seller tab to %s columns: %s", target_cols, exc)

    # If Google Sheets leaves old cells behind for any reason, explicitly clear G:Z.
    # This should run once per process, not every cycle.
    try:
        clear_start_col = target_cols + 1
        worksheet.batch_clear([f"{column_letter(clear_start_col)}:Z"])
    except Exception as exc:
        logging.warning("Could not clear old Seller columns G:Z: %s", exc)

    SHEET_CLEANED_ONCE = True


def write_seller_rows(worksheet: Any, rows: List[List[Any]]) -> None:
    cleanup_old_visible_columns(worksheet)

    payload = [HEADERS] + rows
    target_rows = max(20, len(payload) + 5)
    target_cols = len(HEADERS)

    try:
        if worksheet.row_count != target_rows:
            worksheet.resize(rows=target_rows, cols=target_cols)
        elif worksheet.col_count != target_cols:
            worksheet.resize(cols=target_cols)
    except Exception as exc:
        logging.warning("Could not resize Seller tab; continuing with A:F update: %s", exc)

    blank_row = [""] * target_cols
    padded_payload = payload + [blank_row[:] for _ in range(target_rows - len(payload))]
    last_cell = f"{column_letter(target_cols)}{target_rows}"

    worksheet.update(
        range_name=f"A1:{last_cell}",
        values=padded_payload,
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
        status = safe_order_status(order)
        if status in TERMINAL_ORDER_STATUSES:
            continue
        symbol = str(field(order, "symbol", "")).strip().upper()
        if symbol:
            symbols.add(symbol)

    return symbols


def market_is_open(trading_client: TradingClient) -> Tuple[bool, str]:
    clock = trading_client.get_clock()
    if bool(field(clock, "is_open", False)):
        return True, "market open"
    return False, "market closed"


def position_lifecycle_key(
    symbol: str,
    qty: Optional[Decimal],
    avg_entry_price: Optional[Decimal],
    cost_basis: Optional[Decimal],
) -> str:
    raw = f"{symbol}|{qty}|{avg_entry_price}|{cost_basis}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"seller-{symbol.lower()}-{digest}"[:48]


def time_in_force_from_config(value: str) -> TimeInForce:
    normalized = value.strip().lower()
    if normalized == "gtc":
        return TimeInForce.GTC
    return TimeInForce.DAY


def quantize_limit_price(price: Decimal) -> Decimal:
    if price >= Decimal("1"):
        return price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    return price.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)


def submit_market_sell(
    trading_client: TradingClient,
    symbol: str,
    qty: Decimal,
    client_order_id: str,
) -> Any:
    request = MarketOrderRequest(
        symbol=symbol,
        qty=str(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        type=OrderType.MARKET,
        client_order_id=client_order_id,
    )
    return trading_client.submit_order(order_data=request)


def submit_extended_limit_sell(
    config: Config,
    trading_client: TradingClient,
    symbol: str,
    qty: Decimal,
    limit_price: Decimal,
    client_order_id: str,
) -> Any:
    request = LimitOrderRequest(
        symbol=symbol,
        qty=str(qty),
        side=OrderSide.SELL,
        time_in_force=time_in_force_from_config(config.sell_extended_time_in_force),
        type=OrderType.LIMIT,
        limit_price=str(quantize_limit_price(limit_price)),
        extended_hours=True,
        client_order_id=client_order_id,
    )
    return trading_client.submit_order(order_data=request)


def get_latest_bid(data_client: StockHistoricalDataClient, symbol: str) -> Optional[Decimal]:
    request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    response = data_client.get_stock_latest_quote(request)

    quote = None
    if isinstance(response, dict):
        quote = response.get(symbol)
    else:
        try:
            quote = response[symbol]
        except Exception:
            quote = None

    bid = to_decimal(field(quote, "bid_price"))
    if bid is not None and bid > 0:
        return bid
    return None


def order_is_terminal(status: str) -> bool:
    return status in TERMINAL_ORDER_STATUSES


def wait_for_order_terminal_or_timeout(
    trading_client: TradingClient,
    order_id: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.time() + max(0, timeout_seconds)
    last_status = ""

    while True:
        try:
            order = trading_client.get_order_by_id(order_id)
            last_status = safe_order_status(order)
            if order_is_terminal(last_status):
                return last_status
        except Exception as exc:
            logging.warning("Could not poll order %s: %s", order_id, exc)

        if time.time() >= deadline:
            return last_status or "unknown"

        time.sleep(max(1, poll_seconds))


def cancel_order_safely(trading_client: TradingClient, order_id: str) -> None:
    try:
        trading_client.cancel_order_by_id(order_id)
    except Exception as exc:
        logging.warning("Could not cancel order %s: %s", order_id, exc)


def stepped_extended_sell(
    config: Config,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    symbol: str,
    qty: Decimal,
    current_price: Optional[Decimal],
    client_order_id_base: str,
) -> Tuple[str, str]:
    bid = get_latest_bid(data_client, symbol)
    if current_price is None or current_price <= 0:
        if bid is None:
            raise RuntimeError("No current price or latest bid available for extended-hours limit sell.")
        current_price = bid

    if bid is None or bid <= 0:
        bid = current_price

    anchor = max(current_price, bid)
    step_pct = max(Decimal("0"), min(config.sell_extended_step_pct, Decimal("100")))
    step_fraction = step_pct / Decimal("100")

    if anchor <= 0:
        raise RuntimeError("Invalid extended-hours price anchor.")

    # Avoid a zero second step loop.
    step_seconds = max(1, config.sell_extended_step_seconds)
    total_timeout = max(step_seconds, config.sell_extended_total_timeout_seconds)
    max_attempts = max(1, int(total_timeout / step_seconds))

    last_order_id = ""
    last_status = ""
    last_price = anchor

    for attempt in range(max_attempts):
        if attempt == 0:
            limit_price = anchor
        else:
            # Move a configured fraction of the original anchor-to-bid distance each step.
            distance = max(Decimal("0"), anchor - bid)
            progress = min(Decimal("1"), Decimal(attempt) * step_fraction)
            limit_price = anchor - (distance * progress)
            if limit_price < bid:
                limit_price = bid

        last_price = quantize_limit_price(limit_price)
        client_order_id = f"{client_order_id_base}-x{attempt}"[:48]

        order = submit_extended_limit_sell(
            config=config,
            trading_client=trading_client,
            symbol=symbol,
            qty=qty,
            limit_price=last_price,
            client_order_id=client_order_id,
        )

        last_order_id = safe_order_id(order)
        last_status = wait_for_order_terminal_or_timeout(
            trading_client=trading_client,
            order_id=last_order_id,
            timeout_seconds=step_seconds,
            poll_seconds=config.order_poll_interval_seconds,
        )

        if last_status == "filled":
            return last_order_id, "SELL_SUBMITTED"

        if attempt == max_attempts - 1 and config.sell_extended_leave_final_order:
            return last_order_id, "SELL_SUBMITTED"

        cancel_order_safely(trading_client, last_order_id)

        if limit_price <= bid:
            # We reached bid and did not fill. Do not keep hammering the same price.
            break

    return last_order_id, "SELL_SIGNAL_EXTENDED_UNFILLED"


def build_row(
    symbol: str,
    current_gain_pct: Optional[Decimal],
    peak_gain_pct: Optional[Decimal],
    armed: bool,
    drawdown_from_peak_pct: Optional[Decimal],
    last_action: str,
) -> List[Any]:
    return [
        symbol,
        decimal_to_sheet(current_gain_pct),
        decimal_to_sheet(peak_gain_pct),
        "TRUE" if armed else "FALSE",
        decimal_to_sheet(drawdown_from_peak_pct),
        last_action,
    ]


def merge_state(sheet_state: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    state = dict(sheet_state)
    state.update(STATE_CACHE)
    return state


def update_state_cache(rows: List[List[Any]]) -> None:
    global STATE_CACHE
    STATE_CACHE = {}
    for row in rows:
        record = row_to_record(row)
        symbol = record.get("symbol", "").strip().upper()
        if symbol:
            STATE_CACHE[symbol] = record


def run_cycle(
    config: Config,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    worksheet: Any,
) -> Dict[str, Any]:
    now = utc_now_iso()

    sheet_state = read_seller_state(worksheet)
    state = merge_state(sheet_state)

    positions = trading_client.get_all_positions()
    market_open, market_reason = market_is_open(trading_client)

    order_lookup_note = ""
    protected_sell_symbols: Set[str] = set()
    try:
        protected_sell_symbols = get_recent_sell_order_symbols(
            trading_client,
            config.sell_order_lookback_minutes,
        )
    except Exception as exc:
        # Do not block a trailing sell just because order lookup failed.
        order_lookup_note = f"Order lookup failed; continuing without lookup protection: {exc}"
        logging.warning(order_lookup_note)

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

        side_text = safe_side_text(field(pos, "side", "long"))
        is_long_position = "long" in side_text or side_text == ""

        has_protective_prior_action = last_action in PROTECTIVE_LAST_ACTIONS
        has_recent_sell_order = symbol in protected_sell_symbols

        if market_value is not None and market_value < config.sell_min_market_value:
            sell_signal = False
            last_action = "BELOW_MIN_VALUE"

        if not is_long_position:
            sell_signal = False
            last_action = "SKIPPED_NON_LONG"

        if sell_signal:
            if qty is None or qty <= 0:
                last_action = "SELL_SIGNAL_BLOCKED"
                blocked_signals.append({"symbol": symbol, "reason": "qty missing or <= 0"})
            elif has_protective_prior_action:
                blocked_signals.append({"symbol": symbol, "reason": "prior protective action"})
            elif has_recent_sell_order:
                last_action = "SELL_PROTECTED"
                blocked_signals.append({"symbol": symbol, "reason": "open/recent sell order"})
            elif config.sell_dry_run:
                last_action = "DRY_RUN_SELL_SIGNAL"
                dry_run_signals.append(symbol)
            elif market_open:
                client_order_id = position_lifecycle_key(symbol, qty, avg_entry_price, cost_basis)
                order = submit_market_sell(trading_client, symbol, qty, client_order_id)
                last_action = "SELL_SUBMITTED"
                submitted_sells.append({"symbol": symbol, "order_id": safe_order_id(order)})
            elif config.sell_extended_hours:
                client_order_id = position_lifecycle_key(symbol, qty, avg_entry_price, cost_basis)
                try:
                    order_id, last_action = stepped_extended_sell(
                        config=config,
                        trading_client=trading_client,
                        data_client=data_client,
                        symbol=symbol,
                        qty=qty,
                        current_price=current_price,
                        client_order_id_base=client_order_id,
                    )
                    submitted_sells.append({"symbol": symbol, "order_id": order_id})
                except Exception as exc:
                    last_action = "SELL_SIGNAL_EXTENDED_FAILED"
                    blocked_signals.append({"symbol": symbol, "reason": f"extended sell failed: {exc}"})
            else:
                last_action = "SELL_SIGNAL_MARKET_CLOSED"
                blocked_signals.append({"symbol": symbol, "reason": market_reason})
        else:
            if last_action not in {"BELOW_MIN_VALUE", "SKIPPED_NON_LONG"}:
                last_action = "ARMED" if armed else "MONITORING"

        rows.append(
            build_row(
                symbol=symbol,
                current_gain_pct=current_gain_pct,
                peak_gain_pct=peak_gain_pct,
                armed=armed,
                drawdown_from_peak_pct=drawdown_from_peak_pct,
                last_action=last_action,
            )
        )

    update_state_cache(rows)

    sheet_write_note = ""
    try:
        write_seller_rows(worksheet, rows)
    except Exception as exc:
        # Sheets 429s should not stop sell logic. The in-memory cache preserves state until the next successful write.
        sheet_write_note = f"Seller sheet write failed; trading loop continued: {exc}"
        logging.warning(sheet_write_note)

    result = {
        "state": "ok",
        "updated_at": now,
        "positions": len(rows),
        "dry_run": config.sell_dry_run,
        "market_open": market_open,
        "market_reason": market_reason,
        "extended_hours_enabled": config.sell_extended_hours,
        "extended_step_pct": str(config.sell_extended_step_pct),
        "extended_step_seconds": str(config.sell_extended_step_seconds),
        "protected_sell_symbols": sorted(protected_sell_symbols),
        "order_lookup_note": order_lookup_note,
        "submitted_sells": submitted_sells,
        "dry_run_signals": dry_run_signals,
        "blocked_signals": blocked_signals,
        "sheet_write_note": sheet_write_note,
        "visible_columns": HEADERS,
    }

    LAST_STATUS.clear()
    LAST_STATUS.update(result)

    return result


CONFIG = load_config()
TRADING_CLIENT, DATA_CLIENT = init_clients(CONFIG)
WORKSHEET = init_worksheet(CONFIG)

app = FastAPI(title="Alpaca Seller Bot")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": utc_now_iso(),
        "bot_auto_start": CONFIG.bot_auto_start,
        "seller_tab": CONFIG.seller_tab,
        "visible_columns": HEADERS,
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    return LAST_STATUS


@app.post("/run")
def run_once() -> Dict[str, Any]:
    return run_cycle(CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET)


async def background_loop() -> None:
    while True:
        try:
            result = await asyncio.to_thread(run_cycle, CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET)
            logging.info("Cycle complete: %s", result)
            await asyncio.sleep(max(0, CONFIG.poll_seconds))
        except Exception:
            logging.exception("Cycle failed")
            await asyncio.sleep(max(1, CONFIG.error_backoff_seconds))


@app.on_event("startup")
async def startup_event() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if CONFIG.bot_auto_start:
        asyncio.create_task(background_loop())
