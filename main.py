import asyncio
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

import gspread
from fastapi import FastAPI, Header, HTTPException
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest


# Seller tab visible columns. Only A:F are owned by this bot.
HEADERS = [
    "symbol",
    "gain_pct",
    "peak_gain_pct",
    "armed",
    "drop_from_peak_pct",
    "action",
]

TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "stopped",
    "done_for_day",
    "replaced",
}

STATE_HEADERS = [
    "symbol",
    "position_key",
    "position_qty",
    "avg_entry_price",
    "cost_basis",
    "peak_gain_pct",
    "armed",
    "last_action",
    "order_id",
    "client_order_id",
    "order_status",
    "updated_at",
]

LAST_STATUS: Dict[str, Any] = {"state": "starting"}
STATE_CACHE: Dict[str, Dict[str, str]] = {}
RUN_LOCK = threading.Lock()


@dataclass(frozen=True)
class Config:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool
    google_sheet_id: str
    google_service_account_json: str

    seller_tab: str
    seller_state_tab: str
    sell_arm_gain_pct: Decimal
    sell_trail_drop_pct: Decimal
    sell_disarm_below_gain_pct: Optional[Decimal]
    sell_min_market_value: Decimal
    sell_dry_run: bool

    sell_extended_hours: bool
    sell_extended_step_pct: Decimal
    sell_extended_step_seconds: int
    sell_extended_total_timeout_seconds: int
    sell_extended_leave_final_order: bool
    sell_extended_time_in_force: str
    order_poll_interval_seconds: int

    poll_seconds: int
    error_backoff_seconds: int
    bot_auto_start: bool
    bot_run_token: str


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
        result = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise RuntimeError(f"Invalid decimal env var {name}={value!r}") from exc
    if not result.is_finite():
        raise RuntimeError(f"Invalid decimal env var {name}={value!r}")
    return result


def env_optional_decimal(name: str, default: str) -> Optional[Decimal]:
    value = os.getenv(name, default).strip().lower()
    if value in {"", "none", "off", "disabled"}:
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeError(f"Invalid decimal env var {name}={value!r}") from exc
    if not result.is_finite():
        raise RuntimeError(f"Invalid decimal env var {name}={value!r}")
    return result


def validate_config(config: Config) -> Config:
    if config.sell_arm_gain_pct < 0:
        raise RuntimeError("SELL_ARM_GAIN_PCT must be >= 0")
    if config.sell_trail_drop_pct <= 0:
        raise RuntimeError("SELL_TRAIL_DROP_PCT must be > 0")
    if config.sell_min_market_value < 0:
        raise RuntimeError("SELL_MIN_MARKET_VALUE must be >= 0")
    if config.sell_extended_step_pct <= 0 or config.sell_extended_step_pct > 100:
        raise RuntimeError("SELL_EXTENDED_STEP_PCT must be > 0 and <= 100")
    if config.sell_extended_step_seconds < 1:
        raise RuntimeError("SELL_EXTENDED_STEP_SECONDS must be >= 1")
    if config.sell_extended_total_timeout_seconds < config.sell_extended_step_seconds:
        raise RuntimeError(
            "SELL_EXTENDED_TOTAL_TIMEOUT_SECONDS must be >= SELL_EXTENDED_STEP_SECONDS"
        )
    if config.order_poll_interval_seconds < 1:
        raise RuntimeError("ORDER_POLL_INTERVAL_SECONDS must be >= 1")
    if config.poll_seconds < 1:
        raise RuntimeError("POLL_SECONDS must be >= 1")
    if config.error_backoff_seconds < 1:
        raise RuntimeError("ERROR_BACKOFF_SECONDS must be >= 1")
    if config.sell_extended_time_in_force not in {"day", "gtc"}:
        raise RuntimeError("SELL_EXTENDED_TIME_IN_FORCE must be 'day' or 'gtc'")
    return config


def load_config() -> Config:
    config = Config(
        alpaca_api_key=env_required("ALPACA_API_KEY"),
        alpaca_secret_key=env_required("ALPACA_SECRET_KEY"),
        alpaca_paper=env_bool("ALPACA_PAPER", True),
        google_sheet_id=env_required("GOOGLE_SHEET_ID"),
        google_service_account_json=env_required("GOOGLE_SERVICE_ACCOUNT_JSON"),

        seller_tab=os.getenv("SELLER_TAB", "Seller").strip() or "Seller",
        seller_state_tab=os.getenv("SELLER_STATE_TAB", "SellerState").strip() or "SellerState",
        sell_arm_gain_pct=env_decimal("SELL_ARM_GAIN_PCT", "12"),
        sell_trail_drop_pct=env_decimal("SELL_TRAIL_DROP_PCT", "4"),
        sell_disarm_below_gain_pct=env_optional_decimal("SELL_DISARM_BELOW_GAIN_PCT", "0"),
        sell_min_market_value=env_decimal("SELL_MIN_MARKET_VALUE", "0"),
        sell_dry_run=env_bool("SELL_DRY_RUN", True),

        sell_extended_hours=env_bool("SELL_EXTENDED_HOURS", False),
        sell_extended_step_pct=env_decimal("SELL_EXTENDED_STEP_PCT", "25"),
        sell_extended_step_seconds=env_int("SELL_EXTENDED_STEP_SECONDS", 5),
        sell_extended_total_timeout_seconds=env_int("SELL_EXTENDED_TOTAL_TIMEOUT_SECONDS", 60),
        sell_extended_leave_final_order=env_bool("SELL_EXTENDED_LEAVE_FINAL_ORDER", False),
        sell_extended_time_in_force=os.getenv("SELL_EXTENDED_TIME_IN_FORCE", "day").strip().lower() or "day",
        order_poll_interval_seconds=env_int("ORDER_POLL_INTERVAL_SECONDS", 1),

        poll_seconds=env_int("POLL_SECONDS", 10),
        error_backoff_seconds=env_int("ERROR_BACKOFF_SECONDS", 90),
        bot_auto_start=env_bool("BOT_AUTO_START", True),
        bot_run_token=os.getenv("BOT_RUN_TOKEN", "").strip(),
    )
    return validate_config(config)


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
        result = Decimal(text)
        return result if result.is_finite() else None
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


def get_or_create_worksheet(spreadsheet: Any, title: str, cols: int) -> Any:
    try:
        return spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=20, cols=cols)


def init_worksheets(config: Config) -> Tuple[Any, Any]:
    creds = load_google_credentials(config.google_service_account_json)
    gc = gspread.service_account_from_dict(creds)
    spreadsheet = gc.open_by_key(config.google_sheet_id)
    seller_worksheet = get_or_create_worksheet(spreadsheet, config.seller_tab, len(HEADERS))
    state_worksheet = get_or_create_worksheet(
        spreadsheet, config.seller_state_tab, len(STATE_HEADERS)
    )
    try:
        state_worksheet.hide()
    except Exception as exc:
        logging.warning("Could not hide %s tab: %s", config.seller_state_tab, exc)
    return seller_worksheet, state_worksheet


def column_letter(col: int) -> str:
    result = ""
    while col:
        col, rem = divmod(col - 1, 26)
        result = chr(65 + rem) + result
    return result


def read_rows_by_symbol(worksheet: Any) -> Dict[str, Dict[str, str]]:
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


def write_rows(worksheet: Any, headers: List[str], rows: List[List[Any]]) -> None:
    payload = [headers] + rows
    target_rows = max(20, len(payload) + 5, worksheet.row_count)
    target_cols = len(headers)

    try:
        if worksheet.row_count < target_rows or worksheet.col_count < target_cols:
            worksheet.resize(
                rows=max(worksheet.row_count, target_rows),
                cols=max(worksheet.col_count, target_cols),
            )
    except Exception as exc:
        logging.warning("Could not expand worksheet; continuing with update: %s", exc)

    blank_row = [""] * target_cols
    padded_payload = payload + [blank_row[:] for _ in range(target_rows - len(payload))]
    last_cell = f"{column_letter(target_cols)}{target_rows}"

    worksheet.update(
        range_name=f"A1:{last_cell}",
        values=padded_payload,
        value_input_option="USER_ENTERED",
    )


def write_seller_rows(worksheet: Any, rows: List[List[Any]]) -> None:
    # Update only A:F. Never resize away or clear user-owned columns to the right.
    write_rows(worksheet, HEADERS, rows)


def state_record_to_row(record: Dict[str, str]) -> List[str]:
    return [record.get(header, "") for header in STATE_HEADERS]


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
    return f"seller-{symbol.lower()}-{digest}"[:36]


def is_not_found_error(exc: Exception) -> bool:
    status_code = field(exc, "status_code")
    return status_code == 404 or "404" in str(exc) or "not found" in str(exc).lower()


def get_order_by_client_id(trading_client: TradingClient, client_order_id: str) -> Any:
    try:
        return trading_client.get_order_by_client_id(client_order_id)
    except Exception as exc:
        if is_not_found_error(exc):
            return None
        raise


def allocate_client_order_id(
    trading_client: TradingClient,
    base: str,
    order_kind: str,
) -> Tuple[str, Any]:
    """Return an unused ID, or an existing active order for crash recovery."""
    for attempt in range(100):
        suffix = f"-{order_kind}{attempt}"
        candidate = f"{base[:48 - len(suffix)]}{suffix}"
        existing = get_order_by_client_id(trading_client, candidate)
        if existing is None:
            return candidate, None
        if order_may_be_open(safe_order_status(existing)):
            return candidate, existing
    raise RuntimeError(f"No client-order ID available for {base}")


def get_open_sell_orders(trading_client: TradingClient) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    for order in trading_client.get_orders():
        if safe_side_text(field(order, "side")) != "sell":
            continue
        symbol = str(field(order, "symbol", "")).strip().upper()
        if symbol and order_may_be_open(safe_order_status(order)):
            result.setdefault(symbol, []).append(order)
    return result


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


def order_may_be_open(status: str) -> bool:
    # Unknown broker statuses fail closed: never overlap a possibly live sell order.
    return status not in TERMINAL_ORDER_STATUSES


def wait_for_order_terminal_or_timeout(
    trading_client: TradingClient,
    order_id: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> Any:
    deadline = time.time() + max(0, timeout_seconds)
    last_order = None

    while True:
        try:
            order = trading_client.get_order_by_id(order_id)
            last_order = order
            if order_is_terminal(safe_order_status(order)):
                return order
        except Exception as exc:
            logging.warning("Could not poll order %s: %s", order_id, exc)

        if time.time() >= deadline:
            return last_order

        time.sleep(max(1, poll_seconds))


def cancel_and_confirm(
    trading_client: TradingClient,
    order_id: str,
    poll_seconds: int,
    timeout_seconds: int = 15,
) -> Any:
    try:
        trading_client.cancel_order_by_id(order_id)
    except Exception as exc:
        order = trading_client.get_order_by_id(order_id)
        if safe_order_status(order) not in TERMINAL_ORDER_STATUSES:
            raise RuntimeError(f"Could not cancel order {order_id}: {exc}") from exc
        return order
    return wait_for_order_terminal_or_timeout(
        trading_client, order_id, timeout_seconds, poll_seconds
    )


def current_position_qty(trading_client: TradingClient, symbol: str) -> Decimal:
    try:
        position = trading_client.get_open_position(symbol)
    except Exception as exc:
        if is_not_found_error(exc):
            return Decimal("0")
        raise
    qty = to_decimal(field(position, "qty"))
    if qty is None:
        raise RuntimeError(f"Alpaca returned no position quantity for {symbol}")
    return max(Decimal("0"), qty)


@dataclass(frozen=True)
class OrderOutcome:
    action: str
    order_id: str = ""
    client_order_id: str = ""
    status: str = ""
    submitted: bool = False
    open_order: bool = False


def outcome_from_order(action: str, order: Any, submitted: bool = False) -> OrderOutcome:
    status = safe_order_status(order)
    return OrderOutcome(
        action=action,
        order_id=safe_order_id(order),
        client_order_id=safe_client_order_id(order),
        status=status,
        submitted=submitted,
        open_order=order_may_be_open(status),
    )


def submit_market_sell_reconciled(
    trading_client: TradingClient,
    symbol: str,
    qty: Decimal,
    client_order_id_base: str,
) -> OrderOutcome:
    client_order_id, existing = allocate_client_order_id(
        trading_client, client_order_id_base, "m"
    )
    if existing is not None:
        return outcome_from_order("SELL_ORDER_OPEN", existing)
    order = submit_market_sell(trading_client, symbol, qty, client_order_id)
    status = safe_order_status(order)
    if status in {"rejected", "canceled", "cancelled", "expired"}:
        return outcome_from_order("SELL_SUBMIT_FAILED", order, submitted=True)
    if status == "filled":
        return outcome_from_order("SELL_FILLED", order, submitted=True)
    return outcome_from_order("SELL_SUBMITTED", order, submitted=True)


def stepped_extended_sell(
    config: Config,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    symbol: str,
    current_price: Optional[Decimal],
    client_order_id_base: str,
) -> OrderOutcome:
    bid = get_latest_bid(data_client, symbol)
    if current_price is None or current_price <= 0:
        if bid is None:
            raise RuntimeError("No current price or latest bid available for extended-hours limit sell.")
        current_price = bid

    if bid is None or bid <= 0:
        bid = current_price

    anchor = max(current_price, bid)
    step_fraction = config.sell_extended_step_pct / Decimal("100")

    if anchor <= 0:
        raise RuntimeError("Invalid extended-hours price anchor.")

    step_seconds = config.sell_extended_step_seconds
    deadline = time.time() + config.sell_extended_total_timeout_seconds
    attempt = 0
    last_outcome = OrderOutcome(action="SELL_SIGNAL_EXTENDED_UNFILLED")

    while time.time() < deadline:
        remaining_qty = current_position_qty(trading_client, symbol)
        if remaining_qty <= 0:
            return OrderOutcome(action="SELL_FILLED", status="filled")

        latest_bid = get_latest_bid(data_client, symbol) or bid
        distance = max(Decimal("0"), anchor - latest_bid)
        progress = min(Decimal("1"), Decimal(attempt) * step_fraction)
        limit_price = max(latest_bid, anchor - (distance * progress))
        at_bid = limit_price <= latest_bid

        client_order_id, existing = allocate_client_order_id(
            trading_client, client_order_id_base, f"e{attempt}"
        )
        if existing is None:
            order = submit_extended_limit_sell(
                config=config,
                trading_client=trading_client,
                symbol=symbol,
                qty=remaining_qty,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
            submitted = True
        else:
            order = existing
            submitted = False

        status = safe_order_status(order)
        if status in {"rejected", "stopped"}:
            return outcome_from_order("SELL_SIGNAL_EXTENDED_FAILED", order, submitted)

        remaining_time = max(0, int(deadline - time.time()))
        wait_seconds = remaining_time if at_bid else min(step_seconds, remaining_time)
        polled_order = wait_for_order_terminal_or_timeout(
            trading_client,
            safe_order_id(order),
            wait_seconds,
            config.order_poll_interval_seconds,
        ) or order
        status = safe_order_status(polled_order)
        if status in {"rejected", "stopped"}:
            return outcome_from_order("SELL_SIGNAL_EXTENDED_FAILED", polled_order, submitted)
        last_outcome = outcome_from_order(
            "SELL_ORDER_OPEN" if order_may_be_open(status) else "SELL_SIGNAL_EXTENDED_UNFILLED",
            polled_order,
            submitted,
        )

        if status == "filled" and current_position_qty(trading_client, symbol) <= 0:
            return outcome_from_order("SELL_FILLED", polled_order, submitted)

        if order_may_be_open(status):
            if time.time() >= deadline and config.sell_extended_leave_final_order:
                return outcome_from_order("SELL_ORDER_OPEN", polled_order, submitted)
            canceled_order = cancel_and_confirm(
                trading_client,
                safe_order_id(polled_order),
                config.order_poll_interval_seconds,
            )
            canceled_status = safe_order_status(canceled_order)
            if order_may_be_open(canceled_status):
                return outcome_from_order("SELL_CANCEL_UNCONFIRMED", canceled_order or polled_order)
            if canceled_status == "filled" and current_position_qty(trading_client, symbol) <= 0:
                return outcome_from_order("SELL_FILLED", canceled_order)
            last_outcome = outcome_from_order(
                "SELL_SIGNAL_EXTENDED_UNFILLED", canceled_order, submitted
            )

        attempt += 1

    return last_outcome


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


def merge_state(
    state_sheet: Dict[str, Dict[str, str]],
    legacy_sheet: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    state = dict(legacy_sheet)
    state.update(state_sheet)
    state.update(STATE_CACHE)
    return state


def update_state_cache(records: List[Dict[str, str]]) -> None:
    global STATE_CACHE
    new_cache: Dict[str, Dict[str, str]] = {}
    for record in records:
        symbol = record.get("symbol", "").strip().upper()
        if symbol:
            new_cache[symbol] = dict(record)
    STATE_CACHE = new_cache


def order_action(order: Any) -> str:
    if safe_order_status(order) == "partially_filled":
        return "SELL_PARTIALLY_FILLED"
    return "SELL_ORDER_OPEN"


def run_cycle(
    config: Config,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    worksheet: Any,
    state_worksheet: Any,
) -> Dict[str, Any]:
    with RUN_LOCK:
        return _run_cycle(config, trading_client, data_client, worksheet, state_worksheet)


def _run_cycle(
    config: Config,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    worksheet: Any,
    state_worksheet: Any,
) -> Dict[str, Any]:
    now = utc_now_iso()

    sheet_notes: List[str] = []
    try:
        durable_state = read_rows_by_symbol(state_worksheet)
    except Exception as exc:
        durable_state = {}
        sheet_notes.append(f"State sheet read failed; using memory cache: {exc}")

    try:
        legacy_state = read_rows_by_symbol(worksheet)
    except Exception as exc:
        legacy_state = {}
        sheet_notes.append(f"Seller sheet read failed; using durable state: {exc}")

    state = merge_state(durable_state, legacy_state)

    positions = trading_client.get_all_positions()
    try:
        market_open, market_reason = market_is_open(trading_client)
    except Exception as exc:
        market_open, market_reason = False, f"market clock unavailable: {exc}"

    reconciliation_error = ""
    try:
        open_sell_orders = get_open_sell_orders(trading_client)
    except Exception as exc:
        open_sell_orders = {}
        reconciliation_error = f"could not reconcile open orders: {exc}"

    rows: List[List[Any]] = []
    state_records: List[Dict[str, str]] = []
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
        position_key = position_lifecycle_key(symbol, qty, avg_entry_price, cost_basis)

        prior_position_key = prior.get("position_key", "")
        prior_qty = to_decimal(prior.get("position_qty"))
        prior_avg_entry_price = to_decimal(prior.get("avg_entry_price"))
        position_reduced = bool(
            prior_qty is not None
            and qty is not None
            and qty < prior_qty
            and prior_avg_entry_price == avg_entry_price
        )
        same_lifecycle = bool(
            not prior_position_key
            or prior_position_key == position_key
            or position_reduced
        )
        prior_peak = to_decimal(prior.get("peak_gain_pct")) if same_lifecycle else None

        if current_gain_pct is None:
            peak_gain_pct = prior_peak
        elif prior_peak is None:
            peak_gain_pct = current_gain_pct
        else:
            peak_gain_pct = max(prior_peak, current_gain_pct)

        symbol_open_orders = open_sell_orders.get(symbol, [])
        reset_below_floor = bool(
            not symbol_open_orders
            and config.sell_disarm_below_gain_pct is not None
            and current_gain_pct is not None
            and current_gain_pct < config.sell_disarm_below_gain_pct
        )
        if reset_below_floor:
            peak_gain_pct = current_gain_pct

        # Peak is the source of truth. This repairs legacy rows whose armed flag and peak disagree.
        armed = bool(
            not reset_below_floor
            and peak_gain_pct is not None
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

        side_text = safe_side_text(field(pos, "side", "long"))
        is_long_position = side_text in {"", "long"}
        last_action = "RESET_BELOW_FLOOR" if reset_below_floor else ("ARMED" if armed else "MONITORING")
        order_id = prior.get("order_id", "") if same_lifecycle else ""
        client_order_id = prior.get("client_order_id", "") if same_lifecycle else ""
        order_status = prior.get("order_status", "") if same_lifecycle else ""

        if config.sell_min_market_value > 0 and market_value is None:
            sell_signal = False
            last_action = "VALUE_UNAVAILABLE"
        elif market_value is not None and market_value < config.sell_min_market_value:
            sell_signal = False
            last_action = "BELOW_MIN_VALUE"

        if not is_long_position:
            sell_signal = False
            last_action = "SKIPPED_NON_LONG"

        if symbol_open_orders:
            tracked_order = symbol_open_orders[0]
            order_id = safe_order_id(tracked_order)
            client_order_id = safe_client_order_id(tracked_order)
            order_status = safe_order_status(tracked_order)
            if len(symbol_open_orders) > 1:
                last_action = "MULTIPLE_OPEN_SELL_ORDERS"
                blocked_signals.append(
                    {"symbol": symbol, "reason": f"{len(symbol_open_orders)} open sell orders"}
                )
            else:
                last_action = order_action(tracked_order)
        elif sell_signal:
            if qty is None or qty <= 0:
                last_action = "SELL_SIGNAL_BLOCKED"
                blocked_signals.append({"symbol": symbol, "reason": "qty missing or <= 0"})
            elif reconciliation_error:
                last_action = "SELL_SIGNAL_BLOCKED"
                blocked_signals.append({"symbol": symbol, "reason": reconciliation_error})
            elif config.sell_dry_run:
                last_action = "DRY_RUN_SELL_SIGNAL"
                dry_run_signals.append(symbol)
            elif market_open:
                try:
                    outcome = submit_market_sell_reconciled(
                        trading_client, symbol, qty, position_key
                    )
                    last_action = outcome.action
                    order_id = outcome.order_id
                    client_order_id = outcome.client_order_id
                    order_status = outcome.status
                    if outcome.submitted:
                        submitted_sells.append(
                            {
                                "symbol": symbol,
                                "order_id": outcome.order_id,
                                "status": outcome.status,
                            }
                        )
                    if outcome.action == "SELL_SUBMIT_FAILED":
                        blocked_signals.append(
                            {"symbol": symbol, "reason": f"market order {outcome.status}"}
                        )
                except Exception as exc:
                    last_action = "SELL_SUBMIT_FAILED"
                    blocked_signals.append(
                        {"symbol": symbol, "reason": f"market sell failed: {exc}"}
                    )
            elif config.sell_extended_hours:
                try:
                    outcome = stepped_extended_sell(
                        config=config,
                        trading_client=trading_client,
                        data_client=data_client,
                        symbol=symbol,
                        current_price=current_price,
                        client_order_id_base=position_key,
                    )
                    last_action = outcome.action
                    order_id = outcome.order_id
                    client_order_id = outcome.client_order_id
                    order_status = outcome.status
                    if outcome.submitted:
                        submitted_sells.append(
                            {
                                "symbol": symbol,
                                "order_id": outcome.order_id,
                                "status": outcome.status,
                            }
                        )
                    if outcome.action in {
                        "SELL_CANCEL_UNCONFIRMED",
                        "SELL_SIGNAL_EXTENDED_FAILED",
                        "SELL_SIGNAL_EXTENDED_UNFILLED",
                    }:
                        blocked_signals.append(
                            {"symbol": symbol, "reason": outcome.action.lower()}
                        )
                except Exception as exc:
                    last_action = "SELL_SIGNAL_EXTENDED_FAILED"
                    blocked_signals.append({"symbol": symbol, "reason": f"extended sell failed: {exc}"})
            else:
                last_action = "SELL_SIGNAL_MARKET_CLOSED"
                blocked_signals.append({"symbol": symbol, "reason": market_reason})
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

        state_records.append(
            {
                "symbol": symbol,
                "position_key": position_key,
                "position_qty": decimal_to_sheet(qty, "0.000000001"),
                "avg_entry_price": decimal_to_sheet(avg_entry_price, "0.000000001"),
                "cost_basis": decimal_to_sheet(cost_basis, "0.000000001"),
                "peak_gain_pct": decimal_to_sheet(peak_gain_pct),
                "armed": "TRUE" if armed else "FALSE",
                "last_action": last_action,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "order_status": order_status,
                "updated_at": now,
            }
        )

    update_state_cache(state_records)

    try:
        write_seller_rows(worksheet, rows)
    except Exception as exc:
        sheet_notes.append(f"Seller sheet write failed; trading loop continued: {exc}")

    try:
        write_rows(
            state_worksheet,
            STATE_HEADERS,
            [state_record_to_row(record) for record in state_records],
        )
    except Exception as exc:
        sheet_notes.append(f"State sheet write failed; memory cache retained: {exc}")

    for note in sheet_notes:
        logging.warning(note)

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
        "disarm_below_gain_pct": (
            str(config.sell_disarm_below_gain_pct)
            if config.sell_disarm_below_gain_pct is not None
            else None
        ),
        "submitted_sells": submitted_sells,
        "dry_run_signals": dry_run_signals,
        "blocked_signals": blocked_signals,
        "reconciliation_error": reconciliation_error,
        "sheet_notes": sheet_notes,
        "visible_columns": HEADERS,
    }

    LAST_STATUS.clear()
    LAST_STATUS.update(result)

    return result


CONFIG: Optional[Config] = None
TRADING_CLIENT: Optional[TradingClient] = None
DATA_CLIENT: Optional[StockHistoricalDataClient] = None
WORKSHEET: Any = None
STATE_WORKSHEET: Any = None
BACKGROUND_TASK: Optional[asyncio.Task] = None

app = FastAPI(title="Alpaca Seller Bot")


@app.get("/health")
def health() -> Dict[str, Any]:
    ready = all(
        value is not None
        for value in (CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET, STATE_WORKSHEET)
    )
    return {
        "ok": ready and LAST_STATUS.get("state") != "error",
        "ready": ready,
        "time": utc_now_iso(),
        "bot_auto_start": CONFIG.bot_auto_start if CONFIG else None,
        "seller_tab": CONFIG.seller_tab if CONFIG else None,
        "last_state": LAST_STATUS.get("state"),
        "last_updated_at": LAST_STATUS.get("updated_at"),
        "visible_columns": HEADERS,
    }


@app.get("/status")
def status(x_bot_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if CONFIG and CONFIG.bot_run_token:
        if not x_bot_token or not secrets.compare_digest(x_bot_token, CONFIG.bot_run_token):
            raise HTTPException(status_code=401, detail="Invalid bot token")
    return LAST_STATUS


@app.post("/run")
def run_once(x_bot_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not all((CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET, STATE_WORKSHEET)):
        raise HTTPException(status_code=503, detail="Bot runtime is not ready")
    if not CONFIG.bot_run_token:
        raise HTTPException(
            status_code=503,
            detail="Manual runs are disabled until BOT_RUN_TOKEN is configured",
        )
    if not x_bot_token or not secrets.compare_digest(x_bot_token, CONFIG.bot_run_token):
        raise HTTPException(status_code=401, detail="Invalid bot token")
    return run_cycle(CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET, STATE_WORKSHEET)


async def background_loop() -> None:
    while True:
        try:
            if not all((CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET, STATE_WORKSHEET)):
                raise RuntimeError("Bot runtime is not ready")
            result = await asyncio.to_thread(
                run_cycle,
                CONFIG,
                TRADING_CLIENT,
                DATA_CLIENT,
                WORKSHEET,
                STATE_WORKSHEET,
            )
            logging.info("Cycle complete: %s", result)
            await asyncio.sleep(CONFIG.poll_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.exception("Cycle failed")
            LAST_STATUS.clear()
            LAST_STATUS.update(
                {"state": "error", "updated_at": utc_now_iso(), "error": str(exc)}
            )
            await asyncio.sleep(CONFIG.error_backoff_seconds if CONFIG else 90)


@app.on_event("startup")
async def startup_event() -> None:
    global CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET, STATE_WORKSHEET, BACKGROUND_TASK
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        CONFIG = load_config()
        TRADING_CLIENT, DATA_CLIENT = init_clients(CONFIG)
        WORKSHEET, STATE_WORKSHEET = init_worksheets(CONFIG)
    except Exception as exc:
        logging.exception("Bot initialization failed")
        LAST_STATUS.clear()
        LAST_STATUS.update(
            {"state": "error", "updated_at": utc_now_iso(), "error": str(exc)}
        )
        return

    if CONFIG.bot_auto_start:
        BACKGROUND_TASK = asyncio.create_task(background_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if BACKGROUND_TASK:
        BACKGROUND_TASK.cancel()
        try:
            await BACKGROUND_TASK
        except asyncio.CancelledError:
            pass
