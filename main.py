import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Set, Tuple

import gspread
from fastapi import FastAPI
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


HEADERS = [
    "symbol",
    "qty",
    "avg_entry_price",
    "current_price",
    "market_value",
    "cost_basis",
    "current_gain_pct",
    "peak_gain_pct",
    "armed",
    "drawdown_from_peak_pct",
    "sell_signal",
    "first_seen_at",
    "last_action",
    "last_order_id",
    "exit_client_order_id",
    "last_seen_at",
    "updated_at",
    "notes",
]

# Order statuses that imply a previous exit order is no longer active and can be retried.
RETRYABLE_ORDER_STATUSES = {"rejected", "canceled", "expired", "failed"}
TERMINAL_ORDER_STATUSES = {"filled", "canceled", "rejected", "expired", "failed"}
ACTIVE_OR_UNKNOWN_LAST_ACTIONS = {"SELL_SUBMITTED", "SELL_PROTECTED"}
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

    # Regular-hours exits are market sells. Extended-hours exits must be limit sells.
    sell_extended_hours: bool
    sell_extended_step_pct: Decimal
    sell_extended_step_seconds: Decimal
    sell_extended_total_timeout_seconds: Decimal
    sell_extended_time_in_force: str
    sell_extended_leave_final_order: bool
    order_poll_interval_seconds: Decimal

    # Optional duplicate visibility. Failure to look up orders never blocks a fresh sell trigger.
    sell_order_lookback_minutes: int

    # 0 means run the next cycle immediately after the prior cycle finishes.
    poll_seconds: Decimal
    error_backoff_seconds: Decimal
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
    tif = os.getenv("SELL_EXTENDED_TIME_IN_FORCE", "day").strip().lower() or "day"
    if tif not in {"day", "gtc"}:
        raise RuntimeError("SELL_EXTENDED_TIME_IN_FORCE must be either 'day' or 'gtc'")

    step_pct = env_decimal("SELL_EXTENDED_STEP_PCT", "25")
    if step_pct <= 0 or step_pct > 100:
        raise RuntimeError("SELL_EXTENDED_STEP_PCT must be > 0 and <= 100")

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
        sell_extended_step_pct=step_pct,
        sell_extended_step_seconds=env_decimal("SELL_EXTENDED_STEP_SECONDS", "5"),
        sell_extended_total_timeout_seconds=env_decimal("SELL_EXTENDED_TOTAL_TIMEOUT_SECONDS", "30"),
        sell_extended_time_in_force=tif,
        sell_extended_leave_final_order=env_bool("SELL_EXTENDED_LEAVE_FINAL_ORDER", True),
        order_poll_interval_seconds=env_decimal("ORDER_POLL_INTERVAL_SECONDS", "1"),
        sell_order_lookback_minutes=env_int("SELL_ORDER_LOOKBACK_MINUTES", 60),
        poll_seconds=env_decimal("POLL_SECONDS", "0"),
        error_backoff_seconds=env_decimal("ERROR_BACKOFF_SECONDS", "5"),
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


def price_to_order_decimal(value: Decimal) -> Decimal:
    # US equities >= $1 generally need penny precision; sub-dollar names can use four decimals.
    places = Decimal("0.01") if value >= Decimal("1") else Decimal("0.0001")
    return value.quantize(places, rounding=ROUND_HALF_UP)


def decimal_for_order(value: Decimal) -> str:
    return str(price_to_order_decimal(value))


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


def init_data_client(config: Config) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        config.alpaca_api_key,
        config.alpaca_secret_key,
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
) -> Tuple[Set[str], str]:
    """
    Best-effort duplicate visibility only.

    This intentionally does NOT block fresh sell triggers if order lookup fails. The actual
    duplicate protection comes from sheet state + Alpaca client_order_id.
    """
    after = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        side=OrderSide.SELL,
        after=after,
        limit=500,
    )

    try:
        orders = trading_client.get_orders(filter=request)
    except Exception as exc:
        return set(), f"Recent sell-order lookup failed but did not block exits: {exc}"

    symbols: Set[str] = set()
    for order in orders:
        symbol = str(field(order, "symbol", "")).strip().upper()
        if symbol:
            symbols.add(symbol)

    return symbols, ""


def prior_sell_submission_still_blocks(
    trading_client: TradingClient,
    last_action: str,
    last_order_id: str,
) -> Tuple[bool, str]:
    """
    Prevent duplicate exits after a sell was already submitted.

    This only applies after a previous exit has been recorded on this live row. It does not
    prevent a fresh armed/trail-drop signal from firing.
    """
    if last_action not in ACTIVE_OR_UNKNOWN_LAST_ACTIONS:
        return False, ""

    if not last_order_id:
        return True, "Prior sell action exists but no order id was recorded; skipped duplicate."

    try:
        order = trading_client.get_order_by_id(last_order_id)
    except Exception as exc:
        return True, f"Prior sell order status unavailable; skipped duplicate, not a fresh trigger block: {exc}"

    status = str(field(order, "status", "")).strip().lower()
    if status in RETRYABLE_ORDER_STATUSES:
        return False, f"Prior sell order status was {status}; retry allowed."

    return True, f"Prior sell order status is {status or 'unknown'}; skipped duplicate sell."


def market_is_open(trading_client: TradingClient) -> Tuple[bool, str]:
    clock = trading_client.get_clock()
    if bool(field(clock, "is_open", False)):
        return True, "market open"
    return False, "market closed"


def time_in_force_from_config(value: str) -> TimeInForce:
    if value == "gtc":
        return TimeInForce.GTC
    return TimeInForce.DAY


def make_exit_client_order_id(symbol: str, first_seen_at: str, qty: Optional[Decimal], avg_entry_price: Optional[Decimal]) -> str:
    raw = f"{symbol}|{first_seen_at}|{qty or ''}|{avg_entry_price or ''}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    # Alpaca client_order_id max is sufficient for this compact format.
    return f"seller-{symbol.lower()}-{digest}"


def make_step_client_order_id(base_client_order_id: str, step_index: int) -> str:
    return f"{base_client_order_id}-xh{step_index}"


def get_latest_quote_bid_ask(
    data_client: StockHistoricalDataClient,
    symbol: str,
) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    try:
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = data_client.get_stock_latest_quote(request)
    except Exception as exc:
        return None, None, f"Latest quote lookup failed: {exc}"

    quote = None
    if isinstance(quotes, dict):
        quote = quotes.get(symbol) or quotes.get(symbol.upper()) or quotes.get(symbol.lower())
    else:
        quote = field(quotes, symbol) or field(quotes, symbol.upper()) or field(quotes, symbol.lower())

    if quote is None:
        return None, None, "Latest quote lookup returned no quote."

    bid_price = (
        to_decimal(field(quote, "bid_price"))
        or to_decimal(field(quote, "bp"))
        or to_decimal(field(quote, "bid"))
    )
    ask_price = (
        to_decimal(field(quote, "ask_price"))
        or to_decimal(field(quote, "ap"))
        or to_decimal(field(quote, "ask"))
    )

    note = f"Latest quote bid={bid_price or ''} ask={ask_price or ''}."
    return bid_price, ask_price, note


def stepped_sell_limit_price(
    start_price: Decimal,
    bid_price: Optional[Decimal],
    step_pct: Decimal,
    step_index: int,
) -> Decimal:
    """
    Extended-hours sell chase:
    - step 0 starts at the market/current anchor
    - later steps move down toward the latest bid

    Example with start=10.00, bid=9.80, SELL_EXTENDED_STEP_PCT=25:
    10.00, 9.95, 9.90, 9.85, 9.80
    """
    if bid_price is None or bid_price <= 0:
        return start_price

    if start_price <= bid_price:
        return bid_price

    fraction = min(Decimal("1"), (step_pct * Decimal(step_index)) / Decimal("100"))
    return start_price - ((start_price - bid_price) * fraction)


def order_is_terminal(trading_client: TradingClient, order_id: str) -> Tuple[bool, str]:
    try:
        order = trading_client.get_order_by_id(order_id)
    except Exception as exc:
        return False, f"order status lookup failed: {exc}"

    status = str(field(order, "status", "")).strip().lower()
    return status in TERMINAL_ORDER_STATUSES, status


def wait_for_order_step(
    trading_client: TradingClient,
    order_id: str,
    seconds: Decimal,
    poll_interval_seconds: Decimal,
) -> Tuple[bool, str]:
    deadline = time.time() + max(0.0, float(seconds))
    interval = max(0.1, float(poll_interval_seconds))

    while time.time() < deadline:
        terminal, status = order_is_terminal(trading_client, order_id)
        if terminal:
            return True, status
        time.sleep(min(interval, max(0.0, deadline - time.time())))

    terminal, status = order_is_terminal(trading_client, order_id)
    return terminal, status


def cancel_order_best_effort(trading_client: TradingClient, order_id: str) -> str:
    try:
        trading_client.cancel_order_by_id(order_id)
        return "cancel requested"
    except Exception as exc:
        return f"cancel failed or unnecessary: {exc}"


def submit_regular_market_sell(
    trading_client: TradingClient,
    symbol: str,
    qty: Decimal,
    client_order_id: str,
) -> Any:
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=float(qty),
        side=OrderSide.SELL,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )
    return trading_client.submit_order(order_data=order_request)


def submit_extended_hours_limit_sell_once(
    trading_client: TradingClient,
    config: Config,
    symbol: str,
    qty: Decimal,
    limit_price: Decimal,
    client_order_id: str,
) -> Any:
    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=float(qty),
        side=OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=time_in_force_from_config(config.sell_extended_time_in_force),
        limit_price=float(price_to_order_decimal(limit_price)),
        extended_hours=True,
        client_order_id=client_order_id,
    )
    return trading_client.submit_order(order_data=order_request)


def submit_extended_hours_stepped_sell(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    config: Config,
    symbol: str,
    qty: Decimal,
    current_price: Decimal,
    base_client_order_id: str,
) -> Tuple[str, str, str]:
    """
    Submit a stepped extended-hours limit sell for the full position.

    Returns (last_order_id, last_limit_price, note).
    """
    bid_price, ask_price, quote_note = get_latest_quote_bid_ask(data_client, symbol)

    # "Market" anchor for extended hours. A true market order is not allowed, so step 0 is a
    # sell limit at the current/last market price, then steps walk down toward the bid.
    start_price = current_price
    if ask_price is not None and ask_price > 0:
        # If the quote gives us a better visible selling target than position.current_price,
        # start there rather than immediately conceding to the bid.
        start_price = max(start_price, ask_price)
    if bid_price is not None and bid_price > start_price:
        start_price = bid_price

    total_deadline = time.time() + max(0.0, float(config.sell_extended_total_timeout_seconds))
    step_index = 0
    last_order_id = ""
    last_limit = ""
    notes: List[str] = [quote_note]

    while True:
        if time.time() >= total_deadline and step_index > 0:
            notes.append("Extended chase total timeout reached.")
            break

        limit_price = stepped_sell_limit_price(
            start_price=start_price,
            bid_price=bid_price,
            step_pct=config.sell_extended_step_pct,
            step_index=step_index,
        )
        limit_price = price_to_order_decimal(limit_price)
        last_limit = str(limit_price)
        step_client_order_id = make_step_client_order_id(base_client_order_id, step_index)

        order = submit_extended_hours_limit_sell_once(
            trading_client=trading_client,
            config=config,
            symbol=symbol,
            qty=qty,
            limit_price=limit_price,
            client_order_id=step_client_order_id,
        )
        last_order_id = safe_order_id(order)
        notes.append(f"Extended step {step_index}: submitted sell limit {last_limit}.")

        wait_seconds = min(
            max(0.0, float(config.sell_extended_step_seconds)),
            max(0.0, total_deadline - time.time()),
        )
        terminal, status = wait_for_order_step(
            trading_client=trading_client,
            order_id=last_order_id,
            seconds=Decimal(str(wait_seconds)),
            poll_interval_seconds=config.order_poll_interval_seconds,
        )

        if terminal and status == "filled":
            notes.append(f"Extended step {step_index} filled.")
            return last_order_id, last_limit, " | ".join(notes)

        if terminal and status in RETRYABLE_ORDER_STATUSES:
            notes.append(f"Extended step {step_index} ended with status {status}; trying next step.")
        else:
            reached_bid = bid_price is not None and limit_price <= price_to_order_decimal(bid_price)
            next_fraction = (config.sell_extended_step_pct * Decimal(step_index + 1)) / Decimal("100")
            final_step = reached_bid or next_fraction >= Decimal("1")

            if final_step and config.sell_extended_leave_final_order:
                notes.append(
                    f"Final extended step left live at {last_limit}; status={status or 'unknown'}."
                )
                return last_order_id, last_limit, " | ".join(notes)

            cancel_note = cancel_order_best_effort(trading_client, last_order_id)
            notes.append(f"Extended step {step_index} not filled; {cancel_note}; moving closer to bid.")

        step_index += 1

    return last_order_id, last_limit, " | ".join(notes)


def build_row(
    symbol: str,
    qty: Optional[Decimal],
    avg_entry_price: Optional[Decimal],
    current_price: Optional[Decimal],
    market_value: Optional[Decimal],
    cost_basis: Optional[Decimal],
    current_gain_pct: Optional[Decimal],
    peak_gain_pct: Optional[Decimal],
    armed: bool,
    drawdown_from_peak_pct: Optional[Decimal],
    sell_signal: bool,
    first_seen_at: str,
    last_action: str,
    last_order_id: str,
    exit_client_order_id: str,
    notes: str,
    now: str,
) -> List[Any]:
    return [
        symbol,
        decimal_to_sheet(qty, "0.000000000"),
        decimal_to_sheet(avg_entry_price),
        decimal_to_sheet(current_price),
        decimal_to_sheet(market_value),
        decimal_to_sheet(cost_basis),
        decimal_to_sheet(current_gain_pct),
        decimal_to_sheet(peak_gain_pct),
        "TRUE" if armed else "FALSE",
        decimal_to_sheet(drawdown_from_peak_pct),
        "TRUE" if sell_signal else "FALSE",
        first_seen_at,
        last_action,
        last_order_id,
        exit_client_order_id,
        now,
        now,
        notes,
    ]


def run_cycle(
    config: Config,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    worksheet: Any,
) -> Dict[str, Any]:
    now = utc_now_iso()

    state = read_seller_state(worksheet)
    positions = trading_client.get_all_positions()
    is_market_open, market_reason = market_is_open(trading_client)

    protected_sell_symbols, order_lookup_note = get_recent_sell_order_symbols(
        trading_client,
        config.sell_order_lookback_minutes,
    )

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

        first_seen_at = prior.get("first_seen_at", "").strip() or now
        exit_client_order_id = prior.get("exit_client_order_id", "").strip()
        if not exit_client_order_id:
            exit_client_order_id = make_exit_client_order_id(symbol, first_seen_at, qty, avg_entry_price)

        # Persistent all-time-high gain tracking for this currently-open symbol lifecycle.
        # If the row vanishes after a full sell and the symbol is bought again later, this starts over.
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

        last_action = prior.get("last_action", "")
        last_order_id = prior.get("last_order_id", "")
        notes_parts: List[str] = []
        if order_lookup_note:
            notes_parts.append(order_lookup_note)

        side_text = safe_side_text(field(pos, "side", "long"))
        is_long_position = "long" in side_text or side_text == ""

        prior_sell_blocks, prior_sell_note = prior_sell_submission_still_blocks(
            trading_client,
            last_action,
            last_order_id,
        )
        has_recent_sell_order = symbol in protected_sell_symbols
        if has_recent_sell_order:
            notes_parts.append("Recent sell order found from lookup; visibility only, not a sell blocker.")

        if market_value is not None and market_value < config.sell_min_market_value:
            notes_parts.append("Below SELL_MIN_MARKET_VALUE; sell blocked.")
            sell_signal = False

        if not is_long_position:
            notes_parts.append("Non-long position; skipped by v1 seller.")
            sell_signal = False

        if sell_signal:
            if qty is None or qty <= 0:
                last_action = "SELL_SIGNAL_BLOCKED"
                notes_parts.append("Sell signal, but qty is missing or <= 0.")
                blocked_signals.append({"symbol": symbol, "reason": notes_parts[-1]})
            elif prior_sell_blocks:
                notes_parts.append(prior_sell_note)
                blocked_signals.append({"symbol": symbol, "reason": prior_sell_note})
            elif config.sell_dry_run:
                if is_market_open:
                    order_mode = "regular-hours market sell"
                elif config.sell_extended_hours:
                    order_mode = "extended-hours stepped limit sell"
                else:
                    order_mode = "market-closed blocked sell"
                last_action = "DRY_RUN_SELL_SIGNAL"
                note = f"Dry run: would submit {order_mode} for full position."
                notes_parts.append(note)
                dry_run_signals.append(symbol)
            elif is_market_open:
                order = submit_regular_market_sell(
                    trading_client,
                    symbol,
                    qty,
                    exit_client_order_id,
                )
                last_action = "SELL_SUBMITTED"
                last_order_id = safe_order_id(order)
                notes_parts.append("Submitted regular-hours market sell for full position.")
                submitted_sells.append({"symbol": symbol, "order_id": last_order_id, "mode": "market"})
            elif config.sell_extended_hours:
                if current_price is None or current_price <= 0:
                    last_action = "SELL_SIGNAL_BLOCKED"
                    note = "Sell signal outside regular hours, but current_price is missing; cannot build stepped limit order."
                    notes_parts.append(note)
                    blocked_signals.append({"symbol": symbol, "reason": note})
                else:
                    last_order_id, last_limit_price, chase_note = submit_extended_hours_stepped_sell(
                        trading_client=trading_client,
                        data_client=data_client,
                        config=config,
                        symbol=symbol,
                        qty=qty,
                        current_price=current_price,
                        base_client_order_id=exit_client_order_id,
                    )
                    last_action = "SELL_SUBMITTED"
                    notes_parts.append(chase_note)
                    submitted_sells.append({
                        "symbol": symbol,
                        "order_id": last_order_id,
                        "mode": "extended_stepped_limit",
                        "last_limit_price": last_limit_price,
                    })
            else:
                last_action = "SELL_SIGNAL_MARKET_CLOSED"
                note = "Sell signal, but market is closed and SELL_EXTENDED_HOURS=false."
                notes_parts.append(note)
                blocked_signals.append({"symbol": symbol, "reason": note})
        elif not notes_parts:
            if armed:
                last_action = "ARMED"
            else:
                last_action = "MONITORING"

        rows.append(
            build_row(
                symbol=symbol,
                qty=qty,
                avg_entry_price=avg_entry_price,
                current_price=current_price,
                market_value=market_value,
                cost_basis=cost_basis,
                current_gain_pct=current_gain_pct,
                peak_gain_pct=peak_gain_pct,
                armed=armed,
                drawdown_from_peak_pct=drawdown_from_peak_pct,
                sell_signal=sell_signal,
                first_seen_at=first_seen_at,
                last_action=last_action,
                last_order_id=last_order_id,
                exit_client_order_id=exit_client_order_id,
                notes=" | ".join(notes_parts),
                now=now,
            )
        )

    write_seller_rows(worksheet, rows)

    result = {
        "state": "ok",
        "updated_at": now,
        "positions": len(rows),
        "dry_run": config.sell_dry_run,
        "market_open": is_market_open,
        "market_reason": market_reason,
        "extended_hours_enabled": config.sell_extended_hours,
        "extended_step_pct": str(config.sell_extended_step_pct),
        "extended_step_seconds": str(config.sell_extended_step_seconds),
        "protected_sell_symbols": sorted(protected_sell_symbols),
        "order_lookup_note": order_lookup_note,
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
DATA_CLIENT = init_data_client(CONFIG)
WORKSHEET = init_worksheet(CONFIG)

app = FastAPI(title="Seller Bot")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "tab": CONFIG.seller_tab,
        "dry_run": CONFIG.sell_dry_run,
        "auto_start": CONFIG.bot_auto_start,
        "poll_seconds": str(CONFIG.poll_seconds),
        "extended_hours_enabled": CONFIG.sell_extended_hours,
        "extended_step_pct": str(CONFIG.sell_extended_step_pct),
        "extended_step_seconds": str(CONFIG.sell_extended_step_seconds),
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    return LAST_STATUS


@app.post("/run")
async def run_once() -> Dict[str, Any]:
    return await asyncio.to_thread(run_cycle, CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET)


async def background_loop() -> None:
    while True:
        started = time.time()
        cycle_had_error = False

        try:
            result = await asyncio.to_thread(run_cycle, CONFIG, TRADING_CLIENT, DATA_CLIENT, WORKSHEET)
            logging.info("Cycle complete: %s", result)
        except Exception as exc:
            cycle_had_error = True
            logging.exception("Cycle failed")
            global LAST_STATUS
            LAST_STATUS = {
                "state": "error",
                "updated_at": utc_now_iso(),
                "error": str(exc),
            }

        elapsed = Decimal(str(time.time() - started))
        if cycle_had_error:
            sleep_for = max(Decimal("0"), CONFIG.error_backoff_seconds)
        else:
            sleep_for = max(Decimal("0"), CONFIG.poll_seconds - elapsed)

        if sleep_for > 0:
            await asyncio.sleep(float(sleep_for))
        else:
            # Yield to the event loop, but start the next cycle immediately.
            await asyncio.sleep(0)


@app.on_event("startup")
async def startup_event() -> None:
    if CONFIG.bot_auto_start:
        asyncio.create_task(background_loop())
