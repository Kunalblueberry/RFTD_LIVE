import sys
from dotenv import load_dotenv
load_dotenv()
import os
sys.path.append(os.getenv('ROOT_DIR'))
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, time as dtime, timedelta, date
import pandas as pd
from copy import deepcopy
import math

# Infrastructure imports
from Utils.utils import (
    Logger, get_strategy_info, read_json, now, refresh,
    get_strike_from_trading_symbol, df_to_image, add_comma,
    get_trading_symbol, get_holidays
)
from Utils.alerts import Alert
from SMM.Tagpos.tagpos import Tagpos as tagpos
from Database.db import PickleIO as pio, SHM as shm
from AMS.api import Api
from OES.oes import OES
from SMM.MDA.mda_utils import get_ticks

# Compiled strategy engine
from rftd_engine import RFTDSignalEngine


PARAMS_FILE_PATH = "/home/trader1/RFTD_Live/params_rftd.json"


@dataclass
class Signal:
    symbol: str
    action: str          # 'ENTRY', 'EXIT'
    side: str            # 'BUY', 'SELL'
    quantity: int
    trading_symbol: str
    order_type: str = 'LIMITMARKET'
    limit_price: float = 0
    trigger_price: float = 0
    metadata: Dict = field(default_factory=dict)


class RFTDStrategy:
    """
    RFTD Strategy
    - VAR hedge: placed at strategy_var_entry_time on naked SHORTs, exited at strategy_var_exit_time
    - VAR hedges persisted to SHM so they survive restarts
    - No margin hedge
    - No EOD expiry cleanup
    """

    VERSION = "8.1.0"

    def __init__(self, strategy_name: str, tag_name: str, symbol: str, launch_mode: str):
        self.strategy_name = strategy_name
        self.tag_name = tag_name
        self.symbol = symbol
        self.launch_mode = launch_mode
        self.is_running = False

        # Initialize logger FIRST
        self.logger = Logger(strategy_name)
        self.logger.log(f"[INIT] Starting RFTDStrategy version {self.VERSION}")

        # Load instruments
        self.instruments = pio.get('instruments')
        self.instruments_df = pio.get('instruments_df')

        # Build symbol token map
        self.symbol_token_map = dict(zip(
            self.instruments_df["tradingsymbol"],
            self.instruments_df["exchange_instrument_id"],
        ))

        # Load and validate params
        self.params = self._load_params()
        if not self._validate_params():
            self.logger.log(f"[RFTDStrategy] Params validation failed for: {self.tag_name}", level='error')
            Alert.send_slack(
                message=f"[RFTDStrategy] Params validation failed for: {self.tag_name}. Exiting.",
                channel='alerts'
            )
            sys.exit(1)

        # Extract params
        self.broker_name = self.params.get('broker_name', 'simulation')
        self.account_id = self.params.get('account_id', 'simulation')
        self.slack_channel = self.params.get('slack_channel', 'rftd')
        self.offset_type = self.params.get('offset_type', 'absolute')
        self.offset_value = self.params.get('offset_value', 5)
        self.tagpos_ss_timeframe = self.params.get('tagpos_ss_timeframe', 30)
        self.exchange = self.params.get('exchange', 'NFO')
        self.product = self.params.get('product', 'NRML')
        self.loop_interval = self.params.get('loop_interval', 5)
        self.strike_diff = self.params.get('strike_step', 50)
        self.spot_symbol = self.params.get('symbol_spot', self._get_default_spot_symbol())

        # Capital/risk params
        self.max_margin_limit = self.params.get('max_margin_limit', 95)
        self.total_capital = self.params.get('total_capital', 1000000)
        self.var_risk_pct = self.params.get('var_risk_pct', 5)

        # Lot sizing: how many lots per SHORT (sell) and LONG (buy) leg.
        # sell_capital_ratio=10 → SHORT entries use 10 * lot_size units.
        # buy_capital_ratio=10  → LONG  entries use 10 * lot_size units.
        # EXIT quantity is scaled by the same ratio (engine always sends 1 lot).
        self.sell_capital_ratio = int(self.params.get('sell_capital_ratio', 1))
        self.buy_capital_ratio = int(self.params.get('buy_capital_ratio', 1))

        # VAR hedge timing params
        self.strategy_var_entry_time = dtime(
            *map(int, self.params.get('strategy_var_entry_time', '15:20:00').split(':'))
        ) if isinstance(self.params.get('strategy_var_entry_time'), str) else dtime(15, 20, 0)

        self.strategy_var_exit_time = dtime(
            *map(int, self.params.get('strategy_var_exit_time', '09:20:00').split(':'))
        ) if isinstance(self.params.get('strategy_var_exit_time'), str) else dtime(9, 20, 0)

        # Initialize API
        self._init_api()

        # Initialize OES
        self.oes = OES(self.logger)

        # Order tracking
        self.entry_order_responses = {}
        self.exit_order_responses = {}
        self.hedge_order_responses = {}
        self.rms_order_responses = {}
        self.order_history = {}
        self.slack_alerts_dict = {}

        # VAR hedge tracking - persisted to SHM so it survives restarts
        self.var_hedges = shm.get(f'var_hedges_{self.tag_name}') or {}

        # Timing flags
        _now = now().time()
        # var_done: True if we already placed VAR hedges today (past entry time, non-expiry)
        self.var_done = (_now >= self.strategy_var_entry_time) and (not self._is_expiry_day())
        # var_exit_done: reset each startup so morning exit always runs in its window
        self.var_exit_done = False

        self.logger.log(
            f"[INIT] var_done={self.var_done} var_exit_done={self.var_exit_done} "
            f"var_hedges_loaded={self.var_hedges} at restart time {_now}"
        )

        # Timing
        self.tagpos_ss_timestamp = datetime.combine(now().date(), dtime(9, 15, 0))
        self.last_rms_check = now()

        # Get lot size and expiry info
        self._init_instrument_info()

        # Initialize engine
        self._init_engine()

        # Send startup notification
        self.logger.log(f"[{self.tag_name}] Initialized successfully (v{self.VERSION})")
        Alert.send_slack(
            message=f"[{self.tag_name}] Strategy initialized | Symbol: {self.symbol} | Broker: {self.broker_name} | Version: {self.VERSION}",
            channel=self.slack_channel
        )

        self._send_params_screenshot()

    def _load_params(self) -> Dict:
        """Load params from get_strategy_info() or JSON fallback."""
        params = {}
        try:
            if self.launch_mode == 'cronjob':
                params = get_strategy_info(self.tag_name)
                if params:
                    self.logger.log(f"[_load_params] Loaded from get_strategy_info for {self.tag_name}")
                    return params

            self.logger.log(f"[_load_params] Falling back to JSON: {PARAMS_FILE_PATH}")
            json_data = read_json(file_name=PARAMS_FILE_PATH)

            if json_data and self.tag_name in json_data:
                params = json_data[self.tag_name]
                self.logger.log(f"[_load_params] Loaded from JSON for {self.tag_name}")
            else:
                self.logger.log(f"[_load_params] No params found for {self.tag_name}", level='error')

        except Exception as e:
            self.logger.log(f"[_load_params] Exception: {str(e)}", level='error')

        return params

    def _validate_params(self) -> bool:
        """Validate required params."""
        if not self.params:
            return False

        required = ['broker_name', 'account_id', 'slack_channel', 'exchange']
        missing = [f for f in required if f not in self.params]

        if missing:
            self.logger.log(f"[_validate_params] Missing: {missing}", level='error')
            return False

        return True

    def _init_api(self):
        """Initialize API."""
        self.api = Api(
            broker_name=self.broker_name,
            account_id=self.account_id,
            logger=self.logger
        )

        self.api.get_order_packet()
        self.api.order_packet['api_obj'] = self.api
        self.api.order_packet['exchange'] = self.exchange
        self.api.order_packet['symbol'] = self.symbol
        self.api.order_packet['product'] = self.product
        self.api.order_packet['validity'] = 'DAY'
        self.api.order_packet['tag'] = self.tag_name

        self.api.instruments = self.instruments
        self.logger.log(f"[API] Broker: {self.broker_name}, Account: {self.account_id}")

    def _init_instrument_info(self):
        """Get lot size, strike diff, expiry from instruments."""
        try:
            trading_symbol_info = get_trading_symbol(
                self.logger, self.symbol, 'CE', 1, df=self.instruments_df
            )
            if isinstance(trading_symbol_info, pd.DataFrame) and not trading_symbol_info.empty:
                self.lot_size = int(trading_symbol_info.iloc[0]['lot_size'])
                self.strike_diff = float(trading_symbol_info.iloc[0]['strike_diff'])
                self.current_expiry = trading_symbol_info.iloc[0]['expiry_date']
            else:
                self.lot_size = self.params.get('lot_size', 75)
                self.current_expiry = None

            self.logger.log(f"[INSTRUMENT] Lot: {self.lot_size}, Strike Diff: {self.strike_diff}")
        except Exception as e:
            self.logger.log(f"[_init_instrument_info] Error: {str(e)}", level='error')
            self.lot_size = self.params.get('lot_size', 75)
            self.current_expiry = None

    def _init_engine(self):
        """Initialize RFTD Signal Engine."""
        self.engine = RFTDSignalEngine()

        instrument_config = {
            "instrument_name": self.symbol,
            "symbol_spot": self.spot_symbol,
            "symbol_base": self.symbol,
            "spot_exchange": self.params.get('spot_exchange', 'NSE'),
            "options_exchange": self.exchange,
            "strike_step": self.strike_diff,
        }

        self.engine.initialize(
            api=self.api,
            instruments_df=self.instruments_df,
            instrument_config=instrument_config,
        )

        self.logger.log(f"[ENGINE] Initialized for {self.symbol}")

    def _get_default_spot_symbol(self) -> str:
        """Get default spot symbol."""
        spot_map = {
            'NIFTY': 'NIFTY 50',
            'BANKNIFTY': 'NIFTY BANK',
            'FINNIFTY': 'NIFTY FIN SERVICE',
            'SENSEX': 'SENSEX',
        }
        return spot_map.get(self.symbol, self.symbol)

    def _send_params_screenshot(self):
        """Send params to Slack."""
        try:
            params_df = pd.DataFrame([self.params]).T
            params_df.columns = ['Value']
            df_to_image(params_df, f"{self.tag_name}_params")
            Alert.send_slack(
                message=f"{self.tag_name}_PARAMS",
                file_name=f"{self.tag_name}_params",
                channel=self.slack_channel
            )
        except Exception as e:
            self.logger.log(f"[_send_params_screenshot] Failed: {str(e)}", level='error')

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    def get_positions(self) -> Optional[pd.DataFrame]:
        """Get open positions from TagPos."""
        positions = tagpos.get_positions(self.instruments, tag=self.tag_name)
        if not positions or tagpos.df.empty:
            return None

        df = tagpos.df[tagpos.df['quantity'] != 0].copy()
        if df.empty:
            return None

        df['strike'] = df['trading_symbol'].apply(
            lambda x: int(list(get_strike_from_trading_symbol(x).values())[0])
        )
        df['option_type'] = df['trading_symbol'].str[-2:]

        return df

    def send_tagpos_screenshot(self):
        """Send TagPos screenshot to Slack."""
        try:
            positions = tagpos.get_positions(self.instruments, tag=self.tag_name)
            df = tagpos.df.copy()

            if df is None or df.empty:
                self.logger.log("[SCREENSHOT] No positions")
                return

            total_urpl = round(df['urpl'].sum(), 2) if 'urpl' in df.columns else 0
            total_rpl = round(df['rpl'].sum(), 2) if 'rpl' in df.columns else 0
            total_pl = round(df['pl'].sum(), 2) if 'pl' in df.columns else 0

            new_row = pd.DataFrame([{
                'trading_symbol': "TOTAL",
                'urpl': add_comma(total_urpl),
                'rpl': add_comma(total_rpl),
                'pl': add_comma(total_pl)
            }])
            new_row.index = ['TOTAL']
            df_display = pd.concat([df, new_row]).fillna(0)
            df_display['trading_symbol'] = df_display.index

            df_to_image(df_display, f"{self.tag_name}_tagpos")
            Alert.send_slack(
                message=f"{self.tag_name}_TAGPOS | P&L: {add_comma(total_pl)}",
                file_name=f"{self.tag_name}_tagpos",
                channel=self.slack_channel
            )

        except Exception as e:
            self.logger.log(f"[SCREENSHOT] Error: {str(e)}", level='error')

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================
    def update_order_packet(self, trading_symbol: str, token: str, quantity: int,
                            order_side: str, order_type: str, limit_price: float,
                            trigger_price: float):
        """Update order packet.
        'lots' must be lot count (quantity // lot_size), NOT raw units.
        IIFL validates lots > 0 for SELL orders (margin allocation) and
        silently accepts lots=0 for BUY — that's why only SELL fails when
        lots is wrong.
        """
        self.api.order_packet['trading_symbol'] = trading_symbol
        self.api.order_packet['token'] = token
        self.api.order_packet['quantity'] = quantity
        self.api.order_packet['lots'] = max(1, quantity // self.lot_size)
        self.api.order_packet['order_side'] = order_side
        self.api.order_packet['order_type'] = order_type
        self.api.order_packet['limit_price'] = limit_price
        self.api.order_packet['trigger_price'] = trigger_price

    def get_average_execution_price(self, order_responses: Dict) -> Dict:
        """Calculate avg execution price."""
        avg_prices = {}

        for order_id, details in order_responses.items():
            if details.get('status') != 'COMPLETED':
                continue

            pkt = details.get('order_packet', {})
            ts = pkt.get('trading_symbol', '')
            avg_price = float(pkt.get('average_price', 0))
            exec_qty = float(pkt.get('executed_quantity', 0))

            if ts in avg_prices:
                avg_prices[ts]['cumsum'] += avg_price * exec_qty
                avg_prices[ts]['cumqty'] += exec_qty
                avg_prices[ts]['avg_price'] = round(
                    avg_prices[ts]['cumsum'] / avg_prices[ts]['cumqty'], 2
                )
            else:
                avg_prices[ts] = {
                    'cumsum': avg_price * exec_qty,
                    'cumqty': exec_qty,
                    'avg_price': avg_price
                }

        return avg_prices

    def calculate_and_persist_slippage(self, order_responses: Dict, expected_price_map: Dict):
        """Calculate and persist slippage to SHM."""
        try:
            total_slippage = 0.0

            for order_id, info in order_responses.items():
                if info.get('status') != 'COMPLETED':
                    continue

                pkt = info.get('order_packet', {})
                ts = pkt.get('trading_symbol', '')
                side = pkt.get('order_side', '')
                exec_price = float(pkt.get('average_price', 0))
                exec_qty = int(pkt.get('executed_quantity', 0))
                exp_price = float(expected_price_map.get(ts, 0))

                if exp_price == 0 or exec_qty == 0:
                    continue

                if side == 'BUY':
                    slip = exp_price - exec_price
                else:
                    slip = exec_price - exp_price

                slip_value = round(slip * exec_qty, 2)
                total_slippage += slip_value

            total_slippage = round(total_slippage, 2)
            date_key = now().strftime('%d %b, %Y')
            shm_key = f'slippage_{self.tag_name}'
            existing = shm.get(shm_key) or {}

            if date_key not in existing:
                existing[date_key] = {}

            existing[date_key][self.symbol] = round(
                existing[date_key].get(self.symbol, 0) + total_slippage, 2
            )
            shm.set(shm_key, existing)

            return total_slippage

        except Exception as e:
            self.logger.log(f"[SLIPPAGE] Failed: {str(e)}", level='error')
            return 0.0

    def _refresh_instruments(self):
        """
        Reload instruments from pio.
        Called before order execution so newly rolled weekly expiry symbols
        (e.g. NIFTY26APR24350CE) are present in instruments when OES's
        get_rounded_freeze_qty() does instruments[trading_symbol].
        Without this, SELL orders for new-expiry symbols fail with lots=0.
        """
        self.instruments = pio.get('instruments')
        self.api.instruments = self.instruments
        if hasattr(self.oes, 'instruments'):
            self.oes.instruments = self.instruments

    def strategy_order_execution(self, order_packets: List[Dict], sleep_time: int = 3) -> Dict:
        """Execute orders with chasing and slippage calculation."""
        if not order_packets:
            return {}

        # Reload instruments so OES has the latest symbols (fixes lots=0 on new expiry)
        self._refresh_instruments()

        ts_list = [pkt['trading_symbol'] for pkt in order_packets]
        all_ticks = get_ticks(ts_list, self.instruments)
        expected_prices = {
            pkt['trading_symbol']: all_ticks.get(pkt['trading_symbol'], {}).get('last_price', 0)
            for pkt in order_packets
        }

        self.logger.log(f"[EXECUTE] Placing {len(order_packets)} orders")

        responses = self.oes.place_order(
            order_packets=order_packets,
            offset_type=self.offset_type,
            offset_value=self.offset_value,
        )

        refresh(sleep_time)

        responses.update(self.oes.update_order_packet_status(
            api_obj=self.api,
            order_ids=responses,
            channel_name=self.slack_channel,
        ))

        self.oes.chase_orders_to_fill(
            order_packets=responses,
            api_obj=self.api,
            channel_name=self.slack_channel,
            offset_type=self.offset_type,
            offset_value=self.offset_value,
        )

        refresh(sleep_time)

        responses.update(self.oes.update_order_packet_status(
            api_obj=self.api,
            order_ids=responses,
            channel_name=self.slack_channel,
        ))

        self.calculate_and_persist_slippage(responses, expected_prices)

        for oid, resp in responses.items():
            self.order_history[oid] = resp

        return responses

    # =========================================================================
    # SIGNAL GENERATION AND EXECUTION
    # =========================================================================
    def _get_signal_quantity(self, leg_role: str, trading_symbol: str = '') -> int:
        """
        Compute quantity from capital allocation in rupees.
        sell_capital_ratio / buy_capital_ratio are amounts in LAKHS
        (e.g. sell_capital_ratio=10 → ₹10,00,000 allocated for shorts).

        SHORT: lots = floor(allocated_capital / margin_per_lot)
               margin_per_lot fetched from broker API for 1 lot of this symbol.
        LONG:  lots = floor(allocated_capital / (ltp * lot_size))
               ltp fetched from tick stream for premium-per-lot cost.

        Falls back to 1 lot if the API call fails or returns 0.
        """
        if leg_role == 'SHORT':
            allocated = self.sell_capital_ratio * 100000
            if trading_symbol:
                try:
                    single_lot = {
                        'exchange': self.exchange,
                        'tradingsymbol': trading_symbol,
                        'transaction_type': 'SELL',
                        'variety': 'regular',
                        'order_type': 'LIMIT',
                        'product': self.product,
                        'quantity': self.lot_size,
                        'price': 0,
                    }
                    margin_per_lot = self.api.get_basket_order_margins([single_lot])
                    if margin_per_lot and margin_per_lot > 0:
                        lots = max(1, int(allocated / margin_per_lot))
                        self.logger.log(
                            f"[QTY] SHORT {trading_symbol}: margin/lot={margin_per_lot:,.0f} "
                            f"allocated={allocated:,.0f} → {lots} lots"
                        )
                        return lots * self.lot_size
                except Exception as e:
                    self.logger.log(f"[QTY] Margin fetch failed for {trading_symbol}: {e}", level='error')
            return self.lot_size

        elif leg_role == 'LONG':
            allocated = self.buy_capital_ratio * 100000
            if trading_symbol:
                try:
                    ticks = get_ticks([trading_symbol], self.instruments)
                    ltp = ticks.get(trading_symbol, {}).get('last_price', 0)
                    if ltp and ltp > 0:
                        premium_per_lot = ltp * self.lot_size
                        lots = max(1, int(allocated / premium_per_lot))
                        self.logger.log(
                            f"[QTY] LONG {trading_symbol}: ltp={ltp} premium/lot={premium_per_lot:,.0f} "
                            f"allocated={allocated:,.0f} → {lots} lots"
                        )
                        return lots * self.lot_size
                except Exception as e:
                    self.logger.log(f"[QTY] LTP fetch failed for {trading_symbol}: {e}", level='error')
            return self.lot_size

        return self.lot_size

    def generate_signals(self) -> List[Signal]:
        """Get signals from engine and convert to Signal objects."""
        order_dicts = self.engine.run_cycle()

        if not order_dicts:
            return []

        signals = []
        for od in order_dicts:
            self.logger.log(f"[ENGINE] Signal: {od}")

            price = od.get("entry_price", 0) or od.get("exit_price", 0)
            leg_role = od.get("leg_role", "")
            qty = self._get_signal_quantity(leg_role, od.get("trading_symbol", ""))

            sig = Signal(
                symbol=od["symbol"],
                action=od["action"],
                side=od["side"],
                quantity=qty,
                trading_symbol=od["trading_symbol"],
                order_type="LIMITMARKET",
                limit_price=price,
                trigger_price=0,
                metadata={
                    'leg_role': od.get("leg_role", ""),
                    'opt_type': od.get("opt_type", ""),
                    'strike': od.get("strike", ""),
                    'regime': od.get("regime", ""),
                    'reason': od.get("reason", ""),
                    'stop_price': od.get("stop_price"),
                    'anchor': od.get("anchor", ""),
                    'expiry': od.get("expiry", ""),
                },
            )
            signals.append(sig)

        return signals

    def execute_signals(self, signals: List[Signal], sleep: int = 3) -> Dict:
        """Execute signals."""
        if not signals:
            return {}

        order_packets = []
        alert_messages = []

        for s in signals:
            self.update_order_packet(
                trading_symbol=s.trading_symbol,
                token=self.symbol_token_map.get(s.trading_symbol, ''),
                quantity=s.quantity,
                order_side=s.side,
                order_type=s.order_type,
                limit_price=s.limit_price,
                trigger_price=s.trigger_price
            )

            order_packets.append(deepcopy(self.api.order_packet))

            alert_messages.append(
                f"[{s.action}] {s.metadata.get('leg_role', '')} {s.metadata.get('opt_type', '')} "
                f"{s.metadata.get('strike', '')} | Side: {s.side} | Qty: {s.quantity} | "
                f"Price: {s.limit_price} | Regime: {s.metadata.get('regime', '') or s.metadata.get('reason', '')}"
            )

        responses = self.strategy_order_execution(order_packets, sleep)

        for msg in alert_messages:
            Alert.send_slack(message=msg, channel=self.slack_channel)

        return responses

    # =========================================================================
    # MARGIN FILTERING (entry approval only — no margin hedge)
    # =========================================================================
    def _filter_entries_by_margin(self, entries: List[Signal]) -> List[Signal]:
        """Filter entries by margin/capital limits."""
        approved = []

        for entry in entries:
            positions = self.get_positions()
            current_margin = 0

            if positions is not None:
                positions_list = self._format_positions_for_margin(positions)
                current_margin = self.api.get_basket_order_margins(positions_list)

            new_position = self._format_signal_for_margin(entry)
            new_margin = self.api.get_basket_order_margins([new_position])

            total_margin = current_margin + new_margin
            margin_pct = (total_margin / self.total_capital) * 100 if self.total_capital > 0 else 0

            if margin_pct > self.max_margin_limit:
                skip_msg = (
                    f"[SKIP] {entry.action} {entry.metadata.get('leg_role', '')} "
                    f"{entry.metadata.get('opt_type', '')} {entry.metadata.get('strike', '')} "
                    f"rejected — Margin would be {margin_pct:.1f}% (limit: {self.max_margin_limit}%)"
                )
                self.logger.log(skip_msg)
                Alert.send_slack(message=skip_msg, channel=self.slack_channel)
                continue

            if total_margin > self.total_capital:
                skip_msg = (
                    f"[SKIP] {entry.action} {entry.metadata.get('leg_role', '')} "
                    f"{entry.metadata.get('opt_type', '')} {entry.metadata.get('strike', '')} "
                    f"rejected — Capital limit breach: Required {total_margin:,.0f} > Available {self.total_capital:,.0f}"
                )
                self.logger.log(skip_msg)
                Alert.send_slack(message=skip_msg, channel=self.slack_channel)
                continue

            approved.append(entry)
            self.logger.log(
                f"[APPROVED] {entry.metadata.get('leg_role', '')} {entry.metadata.get('opt_type', '')} "
                f"{entry.metadata.get('strike', '')} | Margin: {margin_pct:.1f}%"
            )

        return approved

    def _format_positions_for_margin(self, df: pd.DataFrame) -> List[Dict]:
        """Format TagPos positions for margin API."""
        positions = []
        for _, row in df.iterrows():
            positions.append({
                'exchange': self.exchange,
                'tradingsymbol': row['trading_symbol'],
                'transaction_type': 'BUY' if row['quantity'] > 0 else 'SELL',
                'variety': 'regular',
                'order_type': 'LIMIT',
                'product': self.product,
                'quantity': abs(row['quantity']),
                'price': row.get('price', 0)
            })
        return positions

    def _format_signal_for_margin(self, signal: Signal) -> Dict:
        """Format signal for margin API."""
        return {
            'exchange': self.exchange,
            'tradingsymbol': signal.trading_symbol,
            'transaction_type': signal.side,
            'variety': 'regular',
            'order_type': 'LIMIT',
            'product': self.product,
            'quantity': signal.quantity,
            'price': signal.limit_price
        }

    # =========================================================================
    # VAR HEDGE MANAGEMENT
    # =========================================================================
    def _compute_sell_payoff(self, sell_df: pd.DataFrame, simulated_spot: float) -> float:
        """Compute combined payoff for SHORT positions at a simulated spot."""
        combined_payoff = 0.0

        for _, row in sell_df.iterrows():
            strike = int(row['strike'])
            opt_type = row['option_type']

            if opt_type == 'CE':
                intrinsic = max(0, simulated_spot - strike)
            else:
                intrinsic = max(0, strike - simulated_spot)

            premium = float(row.get('price', 0))
            qty = abs(row['quantity'])
            payoff = (premium - intrinsic) * qty
            combined_payoff += payoff

        return combined_payoff

    def _get_spot_ltp(self) -> float:
        """
        Get Nifty spot LTP.
        get_ticks uses instruments dict which is NFO-keyed; 'NIFTY 50' (NSE index)
        is often absent from it, causing spot_ltp=0 and VAR to silently fail.
        Falls back to api.get_ltp if get_ticks returns nothing.
        """
        # Primary: get_ticks (fast, uses Zerodha/MOSL tick stream)
        spot_ticks = get_ticks([self.spot_symbol], self.instruments)
        ltp = spot_ticks.get(self.spot_symbol, {}).get('last_price', 0)
        if ltp > 0:
            return float(ltp)

        self.logger.log(f"[SPOT] get_ticks returned 0 for {self.spot_symbol}, trying api.get_ltp")

        # Fallback: broker API LTP call
        try:
            spot_exchange = self.params.get('spot_exchange', 'NSE')
            ltp_resp = self.api.get_ltp({spot_exchange: [self.spot_symbol]})
            if ltp_resp and self.spot_symbol in ltp_resp:
                ltp = float(ltp_resp[self.spot_symbol].get('last_price', 0))
                if ltp > 0:
                    self.logger.log(f"[SPOT] api.get_ltp succeeded: {self.spot_symbol}={ltp}")
                    return ltp
        except Exception as e:
            self.logger.log(f"[SPOT] api.get_ltp failed: {e}", level='error')

        return 0.0

    def compute_var_breach_spot(self, sell_df: pd.DataFrame) -> tuple:
        """Calculate spot levels where VAR risk limit is breached."""

        spot_ltp = self._get_spot_ltp()

        if spot_ltp == 0:
            msg = f"[VAR] Could not get spot LTP for {self.spot_symbol} (both get_ticks and api.get_ltp failed) — VAR hedge cannot be placed"
            self.logger.log(msg, level='error')
            Alert.send_slack(message=msg, channel=self.slack_channel)
            return None, None

        var_limit = self.total_capital * (self.var_risk_pct / 100)
        self.logger.log(f"[VAR] spot={spot_ltp} var_limit={var_limit:,.0f} ({self.var_risk_pct}% of {self.total_capital:,.0f})")

        step = 50
        max_move = 3000
        up_breach = None
        down_breach = None

        for move in range(0, max_move, step):
            simulated_spot = spot_ltp + move
            payoff = self._compute_sell_payoff(sell_df, simulated_spot)
            if payoff <= -var_limit:
                up_breach = simulated_spot
                break

        for move in range(0, max_move, step):
            simulated_spot = spot_ltp - move
            payoff = self._compute_sell_payoff(sell_df, simulated_spot)
            if payoff <= -var_limit:
                down_breach = simulated_spot
                break

        self.logger.log(f"[VAR] Breach points — Up: {up_breach}, Down: {down_breach}")
        return up_breach, down_breach

    def _create_var_hedge(self, breach_spot: float, opt_type: str, quantity: int):
        """Place a single VAR hedge option at the breach strike."""

        if opt_type == 'CE':
            var_strike = math.ceil(breach_spot / self.strike_diff) * self.strike_diff
        else:
            var_strike = math.floor(breach_spot / self.strike_diff) * self.strike_diff

        var_info = get_trading_symbol(
            self.logger, self.symbol, opt_type, 1,
            strikes=[var_strike], df=self.instruments_df
        )

        if isinstance(var_info, pd.DataFrame) and not var_info.empty:
            var_ts = var_info.iloc[0]['trading_symbol']
        else:
            var_ts = var_info.get('trading_symbol', '') if isinstance(var_info, dict) else ''

        if not var_ts:
            msg = f"[VAR] Failed to get trading symbol for {opt_type} strike={var_strike}"
            self.logger.log(msg, level='error')
            Alert.send_slack(message=msg, channel=self.slack_channel)
            return

        self.update_order_packet(
            trading_symbol=var_ts,
            token=self.symbol_token_map.get(var_ts, ''),
            quantity=quantity,
            order_side='BUY',
            order_type='LIMITMARKET',
            limit_price=0,
            trigger_price=0
        )

        responses = self.strategy_order_execution([deepcopy(self.api.order_packet)], sleep_time=3)

        self.var_hedges[f'{opt_type}_VAR'] = var_ts
        self._persist_var_hedges()
        self.hedge_order_responses.update(responses)

        Alert.send_slack(
            message=f"[VAR] Hedge placed: BUY {var_ts} (qty={quantity}) at breach={breach_spot}",
            channel=self.slack_channel
        )

    def _persist_var_hedges(self):
        """Save var_hedges dict to SHM so it survives restarts."""
        shm.set(f'var_hedges_{self.tag_name}', self.var_hedges)

    def _enter_var_hedges(self):
        """
        Evening: place VAR hedges on all naked SHORT positions.
        Triggered at strategy_var_entry_time (default 15:20).
        Skipped on expiry day (positions expire today, overnight hedge not needed).
        """
        self.logger.log("[VAR] Placing VAR hedges for overnight protection")
        Alert.send_slack(
            message=f"[VAR] Placing overnight VAR hedges for {self.tag_name}",
            channel=self.slack_channel
        )

        positions = self.get_positions()
        if positions is None:
            self.logger.log("[VAR] No positions found — nothing to hedge")
            self.var_done = True
            return

        sell_df = positions[positions['quantity'] < 0].copy()
        if sell_df.empty:
            self.logger.log("[VAR] No SHORT positions — nothing to hedge")
            self.var_done = True
            return

        long_df = positions[positions['quantity'] > 0].copy()

        # Gross qty per opt_type (unsigned)
        ce_short_qty = int(sell_df[sell_df['option_type'] == 'CE']['quantity'].abs().sum())
        pe_short_qty = int(sell_df[sell_df['option_type'] == 'PE']['quantity'].abs().sum())
        ce_long_qty  = int(long_df[long_df['option_type'] == 'CE']['quantity'].abs().sum()) if not long_df.empty else 0
        pe_long_qty  = int(long_df[long_df['option_type'] == 'PE']['quantity'].abs().sum()) if not long_df.empty else 0

        # Net naked qty: portion of shorts not covered by same-type longs
        # e.g. CE short=650, CE long=195 → naked=455 (7 lots still unprotected)
        net_naked_ce = max(0, ce_short_qty - ce_long_qty)
        net_naked_pe = max(0, pe_short_qty - pe_long_qty)

        self.logger.log(
            f"[VAR] CE: short={ce_short_qty} long={ce_long_qty} naked={net_naked_ce} | "
            f"PE: short={pe_short_qty} long={pe_long_qty} naked={net_naked_pe}"
        )

        if net_naked_ce == 0 and net_naked_pe == 0:
            self.logger.log("[VAR] All SHORT qty fully covered by matching LONGs — no hedge needed")
            self.var_done = True
            return

        # Breach computation uses full sell_df (conservative: ignores long protection
        # in payoff scan so breach point is the worst-case unhedged scenario)
        up_breach, down_breach = self.compute_var_breach_spot(sell_df)

        if up_breach is None and down_breach is None:
            Alert.send_slack(
                message=f"[VAR] WARNING: Could not compute breach points — VAR hedges NOT placed for {self.tag_name}",
                channel=self.slack_channel
            )
            self.var_done = True
            return

        if up_breach:
            if net_naked_ce > 0:
                self._create_var_hedge(up_breach, 'CE', net_naked_ce)
            else:
                self.logger.log("[VAR] CE fully covered by longs — skipping CE hedge")

        if down_breach:
            if net_naked_pe > 0:
                self._create_var_hedge(down_breach, 'PE', net_naked_pe)
            else:
                self.logger.log("[VAR] PE fully covered by longs — skipping PE hedge")

        self.var_done = True
        Alert.send_slack(
            message=f"[VAR] VAR hedges placed: {self.var_hedges}",
            channel=self.slack_channel
        )

    def _exit_var_hedges(self):
        """
        Morning: exit all VAR hedges placed the previous evening.
        Triggered at strategy_var_exit_time (default 09:20).
        var_hedges is loaded from SHM on startup so this works after restarts.
        """
        if not self.var_hedges:
            self.logger.log("[VAR] No VAR hedges to exit")
            return

        self.logger.log(f"[VAR] Exiting VAR hedges: {self.var_hedges}")

        exit_orders = []
        ts_list = []

        positions = self.get_positions()
        if positions is None:
            self.logger.log("[VAR] No positions found — clearing var_hedges dict")
            self.var_hedges = {}
            self._persist_var_hedges()
            return

        for key, var_ts in list(self.var_hedges.items()):
            var_row = positions[positions['trading_symbol'] == var_ts]
            if var_row.empty:
                self.logger.log(f"[VAR] {var_ts} not in TagPos (already closed?)")
                continue

            qty = abs(var_row.iloc[0]['quantity'])
            if qty == 0:
                continue

            self.update_order_packet(
                trading_symbol=var_ts,
                token=self.symbol_token_map.get(var_ts, ''),
                quantity=qty,
                order_side='SELL',
                order_type='LIMITMARKET',
                limit_price=0,
                trigger_price=0
            )
            exit_orders.append(deepcopy(self.api.order_packet))
            ts_list.append(var_ts)

        if exit_orders:
            responses = self.strategy_order_execution(exit_orders, sleep_time=3)
            self.hedge_order_responses.update(responses)
            Alert.send_slack(
                message=f"[VAR] Exited VAR hedges: {', '.join(ts_list)}",
                channel=self.slack_channel
            )

        self.var_hedges = {}
        self._persist_var_hedges()

    # =========================================================================
    # RMS
    # =========================================================================
    def rms_check_margin_breach(self):
        """
        Safety net: if current portfolio margin exceeds max_margin_limit,
        alert and square off everything. Does NOT touch positions for any
        other reason — quantity management is the engine's job.
        """
        positions = self.get_positions()
        if positions is None:
            return

        positions_list = self._format_positions_for_margin(positions)
        try:
            current_margin = self.api.get_basket_order_margins(
                positions_list, consider_positions=True
            )
        except Exception as e:
            self.logger.log(f"[RMS] Margin fetch failed: {e}", level='error')
            return

        margin_pct = (current_margin / self.total_capital) * 100 if self.total_capital > 0 else 0
        self.logger.log(f"[RMS] Margin: {current_margin:,.0f} ({margin_pct:.1f}% of {self.total_capital:,.0f})")

        if margin_pct > self.max_margin_limit:
            msg = (
                f"[RMS] MARGIN BREACH: {margin_pct:.1f}% > limit {self.max_margin_limit}% "
                f"— squaring off ALL positions for {self.tag_name}"
            )
            self.logger.log(msg, level='error')
            Alert.send_slack(message=msg, channel=self.slack_channel)
            self.exit_all_positions()

    # =========================================================================
    # EXPIRY CHECK
    # =========================================================================
    def _is_expiry_day(self) -> bool:
        """
        Check if today is the expiry day of the current target contract.
        Used to skip VAR hedge placement on expiry day (options expire today,
        no overnight risk to hedge).
        Primary source: engine.get_target_expiry().
        Fallback: self.current_expiry from _init_instrument_info.
        """
        try:
            target_expiry = self.engine.get_target_expiry()
            if target_expiry:
                return target_expiry.date() == now().date()
        except Exception as e:
            self.logger.log(f"[_is_expiry_day] Engine error: {e}", level='error')

        if self.current_expiry:
            if hasattr(self.current_expiry, 'date'):
                return self.current_expiry.date() == now().date()
            return self.current_expiry == now().date()
        return False

    # =========================================================================
    # UTILITY
    # =========================================================================
    def exit_all_positions(self):
        """Exit all positions (manual use)."""
        positions = self.get_positions()
        if positions is None:
            self.logger.log("[EXIT ALL] No positions")
            return

        Alert.send_slack(
            message=f"[EXIT ALL] Squaring off all for {self.tag_name}",
            channel=self.slack_channel
        )

        exit_orders = []
        ts_list = []

        for _, row in positions.iterrows():
            order_side = 'BUY' if row['quantity'] < 0 else 'SELL'

            self.update_order_packet(
                trading_symbol=row['trading_symbol'],
                token=self.symbol_token_map.get(row['trading_symbol'], ''),
                quantity=abs(row['quantity']),
                order_side=order_side,
                order_type='LIMITMARKET',
                limit_price=0,
                trigger_price=0
            )
            exit_orders.append(deepcopy(self.api.order_packet))
            ts_list.append(row['trading_symbol'])

        if exit_orders:
            self.strategy_order_execution(exit_orders, sleep_time=3)
            Alert.send_slack(
                message=f"[EXIT ALL] Squared off: {', '.join(ts_list)}",
                channel=self.slack_channel
            )

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================
    def execute(self):
        """Main execution loop."""

        self.is_running = True
        self.logger.log(f"[EXECUTE] {self.tag_name} started (v{self.VERSION})")
        Alert.send_slack(
            message=f"[EXECUTE] {self.tag_name} execution started (v{self.VERSION})",
            channel=self.slack_channel
        )

        try:
            while self.is_running:
                now_time = now().time()
                now_dt = now()

                # Market start check
                if now_time < dtime(9, 15, 10):
                    self.logger.log("Market not started, waiting...")
                    time.sleep(5)
                    continue

                # Market end check
                if now_time >= dtime(15, 30, 0):
                    self.logger.log("Market ended")
                    break

                # ═══════════════════════════════════════════════════════
                # MORNING: Exit VAR hedges (09:20 – 09:30 window)
                # ═══════════════════════════════════════════════════════
                if (now_time >= self.strategy_var_exit_time and
                        now_time < dtime(9, 30, 0) and
                        not self.var_exit_done):
                    self._exit_var_hedges()
                    self.var_exit_done = True
                    self.var_done = False
                    refresh(3)
                    self.send_tagpos_screenshot()

                # ═══════════════════════════════════════════════════════
                # EVENING: Place VAR hedges (at strategy_var_entry_time, non-expiry)
                # ═══════════════════════════════════════════════════════
                if (now_time >= self.strategy_var_entry_time and
                        not self.var_done and
                        not self._is_expiry_day()):
                    self._enter_var_hedges()
                    refresh(3)
                    self.send_tagpos_screenshot()

                # ═══════════════════════════════════════════════════════
                # SIGNAL PROCESSING
                # ═══════════════════════════════════════════════════════
                signals = self.generate_signals()

                if signals:
                    entries = [s for s in signals if s.action == 'ENTRY']
                    exits = [s for s in signals if s.action == 'EXIT']

                    if exits:
                        self.logger.log(f"[EXIT] Processing {len(exits)} exits")
                        responses = self.execute_signals(exits)
                        self.exit_order_responses.update(responses)
                        refresh(2)
                        self.send_tagpos_screenshot()

                    if entries:
                        approved = self._filter_entries_by_margin(entries)

                        if approved:
                            self.logger.log(f"[ENTRY] Executing {len(approved)} approved entries")
                            responses = self.execute_signals(approved)
                            self.entry_order_responses.update(responses)
                            refresh(2)
                            self.send_tagpos_screenshot()

                # ═══════════════════════════════════════════════════════
                # RMS CHECK (every 30 seconds)
                # ═══════════════════════════════════════════════════════
                if (now_dt - self.last_rms_check).total_seconds() >= 30:
                    self.rms_check_margin_breach()
                    self.last_rms_check = now_dt

                # ═══════════════════════════════════════════════════════
                # SCREENSHOT (every N minutes)
                # ═══════════════════════════════════════════════════════
                if (now_dt.minute % self.tagpos_ss_timeframe == 0 and
                        now_dt >= self.tagpos_ss_timestamp):
                    self.tagpos_ss_timestamp = now_dt + timedelta(minutes=1)
                    self.send_tagpos_screenshot()

                time.sleep(self.loop_interval)

        except Exception as e:
            self.logger.log(f"[EXECUTE] Fatal error: {str(e)}", level='error')
            Alert.send_slack(
                message=f"[{self.tag_name}] Fatal error: {str(e)}",
                channel=self.slack_channel
            )
            raise

        finally:
            self.is_running = False
            try:
                self.engine.eod_shutdown()
            except:
                pass
            self.send_tagpos_screenshot()
            Alert.send_slack(
                message=f"[EXECUTE] {self.tag_name} stopped",
                channel=self.slack_channel
            )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
