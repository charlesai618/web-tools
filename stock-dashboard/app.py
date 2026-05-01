from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import math

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


# ── Technical indicator helpers ──────────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line


def calculate_adx(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calculate_max_pain(calls, puts):
    all_strikes = sorted(
        set(calls['strike'].tolist() + puts['strike'].tolist())
    )
    min_pain = float('inf')
    max_pain_strike = all_strikes[0] if all_strikes else None

    for price in all_strikes:
        call_pain = (
            ((price - calls['strike']) * calls['openInterest'].fillna(0))
            .clip(lower=0).sum()
        )
        put_pain = (
            ((puts['strike'] - price) * puts['openInterest'].fillna(0))
            .clip(lower=0).sum()
        )
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = price

    return max_pain_strike


def safe(val, digits=2):
    try:
        if val is None:
            return None
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_atm_straddle(S, sigma, T, r=0.04):
    """Black-Scholes ATM straddle price (call + put). ATM = S equals K."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return None
    sqrtT = math.sqrt(T)
    d1 = (r + 0.5 * sigma ** 2) * T / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * norm_cdf(d1) - S * math.exp(-r * T) * norm_cdf(d2)
    put  = call - S + S * math.exp(-r * T)
    return call + put


def get_fear_greed():
    # Try CNN endpoint with full browser headers
    try:
        r = requests.get(
            'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Referer':    'https://www.cnn.com/markets/fear-and-greed',
                'Accept':     'application/json, text/plain, */*',
            },
            timeout=6,
        )
        if r.status_code == 200:
            d = r.json()['fear_and_greed']
            return safe(d['score'], 1), d.get('rating', ''), 'market'
    except Exception:
        pass

    # Fallback: alternative.me crypto Fear & Greed (widely-used proxy)
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
        d = r.json()['data'][0]
        return safe(float(d['value']), 1), d.get('value_classification', ''), 'crypto'
    except Exception:
        pass

    return None, None, None


def calculate_fibonacci(hist, current_price):
    """Fibonacci retracement levels from 1-year swing high/low."""
    current_price = float(current_price)   # ensure plain Python float
    high_price = float(hist['High'].max())
    low_price  = float(hist['Low'].min())
    high_idx   = hist['High'].idxmax()
    low_idx    = hist['Low'].idxmin()
    high_date  = high_idx.strftime('%Y-%m-%d')
    low_date   = low_idx.strftime('%Y-%m-%d')
    uptrend    = bool(low_idx < high_idx)  # low came before high → uptrend

    diff = high_price - low_price
    levels = []
    for ratio, label in [
        (0.000, '0%'),
        (0.236, '23.6%'),
        (0.382, '38.2%'),
        (0.500, '50%'),
        (0.618, '61.8%'),
        (0.764, '76.4%'),
        (1.000, '100%'),
        (1.272, '127.2%'),
        (1.618, '161.8%'),
    ]:
        price = high_price - ratio * diff
        levels.append({'ratio': ratio, 'label': label, 'price': round(price, 2)})

    levels.sort(key=lambda x: -x['price'])  # high → low

    for lv in levels:
        lv['above_price'] = bool(lv['price'] > current_price)

    resistances = [lv for lv in levels if lv['price'] > current_price]
    supports    = [lv for lv in levels if lv['price'] <= current_price]

    return {
        'swing_high':         round(high_price, 2),
        'swing_low':          round(low_price, 2),
        'high_date':          high_date,
        'low_date':           low_date,
        'trend':              'uptrend' if uptrend else 'downtrend',
        'levels':             levels,
        'nearest_resistance': resistances[-1] if resistances else None,
        'nearest_support':    supports[0]    if supports    else None,
        'current_price':      round(current_price, 2),
    }


def calculate_atm_history(hist, today_price, default_hv=30):
    """Theoretical 1-day ATM straddle price and realized move for 90 days,
    plus one extra point for today (straddle only, no realized move yet).
    Also sends extended_closes so the frontend can recalculate with any HV window.
    """
    import datetime as dt
    close = hist['Close']
    MAX_HV = 60  # frontend supports up to this window

    # Extended closes: MAX_HV warmup + 90 display days + 1 for last realized move
    needed     = MAX_HV + 91
    ec_series  = close.iloc[-needed:] if len(close) >= needed else close
    extended_closes = [safe(float(v)) for v in ec_series.tolist()]

    # Dates and realized moves for the 90-day display window (fixed regardless of HV)
    tail_close = close.iloc[-91:]
    n          = min(90, len(tail_close) - 1)
    dates      = [tail_close.index[i].strftime('%Y-%m-%d') for i in range(n)]
    realized   = [round(abs(float(tail_close.iloc[i + 1]) - float(tail_close.iloc[i])), 2)
                  for i in range(n)]

    # Default straddles using default_hv-day rolling HV (for first render)
    log_ret    = np.log(close / close.shift(1))
    hv_default = log_ret.rolling(default_hv).std() * np.sqrt(252)
    tail_hv    = hv_default.iloc[-91:]
    T          = 1.0 / 252
    r          = 0.04
    straddles  = []
    for i in range(n):
        S     = float(tail_close.iloc[i])
        sigma = float(tail_hv.iloc[i])
        if not np.isnan(sigma) and sigma > 0:
            st = bs_atm_straddle(S, sigma, T, r)
            straddles.append(round(st, 2) if st else None)
        else:
            straddles.append(None)

    # Today's point: straddle only (realized unknown)
    today_date = dt.date.today().strftime('%Y-%m-%d')
    today_sigma = float(hv_default.iloc[-1]) if not np.isnan(hv_default.iloc[-1]) else None
    today_straddle = None
    if today_sigma and today_sigma > 0:
        st = bs_atm_straddle(float(today_price), today_sigma, T, r)
        today_straddle = round(st, 2) if st else None

    return {
        'dates':            dates,
        'straddles':        straddles,       # default HV window, historical only
        'realized':         realized,
        'extended_closes':  extended_closes, # for client-side HV recalculation
        'default_hv':       default_hv,
        'max_hv':           MAX_HV,
        'today_date':       today_date,
        'today_price':      float(today_price),
        'today_straddle':   today_straddle,  # default HV, no realized
    }


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stock/<ticker>')
def get_stock_data(ticker):
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        hist = t.history(period='1y')
        if hist.empty:
            return jsonify({'error': f'No data found for "{ticker}". Check the symbol.'}), 404

        close = hist['Close']
        high  = hist['High']
        low   = hist['Low']

        current  = close.iloc[-1]
        previous = close.iloc[-2] if len(close) > 1 else current
        chg      = current - previous
        chg_pct  = (chg / previous) * 100

        # ── Moving averages ──────────────────────────────────────────────────
        buy_count = sell_count = 0
        moving_averages = {}
        for p in [5, 10, 20, 50, 100, 200]:
            if len(close) >= p:
                val = close.rolling(p).mean().iloc[-1]
                sig = 'BUY' if current > val else 'SELL'
                buy_count  += sig == 'BUY'
                sell_count += sig == 'SELL'
                moving_averages[f'MA{p}'] = {'value': safe(val), 'signal': sig}

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_series  = calculate_rsi(close)
        current_rsi = safe(rsi_series.iloc[-1])
        if current_rsi is None:
            rsi_signal = 'N/A'
        elif current_rsi < 30:
            rsi_signal = 'OVERSOLD'
            sell_count += 1
        elif current_rsi > 70:
            rsi_signal = 'OVERBOUGHT'
            buy_count  += 1
        else:
            rsi_signal = 'NEUTRAL'
            if current_rsi < 50: sell_count += 1
            else: buy_count += 1

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_s, sig_s, hist_s = calculate_macd(close)
        cur_macd = safe(macd_s.iloc[-1])
        cur_sig  = safe(sig_s.iloc[-1])
        cur_hist = safe(hist_s.iloc[-1])
        macd_signal = 'BUY' if (cur_macd and cur_sig and cur_macd > cur_sig) else 'SELL'
        buy_count  += macd_signal == 'BUY'
        sell_count += macd_signal == 'SELL'

        # ── ADX ──────────────────────────────────────────────────────────────
        adx_s, plus_di_s, minus_di_s = calculate_adx(high, low, close)
        cur_adx      = safe(adx_s.iloc[-1])
        cur_plus_di  = safe(plus_di_s.iloc[-1])
        cur_minus_di = safe(minus_di_s.iloc[-1])
        adx_trend    = 'Strong Trend' if (cur_adx and cur_adx > 25) else 'Weak / No Trend'

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # ── Overall signal ───────────────────────────────────────────────────
        total = buy_count + sell_count
        ratio = sell_count / total if total else 0.5
        if   ratio >= 0.75: overall = 'STRONG SELL'
        elif ratio >= 0.5:  overall = 'SELL'
        elif ratio <= 0.25: overall = 'STRONG BUY'
        elif ratio < 0.5:   overall = 'BUY'
        else:               overall = 'NEUTRAL'

        # ── Options ──────────────────────────────────────────────────────────
        options_data = {}
        try:
            expirations = t.options
            if expirations:
                chain  = t.option_chain(expirations[0])
                calls  = chain.calls
                puts   = chain.puts

                call_oi = calls['openInterest'].fillna(0).sum()
                put_oi  = puts['openInterest'].fillna(0).sum()
                pcr     = safe(put_oi / call_oi) if call_oi > 0 else None

                call_iv = calls['impliedVolatility'].dropna().mean()
                put_iv  = puts['impliedVolatility'].dropna().mean()
                avg_iv  = safe(((call_iv + put_iv) / 2) * 100)

                max_pain = calculate_max_pain(calls, puts)

                if   pcr and pcr > 1.2: pcr_signal = 'VERY BEARISH'
                elif pcr and pcr > 1.0: pcr_signal = 'BEARISH'
                elif pcr and pcr < 0.5: pcr_signal = 'VERY BULLISH'
                elif pcr and pcr < 0.7: pcr_signal = 'BULLISH'
                else:                   pcr_signal = 'NEUTRAL'

                # Current ATM option (nearest expiry)
                atm_strike_val = atm_call_mid = atm_put_mid = atm_straddle_val = atm_iv_val = None
                try:
                    atm_idx   = (calls['strike'] - current).abs().idxmin()
                    atm_strike_val = float(calls.loc[atm_idx, 'strike'])
                    c_row     = calls[calls['strike'] == atm_strike_val].iloc[0]
                    p_rows    = puts[puts['strike'] == atm_strike_val]
                    c_bid = float(c_row.get('bid', 0) or 0)
                    c_ask = float(c_row.get('ask', 0) or 0)
                    atm_call_mid = (c_bid + c_ask) / 2 if c_ask > 0 else float(c_row.get('lastPrice', 0) or 0)
                    if not p_rows.empty:
                        p_row  = p_rows.iloc[0]
                        p_bid  = float(p_row.get('bid', 0) or 0)
                        p_ask  = float(p_row.get('ask', 0) or 0)
                        atm_put_mid = (p_bid + p_ask) / 2 if p_ask > 0 else float(p_row.get('lastPrice', 0) or 0)
                    atm_straddle_val = (atm_call_mid or 0) + (atm_put_mid or 0)
                    atm_iv_val = safe(float(c_row.get('impliedVolatility', 0) or 0) * 100, 1)
                except Exception:
                    pass

                options_data = {
                    'put_call_ratio':      pcr,
                    'pcr_signal':          pcr_signal,
                    'implied_volatility':  avg_iv,
                    'iv_signal':           'HIGH' if (avg_iv and avg_iv > 20) else ('LOW' if (avg_iv and avg_iv < 10) else 'NORMAL'),
                    'call_oi':             int(call_oi),
                    'put_oi':              int(put_oi),
                    'max_pain':            safe(max_pain),
                    'nearest_expiry':      expirations[0],
                    'expirations_count':   len(expirations),
                    'atm_strike':          safe(atm_strike_val),
                    'atm_call_mid':        safe(atm_call_mid),
                    'atm_put_mid':         safe(atm_put_mid),
                    'atm_straddle':        safe(atm_straddle_val),
                    'atm_iv':              atm_iv_val,
                }
        except Exception as e:
            options_data = {'error': str(e)}

        # ── Fear & Greed ─────────────────────────────────────────────────────
        fg_score, fg_rating, fg_source = get_fear_greed()

        # ── Fibonacci retracement ─────────────────────────────────────────────
        fibonacci = calculate_fibonacci(hist, current)

        # ── ATM option history (theoretical 1-day straddles) ──────────────────
        atm_history = calculate_atm_history(hist, current)

        # ── Chart data (90 days) ─────────────────────────────────────────────
        def to_list(s):
            return [safe(v) for v in s.tolist()]

        h90 = hist.tail(90)

        chart_data = {
            'dates':       [d.strftime('%Y-%m-%d') for d in h90.index],
            'close':       to_list(h90['Close']),
            'volume':      [int(v) for v in h90['Volume'].tolist()],
            'ma20':        to_list(close.rolling(20).mean().tail(90)),
            'ma50':        to_list(close.rolling(50).mean().tail(90)),
            'ma200':       to_list(close.rolling(200).mean().tail(90)),
            'bb_upper':    to_list(bb_upper.tail(90)),
            'bb_lower':    to_list(bb_lower.tail(90)),
            'bb_mid':      to_list(bb_mid.tail(90)),
            'rsi':         to_list(rsi_series.tail(90)),
            'macd':        to_list(macd_s.tail(90)),
            'macd_signal': to_list(sig_s.tail(90)),
            'macd_hist':   to_list(hist_s.tail(90)),
        }

        return jsonify({
            'ticker':        ticker,
            'name':          info.get('longName') or info.get('shortName') or ticker,
            'sector':        info.get('sector') or info.get('category') or 'N/A',
            'price':         safe(current),
            'change':        safe(chg),
            'change_pct':    safe(chg_pct),
            'prev_close':    safe(info.get('previousClose') or previous),
            'open':          safe(info.get('open') or h90['Open'].iloc[-1]),
            'day_low':       safe(info.get('dayLow')  or h90['Low'].iloc[-1]),
            'day_high':      safe(info.get('dayHigh') or h90['High'].iloc[-1]),
            'week_52_low':   safe(info.get('fiftyTwoWeekLow')),
            'week_52_high':  safe(info.get('fiftyTwoWeekHigh')),
            'market_cap':    info.get('marketCap'),
            'pe_ratio':      safe(info.get('trailingPE') or info.get('forwardPE')),
            'dividend_yield':safe((info.get('dividendYield') or 0) * 100),
            'volume':        info.get('volume') or int(h90['Volume'].iloc[-1]),
            'avg_volume':    info.get('averageVolume'),
            'moving_averages': moving_averages,
            'buy_count':     buy_count,
            'sell_count':    sell_count,
            'overall_signal':overall,
            'rsi':   {'value': current_rsi, 'signal': rsi_signal},
            'macd':  {'value': cur_macd, 'signal_line': cur_sig, 'histogram': cur_hist, 'signal': macd_signal},
            'adx':   {'value': cur_adx, 'plus_di': cur_plus_di, 'minus_di': cur_minus_di, 'trend': adx_trend},
            'options':     options_data,
            'fear_greed':  {'score': fg_score, 'rating': fg_rating, 'source': fg_source},
            'fibonacci':   fibonacci,
            'atm_history': atm_history,
            'chart_data':  chart_data,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/<ticker>')
def get_options(ticker):
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return jsonify({'error': 'No options available'})

        chain = t.option_chain(expirations[0])
        calls, puts = chain.calls, chain.puts

        call_oi = calls['openInterest'].fillna(0).sum()
        put_oi  = puts['openInterest'].fillna(0).sum()
        pcr     = safe(put_oi / call_oi) if call_oi > 0 else None

        avg_iv = safe(((calls['impliedVolatility'].dropna().mean() +
                        puts['impliedVolatility'].dropna().mean()) / 2) * 100)

        if   pcr and pcr > 1.2: pcr_signal = 'VERY BEARISH'
        elif pcr and pcr > 1.0: pcr_signal = 'BEARISH'
        elif pcr and pcr < 0.5: pcr_signal = 'VERY BULLISH'
        elif pcr and pcr < 0.7: pcr_signal = 'BULLISH'
        else:                   pcr_signal = 'NEUTRAL'

        current = float(t.fast_info.last_price or 0)
        atm_strike_val = atm_call_mid = atm_put_mid = atm_straddle_val = atm_iv_val = None
        try:
            atm_idx        = (calls['strike'] - current).abs().idxmin()
            atm_strike_val = float(calls.loc[atm_idx, 'strike'])
            c_row          = calls[calls['strike'] == atm_strike_val].iloc[0]
            p_rows         = puts[puts['strike'] == atm_strike_val]
            c_bid, c_ask   = float(c_row.get('bid', 0) or 0), float(c_row.get('ask', 0) or 0)
            atm_call_mid   = (c_bid + c_ask) / 2 if c_ask > 0 else float(c_row.get('lastPrice', 0) or 0)
            if not p_rows.empty:
                p_row = p_rows.iloc[0]
                p_bid, p_ask = float(p_row.get('bid', 0) or 0), float(p_row.get('ask', 0) or 0)
                atm_put_mid = (p_bid + p_ask) / 2 if p_ask > 0 else float(p_row.get('lastPrice', 0) or 0)
            atm_straddle_val = (atm_call_mid or 0) + (atm_put_mid or 0)
            atm_iv_val       = safe(float(c_row.get('impliedVolatility', 0) or 0) * 100, 1)
        except Exception:
            pass

        return jsonify({
            'put_call_ratio':     pcr,
            'pcr_signal':         pcr_signal,
            'implied_volatility': avg_iv,
            'iv_signal':          'HIGH' if (avg_iv and avg_iv > 20) else ('LOW' if (avg_iv and avg_iv < 10) else 'NORMAL'),
            'call_oi':            int(call_oi),
            'put_oi':             int(put_oi),
            'max_pain':           safe(calculate_max_pain(calls, puts)),
            'nearest_expiry':     expirations[0],
            'expirations_count':  len(expirations),
            'atm_strike':         safe(atm_strike_val),
            'atm_call_mid':       safe(atm_call_mid),
            'atm_put_mid':        safe(atm_put_mid),
            'atm_straddle':       safe(atm_straddle_val),
            'atm_iv':             atm_iv_val,
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
