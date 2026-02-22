// ============================================================
//  SOLANA PREDICTOR — app.js
//  Fetches live data, runs 5 signals, builds consensus
// ============================================================

// ---------- CONFIG ----------
const REFRESH_INTERVAL_MS = 60000; // refresh every 60 seconds

// ---------- STATE ----------
let previousPrice = null;

// ============================================================
//  UTILITY HELPERS
// ============================================================

function formatPrice(n) {
  return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatLargeNumber(n) {
  if (n >= 1_000_000_000) return '$' + (n / 1_000_000_000).toFixed(2) + 'B';
  if (n >= 1_000_000)     return '$' + (n / 1_000_000).toFixed(2) + 'M';
  return '$' + Number(n).toLocaleString();
}

function setEl(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setClass(id, cls) {
  const el = document.getElementById(id);
  if (el) { el.className = el.className.replace(/bullish|bearish|neutral/g, '').trim() + ' ' + cls; }
}

function now() {
  return new Date().toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ============================================================
//  API CALLS
// ============================================================

// --- Solana + Bitcoin price data from CoinGecko (free, no key needed) ---
async function fetchCryptoData() {
  try {
    const res = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=solana,bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_7d_change=true'
    );
    const data = await res.json();
    return data;
  } catch (e) {
    console.error('CoinGecko fetch failed:', e);
    return null;
  }
}

// --- Fear & Greed Index from Alternative.me (free, no key needed) ---
async function fetchFearGreed() {
  try {
    const res = await fetch('https://api.alternative.me/fng/?limit=1');
    const data = await res.json();
    return parseInt(data.data[0].value);
  } catch (e) {
    console.error('Fear & Greed fetch failed:', e);
    return null;
  }
}

// --- Historical SOL prices for RSI calculation (last 14 days) ---
async function fetchSOLHistory() {
  try {
    const res = await fetch(
      'https://api.coingecko.com/api/v3/coins/solana/market_chart?vs_currency=usd&days=14&interval=daily'
    );
    const data = await res.json();
    // Returns array of [timestamp, price]
    return data.prices.map(p => p[1]);
  } catch (e) {
    console.error('SOL history fetch failed:', e);
    return null;
  }
}

// ============================================================
//  SIGNAL CALCULATORS
// ============================================================

// ---------- SIGNAL 1: MOMENTUM ----------
// Uses 24h price change percentage
// > +3%  → bullish
// < -3%  → bearish
// between → neutral
function calcMomentum(change24h) {
  const verdict = change24h > 3 ? 'bullish' : change24h < -3 ? 'bearish' : 'neutral';
  const valueText = `24h change: ${change24h >= 0 ? '+' : ''}${change24h.toFixed(2)}%`;
  return { verdict, valueText };
}

// ---------- SIGNAL 2: VOLUME ----------
// Compares current 24h volume to a rough SOL average baseline
// If volume > 150% of baseline → bullish (unusual activity)
// If volume < 50% of baseline  → bearish (lack of interest)
// Otherwise → neutral
function calcVolume(vol24h) {
  const BASELINE_VOL = 2_500_000_000; // ~$2.5B is a typical quiet SOL day
  const ratio = vol24h / BASELINE_VOL;
  const verdict = ratio > 1.5 ? 'bullish' : ratio < 0.5 ? 'bearish' : 'neutral';
  const valueText = `24h volume: ${formatLargeNumber(vol24h)} (${(ratio * 100).toFixed(0)}% of baseline)`;
  return { verdict, valueText };
}

// ---------- SIGNAL 3: FEAR & GREED ----------
// 0–24   → Extreme Fear   → bullish (contrarian: market oversold)
// 25–44  → Fear           → bullish (leaning contrarian)
// 45–55  → Neutral        → neutral
// 56–75  → Greed          → bearish (market may be overheated)
// 76–100 → Extreme Greed  → bearish (strong contrarian sell signal)
function calcFearGreed(score) {
  let verdict, label;
  if (score <= 24)      { verdict = 'bullish'; label = 'Extreme Fear'; }
  else if (score <= 44) { verdict = 'bullish'; label = 'Fear'; }
  else if (score <= 55) { verdict = 'neutral';  label = 'Neutral'; }
  else if (score <= 75) { verdict = 'bearish'; label = 'Greed'; }
  else                  { verdict = 'bearish'; label = 'Extreme Greed'; }
  const valueText = `Score: ${score}/100 — ${label}`;
  return { verdict, valueText };
}

// ---------- SIGNAL 4: BTC CORRELATION ----------
// If BTC is up > 2% → SOL likely follows → bullish
// If BTC is down > 2% → SOL likely follows → bearish
// Otherwise → neutral
function calcBTCCorrelation(btcChange24h) {
  const verdict = btcChange24h > 2 ? 'bullish' : btcChange24h < -2 ? 'bearish' : 'neutral';
  const valueText = `BTC 24h change: ${btcChange24h >= 0 ? '+' : ''}${btcChange24h.toFixed(2)}%`;
  return { verdict, valueText };
}

// ---------- SIGNAL 5: RSI ----------
// Classic 14-period Relative Strength Index
// RSI > 70 → overbought → bearish
// RSI < 30 → oversold   → bullish
// 30–70    → neutral
function calcRSI(prices) {
  if (!prices || prices.length < 15) return { verdict: 'neutral', valueText: 'RSI: insufficient data', rsiValue: 50 };

  const changes = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  const gains = changes.map(c => c > 0 ? c : 0);
  const losses = changes.map(c => c < 0 ? Math.abs(c) : 0);

  const avgGain = gains.slice(-14).reduce((a, b) => a + b, 0) / 14;
  const avgLoss = losses.slice(-14).reduce((a, b) => a + b, 0) / 14;

  if (avgLoss === 0) return { verdict: 'bearish', valueText: 'RSI: 100 — Extremely overbought', rsiValue: 100 };

  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));

  let verdict;
  if (rsi > 70)      verdict = 'bearish';
  else if (rsi < 30) verdict = 'bullish';
  else               verdict = 'neutral';

  const valueText = `RSI (14-day): ${rsi.toFixed(1)} — ${rsi > 70 ? 'Overbought ⚠️' : rsi < 30 ? 'Oversold 💡' : 'Neutral zone'}`;
  return { verdict, valueText, rsiValue: rsi };
}

// ============================================================
//  CONSENSUS ENGINE
//  Collects all 5 verdicts, counts votes, builds explanation
// ============================================================

function buildConsensus(signals) {
  const counts = { bullish: 0, bearish: 0, neutral: 0 };
  signals.forEach(s => counts[s.verdict]++);

  let overallVerdict, confidence;

  const max = Math.max(counts.bullish, counts.bearish);

  if (counts.bullish > counts.bearish && counts.bullish >= 3) {
    overallVerdict = 'bullish';
    confidence = Math.round((counts.bullish / 5) * 100);
  } else if (counts.bearish > counts.bullish && counts.bearish >= 3) {
    overallVerdict = 'bearish';
    confidence = Math.round((counts.bearish / 5) * 100);
  } else {
    overallVerdict = 'neutral';
    confidence = Math.round(((5 - Math.abs(counts.bullish - counts.bearish)) / 5) * 60);
  }

  // Build a human-readable explanation of WHY the signals agree
  const agreedSignals = signals.filter(s => s.verdict === overallVerdict).map(s => s.name);
  const disagreedSignals = signals.filter(s => s.verdict !== overallVerdict).map(s => s.name);

  let explanation = '';

  if (overallVerdict === 'bullish') {
    explanation = `${agreedSignals.join(', ')} are all pointing upward. `;
    if (counts.bullish === 5) {
      explanation += 'This is a rare full agreement — all 5 signals are bullish simultaneously. While this increases confidence, crypto markets can still reverse unexpectedly. The alignment suggests genuine buying pressure across multiple dimensions.';
    } else if (counts.bullish === 4) {
      explanation += `${disagreedSignals.join(', ')} is the only signal not agreeing. Strong 4/5 bullish consensus suggests meaningful upward pressure.`;
    } else {
      explanation += `With only 3/5 signals bullish, this is a weak consensus. Proceed with extra caution — the market is mixed.`;
    }
  } else if (overallVerdict === 'bearish') {
    explanation = `${agreedSignals.join(', ')} are all pointing downward. `;
    if (counts.bearish === 5) {
      explanation += 'Full bearish agreement across all 5 signals — this is a strong warning sign. Multiple data sources simultaneously indicate downward pressure. This does not guarantee a drop but suggests significant risk.';
    } else if (counts.bearish === 4) {
      explanation += `${disagreedSignals.join(', ')} is the only signal not agreeing. Strong 4/5 bearish consensus suggests meaningful downward pressure.`;
    } else {
      explanation += `With only 3/5 signals bearish, this is a weak consensus. The market is sending mixed signals.`;
    }
  } else {
    explanation = `Signals are divided — ${counts.bullish} bullish, ${counts.bearish} bearish, ${counts.neutral} neutral. `;
    explanation += 'When signals disagree like this, the market is in an uncertain, undecided state. This often happens during consolidation periods before a larger move in either direction. No strong prediction can be made — this is a good time to wait and watch.';
  }

  return { overallVerdict, confidence, counts, explanation };
}

// ============================================================
//  UPDATE THE UI
// ============================================================

function updateSignalCard(id, signal) {
  const card = document.getElementById('card-' + id);
  const verdict = document.getElementById('verdict-' + id);
  const value = document.getElementById('value-' + id);

  if (card) card.className = 'signal-card ' + signal.verdict;

  if (verdict) {
    verdict.className = 'signal-verdict ' + signal.verdict;
    verdict.textContent = signal.verdict === 'bullish' ? '▲ BULLISH'
                        : signal.verdict === 'bearish' ? '▼ BEARISH'
                        : '● NEUTRAL';
  }

  if (value) value.textContent = signal.valueText;
}

function updateUI(cryptoData, fearGreedScore, solHistory) {
  // --- Price ---
  const sol = cryptoData?.solana;
  const btc = cryptoData?.bitcoin;

  if (sol) {
    const price = sol.usd;
    setEl('sol-price', formatPrice(price));

    const change = sol.usd_24h_change;
    const changeEl = document.getElementById('price-change');
    if (changeEl) {
      changeEl.textContent = `${change >= 0 ? '▲' : '▼'} ${Math.abs(change).toFixed(2)}% in last 24h`;
      changeEl.className = 'price-change ' + (change >= 0 ? 'up' : 'down');
    }
  }

  // --- Run all 5 signals ---
  const momentum  = { name: 'Momentum',        ...calcMomentum(sol?.usd_24h_change ?? 0) };
  const volume    = { name: 'Volume',           ...calcVolume(sol?.usd_24h_vol ?? 0) };
  const sentiment = { name: 'Fear & Greed',     ...calcFearGreed(fearGreedScore ?? 50) };
  const btcCorr   = { name: 'BTC Correlation',  ...calcBTCCorrelation(btc?.usd_24h_change ?? 0) };
  const rsi       = { name: 'RSI',              ...calcRSI(solHistory) };

  const allSignals = [momentum, volume, sentiment, btcCorr, rsi];

  // --- Update each signal card ---
  updateSignalCard('momentum',  momentum);
  updateSignalCard('volume',    volume);
  updateSignalCard('sentiment', sentiment);
  updateSignalCard('btc',       btcCorr);
  updateSignalCard('rsi',       rsi);

  // --- Build consensus ---
  const consensus = buildConsensus(allSignals);

  // Update consensus badge
  const badge = document.getElementById('consensus-badge');
  if (badge) {
    badge.className = 'consensus-badge ' + consensus.overallVerdict;
    badge.textContent = consensus.overallVerdict === 'bullish' ? '▲ BULLISH'
                      : consensus.overallVerdict === 'bearish' ? '▼ BEARISH'
                      : '● NEUTRAL';
  }

  // Update confidence bar
  const bar = document.getElementById('confidence-bar');
  if (bar) bar.style.width = consensus.confidence + '%';
  setEl('confidence-text', `Signal confidence: ${consensus.confidence}% — based on ${Math.max(consensus.counts.bullish, consensus.counts.bearish, consensus.counts.neutral)}/5 signals agreeing`);

  // Update vote counts
  setEl('bullish-count', consensus.counts.bullish + ' / 5');
  setEl('bearish-count', consensus.counts.bearish + ' / 5');
  setEl('neutral-count', consensus.counts.neutral + ' / 5');

  // Update consensus explanation
  setEl('breakdown-explanation', consensus.explanation);

  // Update last-updated timestamp
  setEl('last-updated', 'Last updated: ' + now());
}

// ============================================================
//  MAIN — fetch everything and run
// ============================================================

async function run() {
  setEl('last-updated', 'Fetching live data...');

  try {
    // Fetch all data sources in parallel (faster than one by one)
    const [cryptoData, fearGreedScore, solHistory] = await Promise.all([
      fetchCryptoData(),
      fetchFearGreed(),
      fetchSOLHistory()
    ]);

    updateUI(cryptoData, fearGreedScore, solHistory);

  } catch (err) {
    console.error('Run failed:', err);
    setEl('last-updated', 'Error fetching data — retrying in 60s');
  }
}

// ============================================================
//  INIT — run immediately, then repeat every 60 seconds
// ============================================================
run();
setInterval(run, REFRESH_INTERVAL_MS);