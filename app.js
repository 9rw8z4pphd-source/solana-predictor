// ============================================================
//  SOLANA PREDICTOR v2 — app.js
//  7 signals · live chart · history log · consensus engine
// ============================================================

const REFRESH_MS  = 60000;
const CIRCUMFERENCE = 314; // 2 * PI * 50 (ring radius)

let priceChart    = null;
let historyLog    = [];
let countdown     = 60;
let countdownTimer = null;

// ============================================================
//  UTILITIES
// ============================================================
const $ = id => document.getElementById(id);
const setText = (id, val) => { const el = $(id); if (el) el.textContent = val; };

function formatPrice(n) {
  return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function formatBig(n) {
  if (n >= 1e12) return '$' + (n/1e12).toFixed(2) + 'T';
  if (n >= 1e9)  return '$' + (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6)  return '$' + (n/1e6).toFixed(2) + 'M';
  return '$' + Number(n).toLocaleString();
}
function formatPct(n) {
  return (n >= 0 ? '+' : '') + Number(n).toFixed(2) + '%';
}
function timeNow() {
  return new Date().toLocaleTimeString('en-AU', { hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

// ============================================================
//  COUNTDOWN RING
// ============================================================
function startCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  countdown = 60;
  countdownTimer = setInterval(() => {
    countdown--;
    if (countdown < 0) countdown = 60;
    setText('refresh-label', countdown + 's');
    const offset = CIRCUMFERENCE - (countdown / 60) * 94.2;
    const ring = $('ring-fill');
    if (ring) ring.style.strokeDashoffset = offset;
  }, 1000);
}

// ============================================================
//  API CALLS
// ============================================================
async function fetchCryptoData() {
  try {
    const r = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=solana,bitcoin&vs_currencies=usd' +
      '&include_24hr_change=true&include_24hr_vol=true&include_7d_change=true&include_market_cap=true'
    );
    return await r.json();
  } catch(e) { console.error('CoinGecko price error:', e); return null; }
}

async function fetchSOLDetails() {
  try {
    const r = await fetch(
      'https://api.coingecko.com/api/v3/coins/solana?localization=false&tickers=false&community_data=false&developer_data=false'
    );
    return await r.json();
  } catch(e) { console.error('SOL details error:', e); return null; }
}

async function fetchFearGreed() {
  try {
    const r = await fetch('https://api.alternative.me/fng/?limit=1');
    const d = await r.json();
    return parseInt(d.data[0].value);
  } catch(e) { console.error('Fear&Greed error:', e); return 50; }
}

async function fetchSOLHistory() {
  try {
    const r = await fetch(
      'https://api.coingecko.com/api/v3/coins/solana/market_chart?vs_currency=usd&days=14&interval=daily'
    );
    const d = await r.json();
    return { prices: d.prices.map(p=>p[1]), timestamps: d.prices.map(p=>p[0]) };
  } catch(e) { console.error('SOL history error:', e); return null; }
}

// ============================================================
//  SIGNAL CALCULATORS
// ============================================================

function sigMomentum(change24h) {
  const v = change24h ?? 0;
  const verdict = v > 3 ? 'bullish' : v < -3 ? 'bearish' : 'neutral';
  const why = v > 3
    ? `Price is up ${formatPct(v)} — strong buying pressure over 24h`
    : v < -3
    ? `Price is down ${formatPct(v)} — sellers are dominating over 24h`
    : `Price moved ${formatPct(v)} — within neutral range (±3%)`;
  return { name:'Momentum', verdict, valueText:`24h change: ${formatPct(v)}`, why };
}

function sigVolume(vol24h) {
  const BASELINE = 2_500_000_000;
  const ratio = (vol24h ?? 0) / BASELINE;
  const verdict = ratio > 1.5 ? 'bullish' : ratio < 0.5 ? 'bearish' : 'neutral';
  const why = ratio > 1.5
    ? `Volume is ${(ratio*100).toFixed(0)}% of baseline — unusually high activity signals potential big move`
    : ratio < 0.5
    ? `Volume is only ${(ratio*100).toFixed(0)}% of baseline — market is quiet and directionless`
    : `Volume at ${(ratio*100).toFixed(0)}% of baseline — normal trading activity`;
  return { name:'Volume', verdict, valueText:`24h vol: ${formatBig(vol24h)} (${(ratio*100).toFixed(0)}% of $2.5B baseline)`, why };
}

function sigFearGreed(score) {
  score = score ?? 50;
  let verdict, label;
  if      (score <= 24) { verdict='bullish'; label='Extreme Fear — contrarian buy signal'; }
  else if (score <= 44) { verdict='bullish'; label='Fear — market may be oversold'; }
  else if (score <= 55) { verdict='neutral';  label='Neutral — no strong emotional extreme'; }
  else if (score <= 75) { verdict='bearish'; label='Greed — market getting overheated'; }
  else                  { verdict='bearish'; label='Extreme Greed — potential reversal risk'; }
  const why = `Score ${score}/100: ${label}`;
  return { name:'Fear & Greed', verdict, valueText:`Fear & Greed Index: ${score}/100`, why };
}

function sigBTC(btcChange) {
  const v = btcChange ?? 0;
  const verdict = v > 2 ? 'bullish' : v < -2 ? 'bearish' : 'neutral';
  const why = v > 2
    ? `BTC is up ${formatPct(v)} — SOL typically follows Bitcoin's upward moves`
    : v < -2
    ? `BTC is down ${formatPct(v)} — Bitcoin weakness tends to drag SOL lower`
    : `BTC moved ${formatPct(v)} — within neutral range, correlation signal is weak`;
  return { name:'BTC Correlation', verdict, valueText:`BTC 24h: ${formatPct(v)}`, why };
}

function sigRSI(prices) {
  if (!prices || prices.length < 15) return { name:'RSI', verdict:'neutral', valueText:'RSI: not enough data', why:'Need 14+ days of data to calculate RSI' };
  const changes = prices.slice(-15).map((p,i,a) => i===0 ? 0 : p-a[i-1]).slice(1);
  const gains = changes.map(c => c>0 ? c : 0);
  const losses = changes.map(c => c<0 ? Math.abs(c) : 0);
  const avgG = gains.reduce((a,b)=>a+b,0)/14;
  const avgL = losses.reduce((a,b)=>a+b,0)/14;
  if (avgL === 0) return { name:'RSI', verdict:'bearish', valueText:'RSI: 100 — Extremely overbought', why:'No losing days in 14 periods — heavily overbought' };
  const rsi = 100 - (100/(1+(avgG/avgL)));
  const verdict = rsi > 70 ? 'bearish' : rsi < 30 ? 'bullish' : 'neutral';
  const why = rsi > 70
    ? `RSI ${rsi.toFixed(1)} — above 70 means overbought, price may pull back soon`
    : rsi < 30
    ? `RSI ${rsi.toFixed(1)} — below 30 means oversold, price may bounce upward`
    : `RSI ${rsi.toFixed(1)} — in neutral zone between 30 and 70`;
  return { name:'RSI', verdict, valueText:`RSI (14-day): ${rsi.toFixed(1)}`, why };
}

function sigTrend(prices, currentPrice) {
  if (!prices || prices.length < 7) return { name:'7D Trend', verdict:'neutral', valueText:'Trend: insufficient data', why:'Not enough price history to calculate trend' };
  const last7 = prices.slice(-7);
  const avg = last7.reduce((a,b)=>a+b,0) / last7.length;
  const pctAbove = ((currentPrice - avg) / avg) * 100;
  const verdict = pctAbove > 3 ? 'bullish' : pctAbove < -3 ? 'bearish' : 'neutral';
  const why = pctAbove > 3
    ? `Current price is ${formatPct(pctAbove)} above the 7-day average — bullish weekly trend`
    : pctAbove < -3
    ? `Current price is ${formatPct(pctAbove)} below the 7-day average — bearish weekly trend`
    : `Current price is ${formatPct(pctAbove)} from the 7-day average — no strong trend`;
  return { name:'7D Trend', verdict, valueText:`Price vs 7d avg: ${formatPct(pctAbove)} (avg: ${formatPrice(avg)})`, why };
}

function sigVolatility(prices) {
  if (!prices || prices.length < 7) return { name:'Volatility', verdict:'neutral', valueText:'Volatility: insufficient data', why:'Not enough data to measure volatility' };
  const last7 = prices.slice(-7);
  const pctChanges = last7.slice(1).map((p,i) => Math.abs((p-last7[i])/last7[i]*100));
  const avgSwing = pctChanges.reduce((a,b)=>a+b,0) / pctChanges.length;
  let verdict, label;
  if (avgSwing > 8)      { verdict='neutral'; label='Extreme volatility — signals are unreliable'; }
  else if (avgSwing > 4) { verdict='neutral'; label='High volatility — predictions less certain'; }
  else if (avgSwing < 1.5){ verdict='bullish'; label='Low volatility — market is calm and stable'; }
  else                   { verdict='neutral'; label='Normal volatility range'; }
  const why = `Average daily swing of ${avgSwing.toFixed(1)}% over 7 days — ${label}`;
  return { name:'Volatility', verdict, valueText:`Avg daily swing: ±${avgSwing.toFixed(1)}%`, why };
}

// ============================================================
//  CONSENSUS ENGINE
// ============================================================
function buildConsensus(signals) {
  const counts = { bullish:0, bearish:0, neutral:0 };
  signals.forEach(s => counts[s.verdict]++);
  const total = signals.length;

  let verdict, confidence;
  if (counts.bullish > counts.bearish && counts.bullish > counts.neutral) {
    verdict = 'bullish';
    confidence = Math.round((counts.bullish/total)*100);
  } else if (counts.bearish > counts.bullish && counts.bearish > counts.neutral) {
    verdict = 'bearish';
    confidence = Math.round((counts.bearish/total)*100);
  } else {
    verdict = 'neutral';
    confidence = Math.round(((total - Math.abs(counts.bullish-counts.bearish)) / total) * 55);
  }

  const agreed = signals.filter(s=>s.verdict===verdict).map(s=>s.name);
  const opposed = signals.filter(s=>s.verdict!==verdict).map(s=>s.name);

  let explanation = '';
  const strength = counts[verdict];

  if (verdict === 'bullish') {
    if (strength >= 6) explanation = `Very strong bullish signal — ${agreed.join(', ')} are all pointing upward simultaneously. This rare alignment across ${strength}/7 signals suggests genuine broad-based buying pressure from multiple market dimensions.`;
    else if (strength >= 5) explanation = `Strong bullish consensus — ${agreed.join(', ')} agree on upward pressure. ${opposed.join(', ')} ${opposed.length===1?'is':'are'} not confirming, adding slight uncertainty.`;
    else if (strength >= 4) explanation = `Moderate bullish lean — ${agreed.join(', ')} suggest upward momentum but ${opposed.join(', ')} disagree. A real signal, but not overwhelming.`;
    else explanation = `Weak bullish lean — only ${strength}/7 signals agree. Market is mixed. Treat with caution.`;
  } else if (verdict === 'bearish') {
    if (strength >= 6) explanation = `Very strong bearish signal — ${agreed.join(', ')} all point downward. This broad agreement across ${strength}/7 signals suggests significant downward pressure across multiple market dimensions.`;
    else if (strength >= 5) explanation = `Strong bearish consensus — ${agreed.join(', ')} align on downward pressure. ${opposed.join(', ')} ${opposed.length===1?'is':'are'} not confirming.`;
    else if (strength >= 4) explanation = `Moderate bearish lean — ${agreed.join(', ')} suggest downward momentum but ${opposed.join(', ')} disagree.`;
    else explanation = `Weak bearish lean — only ${strength}/7 signals agree. Market is mixed.`;
  } else {
    explanation = `Signals are divided: ${counts.bullish} bullish, ${counts.bearish} bearish, ${counts.neutral} neutral. When signals disagree this clearly, the market is in an undecided state — often consolidating before a larger move. No reliable prediction can be made right now.`;
  }

  return { verdict, confidence, counts, explanation };
}

// ============================================================
//  CHART
// ============================================================
function buildChart(history) {
  if (!history || !history.prices) return;

  const labels = history.timestamps.slice(-7).map(ts =>
    new Date(ts).toLocaleDateString('en-AU', { month:'short', day:'numeric' })
  );
  const prices = history.prices.slice(-7);

  const ctx = document.getElementById('price-chart');
  if (!ctx) return;

  if (priceChart) priceChart.destroy();

  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'SOL/USD',
        data: prices,
        borderColor: '#9945ff',
        borderWidth: 2.5,
        pointBackgroundColor: '#9945ff',
        pointBorderColor: '#04040f',
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 8,
        fill: true,
        backgroundColor: (ctx) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 300);
          gradient.addColorStop(0, 'rgba(153,69,255,0.25)');
          gradient.addColorStop(1, 'rgba(153,69,255,0.01)');
          return gradient;
        },
        tension: 0.4,
      }]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0e0e24',
          borderColor: '#1a1a3a',
          borderWidth: 1,
          titleColor: '#9999bb',
          bodyColor: '#ffffff',
          padding: 12,
          callbacks: {
            label: ctx => ' ' + formatPrice(ctx.parsed.y)
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#555577', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#555577',
            font: { size: 11 },
            callback: v => '$' + v.toLocaleString()
          }
        }
      }
    }
  });
}

// ============================================================
//  UI UPDATERS
// ============================================================
function updateSignalCard(id, signal) {
  const card     = $('card-'+id);
  const verdict  = $('verdict-'+id);
  const value    = $('value-'+id);
  const why      = $('why-'+id);

  if (card)    card.className = 'signal-card ' + signal.verdict;
  if (verdict) {
    verdict.className = 'signal-verdict ' + signal.verdict;
    verdict.textContent = signal.verdict === 'bullish' ? '▲ BULLISH'
                        : signal.verdict === 'bearish' ? '▼ BEARISH' : '● NEUTRAL';
  }
  if (value) value.textContent  = signal.valueText;
  if (why)   why.textContent    = '→ ' + signal.why;
}

function updateHistoryLog(price, consensus) {
  const entry = {
    time:       timeNow(),
    price:      formatPrice(price),
    verdict:    consensus.verdict,
    confidence: consensus.confidence + '%',
    votes:      `${consensus.counts.bullish}B ${consensus.counts.bearish}Ba ${consensus.counts.neutral}N`
  };
  historyLog.unshift(entry);
  if (historyLog.length > 10) historyLog.pop();

  const log = $('history-log');
  if (!log) return;
  log.innerHTML = historyLog.map(e => `
    <div class="history-row">
      <span>${e.time}</span>
      <span>${e.price}</span>
      <span class="${e.verdict}">${e.verdict.toUpperCase()}</span>
      <span>${e.confidence}</span>
      <span style="color:var(--text-3)">${e.votes}</span>
    </div>
  `).join('');
}

function animateRing(confidence, verdict) {
  const ring = $('ring-progress');
  if (!ring) return;
  ring.className = 'ring-progress ' + verdict;
  const offset = CIRCUMFERENCE - (confidence / 100) * CIRCUMFERENCE;
  ring.style.strokeDashoffset = offset;
}

// ============================================================
//  MAIN UPDATE
// ============================================================
function updateUI(crypto, fearGreed, history, details) {
  const sol = crypto?.solana;
  const btc = crypto?.bitcoin;
  const price = sol?.usd ?? 0;

  // --- Market bar ---
  setText('market-cap',  sol ? formatBig(sol.usd_market_cap) : '--');
  setText('market-vol',  sol ? formatBig(sol.usd_24h_vol)    : '--');
  setText('market-rank', details ? '#' + details.market_cap_rank : '--');
  setText('btc-price',   btc ? formatPrice(btc.usd) : '--');

  const change7d = sol?.usd_7d_change;
  const el7d = $('market-7d');
  if (el7d && change7d != null) {
    el7d.textContent = formatPct(change7d);
    el7d.className = 'market-stat-value ' + (change7d >= 0 ? 'up' : 'down');
  }

  // --- Hero price ---
  setText('sol-price', formatPrice(price));

  const c24 = sol?.usd_24h_change;
  const p24 = $('change-24h');
  if (p24 && c24 != null) {
    p24.textContent = '24h ' + formatPct(c24);
    p24.className = 'price-change-pill ' + (c24 >= 0 ? 'up' : 'down');
  }
  const p7 = $('change-7d');
  if (p7 && change7d != null) {
    p7.textContent = '7d ' + formatPct(change7d);
    p7.className = 'price-change-pill ' + (change7d >= 0 ? 'up' : 'down');
  }

  // --- Run 7 signals ---
  const signals = [
    sigMomentum(c24),
    sigVolume(sol?.usd_24h_vol),
    sigFearGreed(fearGreed),
    sigBTC(btc?.usd_24h_change),
    sigRSI(history?.prices),
    sigTrend(history?.prices, price),
    sigVolatility(history?.prices),
  ];

  updateSignalCard('momentum',   signals[0]);
  updateSignalCard('volume',     signals[1]);
  updateSignalCard('sentiment',  signals[2]);
  updateSignalCard('btc',        signals[3]);
  updateSignalCard('rsi',        signals[4]);
  updateSignalCard('trend',      signals[5]);
  updateSignalCard('volatility', signals[6]);

  // --- Consensus ---
  const consensus = buildConsensus(signals);

  animateRing(consensus.confidence, consensus.verdict);

  const rv = $('ring-verdict');
  if (rv) {
    rv.className = 'ring-verdict ' + consensus.verdict;
    rv.textContent = consensus.verdict.toUpperCase();
  }
  setText('ring-pct', consensus.confidence + '%');
  setText('confidence-text', `${consensus.counts[consensus.verdict]} of 7 signals agree`);

  setText('hero-bullish', consensus.counts.bullish);
  setText('hero-bearish', consensus.counts.bearish);
  setText('hero-neutral',  consensus.counts.neutral);
  setText('breakdown-explanation', consensus.explanation);

  // --- History ---
  updateHistoryLog(price, consensus);

  // --- Chart ---
  buildChart(history);

  // --- Timestamp ---
  setText('last-updated', 'Updated ' + timeNow());
}

// ============================================================
//  RUN
// ============================================================
async function run() {
  setText('last-updated', 'Fetching live data...');
  try {
    const [crypto, fearGreed, history, details] = await Promise.all([
      fetchCryptoData(),
      fetchFearGreed(),
      fetchSOLHistory(),
      fetchSOLDetails(),
    ]);
    updateUI(crypto, fearGreed, history, details);
  } catch(e) {
    console.error('Run error:', e);
    setText('last-updated', 'Error — retrying in 60s');
  }
  startCountdown();
}

run();
setInterval(run, REFRESH_MS);