// Poll-average and runoff-projection model, shared by index.html and backtest/backtest.cjs.
// Polls are { pollster, key, end (day number), n, lula } where `lula` is candidate A's
// two-way share in pp (Lula in 2026; the first runoff candidate in historical tests).
(function (root) {
  const DEFAULTS = {
    windowDays: 14,   // polls older than this drop out of the average
    halfLife: 7,      // recency weight halves every N days
    nCap: 4000,       // sample-size influence capped before sqrt weighting
    minPolls: 4,      // post-1st-round polls needed before projecting
    bw: 14,           // LOESS bandwidth (days)
    alpha: 0.5, beta: 0.3, phi: 0.85,  // damped Holt
    q: 0.1, house: 1.5,                // Kalman: daily drift variance (pp²), house-effect sd (pp)
    minErr: 1.5,      // floor for the 1st-round poll error (pp)
    // Ensemble: method weights and the error model used for the win probability.
    weights: { ols: 1, loess: 1, holt: 1, kalman: 1, corrected: 1 },
    errSd: null,      // h (days to election) -> sd (pp); null = method spread + 1st-round miss
  };

  const norm = s => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();

  // One key per institute across sources and years: "Genial/Quaest" -> "quaest",
  // "Globo e Folha/Datafolha" -> "datafolha", "Modalmais/Futura" -> "futura".
  const KNOWN = ['datafolha', 'ibope', 'ipec', 'quaest', 'atlasintel', 'poderdata', 'datapoder360', 'futura', 'gerp', 'palver',
    'vox', 'verita', 'nexus', 'ipespe', 'mda', 'sensus', 'parana', 'ideia', 'datatrends', 'indexa', 'ifp', 'alfa', 'gpp', 'fsb'];
  const STOP = new Set(['instituto', 'inteligencia', 'pesquisa', 'pesquisas', 'brasil', 'de', 'e', 'o', 's', 'ltda', 'btg', 'pactual',
    'genial', 'modalmais', 'apex', 'aya', 'meio', 'broadcast', '100', 'cidades', 'cnt', 'globo', 'folha', 'estado', 'paulo', 'cut', 'xp']);
  function pollsterKey(s) {
    const t = norm(s).split(/[^a-z0-9]+/).filter(Boolean);
    const known = t.find(x => KNOWN.includes(x));
    if (known) return known === 'datapoder360' ? 'poderdata' : known === 'fsb' ? 'nexus' : known;
    return t.find(x => !STOP.has(x)) || norm(s);
  }

  const weight = (p, c) => Math.sqrt(Math.min(p.n || 1000, c.nCap));
  const erf = x => {
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  };
  const ncdf = z => 0.5 * (1 + erf(z / Math.SQRT2));

  function wls(pts) {
    let sw = 0, st = 0, sy = 0;
    for (const p of pts) { sw += p.w; st += p.w * p.t; sy += p.w * p.y; }
    const mt = st / sw, my = sy / sw;
    let sxx = 0, sxy = 0;
    for (const p of pts) { sxx += p.w * (p.t - mt) ** 2; sxy += p.w * (p.t - mt) * (p.y - my); }
    const b = sxx > 0 ? sxy / sxx : 0;
    return d => my + b * (d - mt);
  }

  // Latest poll per institute within the window, weighted by sample size and recency.
  function average(polls, d, cfg = {}) {
    const c = { ...DEFAULTS, ...cfg };
    const latest = {};
    for (const p of polls) {
      if (p.end > d || d - p.end > c.windowDays) continue;
      if (!latest[p.key] || p.end > latest[p.key].end) latest[p.key] = p;
    }
    const used = Object.values(latest);
    if (!used.length) return null;
    let sw = 0, s = 0;
    for (const p of used) {
      const w = weight(p, c) * Math.pow(0.5, (d - p.end) / c.halfLife);
      sw += w; s += w * p.lula;
    }
    return { lula: s / sw, flavio: 100 - s / sw, used };
  }

  // Each method maps polls known by `today` to a value at day d (candidate A's share).
  // ctx: { today, e1, e2, from, bias: {key: pp}, firstRound: A's two-way 1st-round share }
  function methods(polls, ctx, c) {
    const known = polls.filter(p => p.end <= ctx.today);
    const post = known.filter(p => p.end > ctx.e1);
    const now = average(polls, ctx.today, c);
    if (post.length < c.minPolls || !now) return null;
    const tLast = Math.max(...post.map(p => p.end));
    const m = {};

    // Current weighted average, held flat (baseline).
    m.avg = () => now.lula;

    // Weighted linear regression over all runoff-campaign polls.
    m.ols = wls(post.map(p => ({ t: p.end, y: p.lula, w: weight(p, c) })));

    // LOESS: local linear fit around the latest poll (tricube kernel), extrapolated.
    m.loess = wls(known.filter(p => tLast - p.end < c.bw)
      .map(p => ({ t: p.end, y: p.lula, w: weight(p, c) * (1 - ((tLast - p.end) / c.bw) ** 3) ** 3 })));

    // Damped Holt (level + trend) over the daily average since the 1st round.
    let L = null, T = 0;
    for (let d = ctx.e1 + 1; d <= ctx.today; d++) {
      const a = average(polls, d, c);
      if (!a) continue;
      if (L === null) { L = a.lula; continue; }
      const prev = L;
      L = c.alpha * a.lula + (1 - c.alpha) * (L + c.phi * T);
      T = c.beta * (L - prev) + (1 - c.beta) * c.phi * T;
    }
    m.holt = d => { let s = 0, ph = 1; for (let i = 1; i <= d - ctx.today; i++) { ph *= c.phi; s += ph; } return L + T * s; };

    // Kalman filter (random-walk level), each poll weighted by sampling + house error.
    let x = null, P = 25, tPrev = null;
    for (const p of known.filter(p => p.end >= (ctx.from ?? ctx.e1 - 35)).sort((a, b) => a.end - b.end)) {
      if (x === null) { x = p.lula; tPrev = p.end; continue; }
      P += c.q * (p.end - tPrev); tPrev = p.end;
      const R = p.lula * (100 - p.lula) / (p.n || 1000) + c.house ** 2;
      const K = P / (P + R);
      x += K * (p.lula - x); P *= 1 - K;
    }
    m.kalman = () => x;

    // Current average corrected by these institutes' 1st-round miss.
    const bias = ctx.bias || {};
    const misses = [...new Set(post.map(p => p.key))].filter(k => k in bias).map(k => bias[k]);
    const meanMiss = misses.length ? misses.reduce((a, b) => a + b, 0) / misses.length : 0;
    m.corrected = () => now.lula + meanMiss;

    // The 1st-round result between the two finalists (no polls at all).
    if (ctx.firstRound != null) m.firstRound = () => ctx.firstRound;

    return { m, now, misses };
  }

  const NAMES = {
    avg: 'Média atual', ols: 'Regressão linear', loess: 'LOESS', holt: 'Holt amortecido', kalman: 'Filtro de Kalman',
    corrected: 'Correção pelo erro do 1º turno', firstRound: 'Resultado do 1º turno',
  };

  function forecast(polls, ctx, cfg = {}) {
    const c = { ...DEFAULTS, ...cfg };
    if (ctx.today >= ctx.e2) return null;
    const r = methods(polls, ctx, c);
    if (!r) return null;
    const list = Object.entries(c.weights).filter(([k, w]) => w > 0 && r.m[k])
      .map(([k, w]) => ({ key: k, name: NAMES[k], w, end: Math.min(99, Math.max(1, r.m[k](ctx.e2))) }));
    const sw = list.reduce((s, m) => s + m.w, 0);
    const mu = list.reduce((s, m) => s + m.w * m.end, 0) / sw;
    const h = ctx.e2 - ctx.today;
    let sd;
    if (c.errSd) sd = c.errSd(h);
    else {
      const spread = list.length > 1 ? Math.sqrt(list.reduce((s, m) => s + (m.end - mu) ** 2, 0) / (list.length - 1)) : 0;
      const histErr = Math.max(c.minErr, r.misses.length ? Math.sqrt(r.misses.reduce((s, b) => s + b * b, 0) / r.misses.length) : c.minErr);
      sd = Math.hypot(spread, histErr);
    }
    const pB = ncdf((50 - mu) / sd);  // chance candidate B (Flávio in 2026) finishes ahead
    const at = d => r.now.lula + (mu - r.now.lula) * (d - ctx.today) / h;
    // 80% band: the calibrated error for a projection d days ahead, or a cone to the election-day sd.
    const band = d => 1.2816 * (c.errSd ? c.errSd(d - ctx.today) : sd * (d - ctx.today) / h);
    return { methods: list, all: r.m, mu, sd, pFlavio: pB, at, band, start: r.now.lula };
  }

  root.PollModel = { DEFAULTS, NAMES, norm, pollsterKey, average, methods, forecast, ncdf };
})(typeof globalThis !== 'undefined' ? globalThis : this);
