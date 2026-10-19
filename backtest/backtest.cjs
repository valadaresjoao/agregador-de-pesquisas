// Backtest of the runoff projection against past Brazilian presidential runoffs.
// Runs the same model.js the page uses, "as of" several days before each runoff,
// and scores every method and ensemble with leave-one-election-out validation.
//
// Usage: node backtest/backtest.cjs [--json]
require('../model.js');
const M = globalThis.PollModel;

const RAW = 'https://raw.githubusercontent.com/Fcairo1/agregador-br/main/data/';
const YEARS = [1989, 2002, 2006, 2010, 2014, 2018, 2022];
const HORIZONS = [21, 14, 10, 7, 5, 3, 1];  // days before the runoff
const day = s => Date.UTC(+s.slice(0, 4), +s.slice(5, 7) - 1, +s.slice(8, 10)) / 864e5;

function parseCSV(t) {
  const [head, ...rows] = t.trim().split('\n').map(l => l.replace(/\r$/, '').split(','));
  return rows.map(r => Object.fromEntries(head.map((k, i) => [k, r[i] ?? ''])));
}
const get = async (path, type = 'text') => {
  const r = await fetch(RAW + path);
  if (!r.ok) throw new Error(path + ' ' + r.status);
  return r[type]();
};
const findPct = (list, key, name) => {
  const hit = list.find(c => M.norm(c.name).includes(key) || M.norm(c.name).includes(M.norm(name)));
  if (!hit) throw new Error('no result for ' + key);
  return hit.pct;
};

async function loadElection(y, hist) {
  const h = hist[String(y)];
  const cands = parseCSV(await get(`candidates.presidente-${y}-2t.csv`));
  const polls2 = parseCSV(await get(`polls.presidente-${y}-2t.csv`));
  const [A, B] = Object.keys(polls2[0]).slice(8);  // candidate columns after "section"
  const nameOf = k => (cands.find(c => c.key === k) || { name: k }).name;
  const a2 = findPct(h.second, A, nameOf(A)), b2 = findPct(h.second, B, nameOf(B));
  const a1 = findPct(h.first, A, nameOf(A)), b1 = findPct(h.first, B, nameOf(B));
  const e1 = day(h.day1), e2 = day(h.day2);
  const firstRound = a1 / (a1 + b1) * 100;

  // Each institute's miss in the final 1st-round week, on A's two-way share.
  const bias = {}, hit = new Set();
  try {
    const polls1 = parseCSV(await get(`polls.presidente-${y}.csv`));
    const last = {};
    for (const r of polls1) {
      const e = day(r.end), k = M.pollsterKey(r.pollster);
      if (e >= e1 || e < e1 - 7 || !(+r[A] > 0) || !(+r[B] > 0)) continue;
      if (!last[k] || e >= day(last[k].end)) last[k] = r;
    }
    for (const [k, r] of Object.entries(last)) {
      bias[k] = firstRound - +r[A] / (+r[A] + +r[B]) * 100;
      // Same rule as the page's filter: final 1st-round poll had the right leader of the two finalists.
      if ((+r[A] > +r[B]) === (a1 > b1)) hit.add(k);
    }
  } catch { /* no 1st-round file (1989) */ }

  const polls = polls2.filter(r => +r[A] > 0 && +r[B] > 0).map(r => ({
    pollster: r.pollster, key: M.pollsterKey(r.pollster), end: day(r.end), n: +r.n || 0, lula: +r[A] / (+r[A] + +r[B]) * 100 }));
  return { y, A, B, e1, e2, polls, bias, hit, firstRound, actual: a2 / (a2 + b2) * 100 };
}

// One row per (election, horizon) where the model can project: each method's value at election day.
function rows(elections, cfg) {
  const out = [];
  for (const el of elections) for (const h of HORIZONS) {
    const ctx = { today: el.e2 - h, e1: el.e1, e2: el.e2, from: el.e1 - 35, bias: el.bias, firstRound: el.firstRound };
    const r = M.methods(el.polls, ctx, { ...M.DEFAULTS, ...cfg });
    if (!r) continue;
    const pred = Object.fromEntries(Object.entries(r.m).map(([k, f]) => [k, f(el.e2)]));
    out.push({ y: el.y, h, actual: el.actual, pred, nPolls: el.polls.filter(p => p.end <= ctx.today).length });
  }
  return out;
}

const rmse = es => Math.sqrt(es.reduce((s, e) => s + e * e, 0) / es.length);
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const combine = (pred, w) => { let s = 0, sw = 0; for (const k in w) if (pred[k] != null && w[k] > 0) { s += w[k] * pred[k]; sw += w[k]; } return s / sw; };

// Weights ∝ 1/MSE of each method, learned on the training rows only.
function invMseWeights(train, keys) {
  const w = {};
  for (const k of keys) { const es = train.filter(r => r.pred[k] != null).map(r => r.pred[k] - r.actual); if (es.length) w[k] = 1 / Math.max(0.25, mean(es.map(e => e * e))); }
  return w;
}
function bestSingle(train, keys) {
  let best = null, bestE = Infinity;
  for (const k of keys) { const es = train.filter(r => r.pred[k] != null).map(r => r.pred[k] - r.actual); if (es.length && rmse(es) < bestE) { bestE = rmse(es); best = k; } }
  return { [best]: 1 };
}

// Leave-one-election-out: weights for year y come only from the other years.
function loeo(all, keys, strategy) {
  const res = [];
  for (const y of [...new Set(all.map(r => r.y))]) {
    const train = all.filter(r => r.y !== y), test = all.filter(r => r.y === y);
    const w = strategy(train, keys);
    for (const r of test) res.push({ ...r, ens: combine(r.pred, w), w });
  }
  return res;
}

(async () => {
  const hist = JSON.parse(await get('historico-resultados.json'));
  const elections = [];
  for (const y of YEARS) {
    try { elections.push(await loadElection(y, hist)); }
    catch (e) { console.error(`skip ${y}: ${e.message}`); }
  }
  const all = rows(elections);
  const keys = Object.keys(M.NAMES);
  const report = { elections: elections.map(e => ({ y: e.y, A: e.A, B: e.B, polls: e.polls.length, actualA: +e.actual.toFixed(2), firstRoundA: +e.firstRound.toFixed(2), biasInstitutes: Object.keys(e.bias).length })), rows: all.length };

  // Single methods (no fitting involved, so in-sample = out-of-sample).
  report.methods = Object.fromEntries(keys.map(k => {
    const rs = all.filter(r => r.pred[k] != null), es = rs.map(r => r.pred[k] - r.actual);
    return [k, { n: es.length, rmse: +rmse(es).toFixed(2), bias: +mean(es).toFixed(2),
      winner: +(mean(rs.map(r => (r.pred[k] > 50) === (r.actual > 50) ? 1 : 0)) * 100).toFixed(0) }];
  }));

  const strategies = {
    'igual (5 atuais)': () => ({ ols: 1, loess: 1, holt: 1, kalman: 1, corrected: 1 }),
    'igual (todos)': (t, ks) => Object.fromEntries(ks.map(k => [k, 1])),
    'inverso do erro (todos)': invMseWeights,
    'melhor método único': bestSingle,
  };
  report.ensembles = {};
  for (const [name, s] of Object.entries(strategies)) {
    const res = loeo(all, keys, s), es = res.map(r => r.ens - r.actual);
    report.ensembles[name] = { rmse: +rmse(es).toFixed(2), bias: +mean(es).toFixed(2),
      winner: +(mean(res.map(r => (r.ens > 50) === (r.actual > 50) ? 1 : 0)) * 100).toFixed(0),
      byHorizon: Object.fromEntries(HORIZONS.map(h => [h, res.filter(r => r.h === h)]).filter(([, v]) => v.length).map(([h, v]) => [h, +rmse(v.map(r => r.ens - r.actual)).toFixed(2)])),
      byYear: Object.fromEntries([...new Set(res.map(r => r.y))].map(y => [y, +rmse(res.filter(r => r.y === y).map(r => r.ens - r.actual)).toFixed(2)])) };
  }
  report.weightsAllData = invMseWeights(all, keys);
  for (const k in report.weightsAllData) report.weightsAllData[k] = +report.weightsAllData[k].toFixed(3);

  // Average settings: pick window/half-life on 6 elections, score on the 7th.
  const grid = [];
  for (const windowDays of [7, 10, 14, 21]) for (const halfLife of [3, 5, 7, 10, 14]) grid.push({ windowDays, halfLife });
  const gridRows = grid.map(g => ({ g, rows: rows(elections, g) }));
  const avgErr = (rs, filter) => rs.filter(filter).map(r => r.pred.avg - r.actual);
  const loeoGrid = [];
  for (const y of YEARS) {
    const scored = gridRows.map(({ g, rows: rs }) => ({ g, train: rmse(avgErr(rs, r => r.y !== y)), test: avgErr(rs, r => r.y === y) }));
    const best = scored.sort((a, b) => a.train - b.train)[0];
    if (best.test.length) loeoGrid.push(...best.test.map(e => ({ y, e, g: best.g })));
  }
  report.avgSettings = {
    atual: rmse(avgErr(gridRows.find(x => x.g.windowDays === 14 && x.g.halfLife === 7).rows, () => true)),
    escolhidoFora: rmse(loeoGrid.map(x => x.e)),
    escolhas: [...new Set(loeoGrid.map(x => JSON.stringify(x.g)))],
    melhorComTudo: gridRows.map(({ g, rows: rs }) => ({ ...g, rmse: +rmse(avgErr(rs, () => true)).toFixed(2) })).sort((a, b) => a.rmse - b.rmse).slice(0, 5),
  };

  // The page's filter: does averaging only the institutes that called the 1st round right do better?
  const filt = [];
  for (const el of elections) {
    if (!el.hit.size) continue;
    const sub = el.polls.filter(p => el.hit.has(p.key));
    for (const h of HORIZONS) {
      const today = el.e2 - h;
      if (el.polls.filter(p => p.end > el.e1 && p.end <= today).length < M.DEFAULTS.minPolls) continue;
      const aAll = M.average(el.polls, today), aSub = M.average(sub, today);
      if (aAll && aSub) filt.push({ y: el.y, h, all: aAll.lula - el.actual, sub: aSub.lula - el.actual });
    }
  }
  report.hitFilter = { rows: filt.length, anos: [...new Set(filt.map(r => r.y))],
    rmseTodos: +rmse(filt.map(r => r.all)).toFixed(2), rmseSoAcertaram: +rmse(filt.map(r => r.sub)).toFixed(2),
    porAno: Object.fromEntries([...new Set(filt.map(r => r.y))].map(y => [y, {
      todos: +rmse(filt.filter(r => r.y === y).map(r => r.all)).toFixed(2), acertaram: +rmse(filt.filter(r => r.y === y).map(r => r.sub)).toFixed(2) }])) };

  // Error model for the win probability: sd(h)² = a + b·h fitted to out-of-sample errors.
  const chosen = loeo(all, keys, invMseWeights);
  const xs = chosen.map(r => r.h), ys = chosen.map(r => (r.ens - r.actual) ** 2);
  const mx = mean(xs), my = mean(ys);
  let b = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  b = Math.max(0, b);
  const a = Math.max(0.5, my - b * mx);
  report.errModel = { a: +a.toFixed(3), b: +b.toFixed(3), sd1: +Math.sqrt(a + b).toFixed(2), sd7: +Math.sqrt(a + 7 * b).toFixed(2), sd14: +Math.sqrt(a + 14 * b).toFixed(2) };

  // Calibration: old probability (method spread + 1st-round miss) vs fitted sd(h), scored out-of-sample.
  const score = probs => ({
    brier: +mean(probs.map(p => (p.pA - (p.actual > 50 ? 1 : 0)) ** 2)).toFixed(3),
    cobertura80: +(mean(probs.map(p => Math.abs(p.mu - p.actual) <= 1.2816 * p.sd ? 1 : 0)) * 100).toFixed(0),
  });
  const oldP = [], newP = [];
  for (const r of chosen) {
    const el = elections.find(e => e.y === r.y);
    const ctx = { today: el.e2 - r.h, e1: el.e1, e2: el.e2, from: el.e1 - 35, bias: el.bias };
    const f = M.forecast(el.polls, ctx);
    oldP.push({ pA: 1 - f.pFlavio, mu: f.mu, sd: f.sd, actual: r.actual });
    const sd = Math.sqrt(a + b * r.h);
    newP.push({ pA: 1 - M.ncdf((50 - r.ens) / sd), mu: r.ens, sd, actual: r.actual });
  }
  report.calibration = { antigo: score(oldP), novo: score(newP) };

  if (process.argv.includes('--write')) {
    const cal = { generated: new Date().toISOString().slice(0, 10), elections: elections.map(e => e.y), rows: all.length,
      weights: report.weightsAllData, errSd: { a: report.errModel.a, b: report.errModel.b },
      rmse: report.ensembles['inverso do erro (todos)'].rmse };
    require('fs').writeFileSync(require('path').join(__dirname, '..', 'calibration.js'),
      '// Generated by backtest/backtest.cjs --write. Do not edit by hand.\nglobalThis.PollCalibration = ' + JSON.stringify(cal, null, 2) + ';\n');
    console.error('calibration.js escrito');
  }

  if (process.argv.includes('--json')) { console.log(JSON.stringify({ report, all }, null, 1)); return; }
  console.log('Eleições:'); console.table(report.elections);
  console.log(`Linhas (eleição × horizonte com dados): ${report.rows}`);
  console.log('\nMétodos isolados (erro em pp na fatia do candidato A; "winner" = % de acerto do vencedor):'); console.table(report.methods);
  console.log('\nEnsembles (validação deixando uma eleição de fora):');
  console.table(Object.fromEntries(Object.entries(report.ensembles).map(([k, v]) => [k, { rmse: v.rmse, bias: v.bias, winner: v.winner }])));
  for (const [k, v] of Object.entries(report.ensembles)) console.log(k, 'por horizonte', v.byHorizon, 'por ano', v.byYear);
  console.log('\nPesos inverso-do-erro com todos os dados:', report.weightsAllData);
  console.log('\nJanela/meia-vida da média:', JSON.stringify(report.avgSettings, null, 1));
  console.log('\nFiltro "acertaram o 1º turno":', JSON.stringify(report.hitFilter, null, 1));
  console.log('\nModelo de erro sd(h)² = a + b·h:', report.errModel);
  console.log('Calibração da chance (fora da amostra):', report.calibration);
})();
