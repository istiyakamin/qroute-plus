import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";

// ==========================================================
// Q-Route+ Web Simulation (single-file React)
// - Bernstein quantile gate (correlation-aware)
// - Shrinkage + low-rank covariance
// - Selective skipping with penalties
// - Bandit-guided ALNS (lightweight UCB)
// - Weekly risk budget knob (Boole bound shown)
// - Monte Carlo evaluation with correlated sampling
// ==========================================================

// -------------- Math / Utils --------------
function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

// Random low-rank factors (p x r)
function randomLowRankFactors(p, r, scale) {
  const L = Array.from({ length: p }, () => Array.from({ length: r }, () => randn() * scale));
  // normalize columns for stability
  for (let k = 0; k < r; k++) {
    let norm = 1e-9;
    for (let i = 0; i < p; i++) norm += L[i][k] ** 2;
    norm = Math.sqrt(norm);
    for (let i = 0; i < p; i++) L[i][k] /= norm;
  }
  return L;
}

// -------------- Instance generator --------------
function generateInstance({ p, r, surge, meanMin, meanMax, dMin, dMax, twWidth, gridSize }) {
  // Means and diagonal variances
  const mu = Array.from({ length: p }, () => meanMin + Math.random() * (meanMax - meanMin));
  let D = Array.from({ length: p }, () => dMin + Math.random() * (dMax - dMin));
  const L = randomLowRankFactors(p, r, surge);

  // Shrinkage (diagonal proxy)
  const tau = D.reduce((a, v) => a + v, 0) / Math.max(1, p);
  const lambda = 0.3; // shrinkage weight
  D = D.map(v => (1 - lambda) * v + lambda * tau);

  // Bernstein cap parameter b_i
  const b = Array.from({ length: p }, (_, i) => 2 * Math.sqrt(D[i]) + 0.2 * mu[i]);

  // Coordinates + time windows + service
  const coords = [];
  const e = []; // open
  const l = []; // close
  const s = []; // service time
  for (let i = 0; i < p; i++) {
    coords.push({ x: Math.random() * gridSize, y: Math.random() * gridSize });
    const open = 8 * 60 + Math.floor(Math.random() * 120); // 08:00-10:00
    const close = open + twWidth + Math.floor(Math.random() * 90);
    e.push(open); l.push(close); s.push(5 + Math.floor(Math.random() * 10));
  }
  const depot = { x: gridSize / 2, y: gridSize / 2 };
  return { mu, D, b, L, e, l, s, coords, depot, p, r };
}

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function travelMinutes(a, b, speedKmph = 30) {
  const km = (dist(a, b) / 1000.0) * 50; // grid scale
  return (km / speedKmph) * 60;
}

// -------------- Quantile Route Oracle --------------
function bernsteinQuantile(muR, sigma2R, bR, eps) {
  const ln = Math.log(1 / clamp(eps, 1e-6, 0.5));
  return muR + Math.sqrt(2 * sigma2R * ln) + (bR / 3) * ln;
}

function makeRouteCaches(p, r) {
  return {
    u: new Array(p).fill(false), // used flags
    g: new Array(r).fill(0),     // L^T * indicator
    sumD: 0,
    mu: 0,
    bMax: 0,
    seq: [],
  };
}

function findFeasiblePosition(inst, seq, j, { depot, speed }) {
  for (let pos = 0; pos <= seq.length; pos++) {
    const test = seq.slice();
    test.splice(pos, 0, j);
    if (forwardFeasible(inst, test, depot, speed)) return pos;
  }
  return -1;
}

function forwardFeasible(inst, seq, depot, speed) {
  const { e, l, s, coords } = inst;
  let time = 8 * 60; // start 08:00
  let last = depot;
  for (let idx of seq) {
    time += travelMinutes(last, coords[idx], speed);
    time = Math.max(time, e[idx]);
    if (time > l[idx]) return false;
    time += s[idx];
    last = coords[idx];
  }
  return true; // return-to-depot omitted for simplicity
}

function tryInsert({ inst, caches, j, C, eps, timeParams }) {
  const { mu, D, b, L } = inst;
  // Update order-independent stats
  const muPrime = caches.mu + mu[j];
  const gPrime = caches.g.slice();
  for (let k = 0; k < L[0].length; k++) gPrime[k] += L[j][k];
  const sumDPrime = caches.sumD + D[j];
  const sigma2Prime = gPrime.reduce((acc, x) => acc + x * x, 0) + sumDPrime;
  const bPrime = Math.max(caches.bMax, b[j]);

  // Capacity gate
  const q = bernsteinQuantile(muPrime, sigma2Prime, bPrime, eps);
  if (q > C) return { ok: false, reason: `Gate FAIL: q=${q.toFixed(2)} > C=${C}` };

  // Time window feasibility: try best position
  const pos = findFeasiblePosition(inst, caches.seq, j, timeParams);
  if (pos < 0) return { ok: false, reason: `Time-window FAIL` };

  // Commit
  caches.mu = muPrime; caches.g = gPrime; caches.sumD = sumDPrime; caches.bMax = bPrime; caches.u[j] = true;
  const newSeq = caches.seq.slice(); newSeq.splice(pos, 0, j); caches.seq = newSeq;
  return { ok: true, reason: `PASS` };
}

// -------------- Bandit ALNS --------------
const OPERATORS = ["randomRemove", "swapTwo", "relocateOne"];

function pickOperator(stats, t, c = 1.2) {
  let best = 0, bestVal = -Infinity;
  for (let k = 0; k < OPERATORS.length; k++) {
    const n = stats.count[k] || 0; const m = stats.mean[k] || 0;
    const ucb = m + c * Math.sqrt(Math.log(Math.max(2, t)) / Math.max(1, n));
    if (ucb > bestVal) { bestVal = ucb; best = k; }
  }
  return best;
}

function updateBandit(stats, k, reward) {
  const n = (stats.count[k] || 0) + 1;
  const m = stats.mean[k] || 0;
  stats.mean[k] = m + (reward - m) / n;
  stats.count[k] = n;
}

function rebuildCaches(inst, seq) {
  const caches = makeRouteCaches(inst.p, inst.r);
  for (let j of seq) {
    caches.mu += inst.mu[j];
    caches.sumD += inst.D[j];
    for (let k = 0; k < inst.r; k++) caches.g[k] += inst.L[j][k];
    caches.bMax = Math.max(caches.bMax, inst.b[j]);
    caches.u[j] = true;
  }
  caches.seq = seq.slice();
  return caches;
}

function applyOperator(opk, state) {
  const { inst, routes, C, eps, timeParams } = state;
  const r = 0; // single route demo
  const caches = routes[r];
  if (!caches || caches.seq.length === 0) return { delta: 0, changed: false };

  if (OPERATORS[opk] === "randomRemove") {
    const pos = Math.floor(Math.random() * caches.seq.length);
    const j = caches.seq[pos];
    const remaining = caches.seq.filter((_, i) => i !== pos);
    const rebuilt = rebuildCaches(inst, remaining);
    const res = tryInsert({ inst, caches: rebuilt, j, C, eps, timeParams });
    if (res.ok) { routes[r] = rebuilt; return { delta: -1, changed: true }; }
  }

  if (OPERATORS[opk] === "swapTwo" && caches.seq.length >= 2) {
    const i = Math.floor(Math.random() * caches.seq.length);
    let j = Math.floor(Math.random() * caches.seq.length);
    if (j === i) j = (j + 1) % caches.seq.length;
    const perm = caches.seq.slice();
    [perm[i], perm[j]] = [perm[j], perm[i]];
    if (forwardFeasible(inst, perm, state.depot, timeParams.speed)) {
      routes[r].seq = perm; return { delta: -0.5, changed: true };
    }
  }

  if (OPERATORS[opk] === "relocateOne" && caches.seq.length >= 2) {
    const i = Math.floor(Math.random() * caches.seq.length);
    let j = Math.floor(Math.random() * (caches.seq.length + 1));
    if (j === i) j = (j + 1) % (caches.seq.length + 1);
    const perm = caches.seq.slice();
    const v = perm.splice(i, 1)[0];
    perm.splice(j, 0, v);
    if (forwardFeasible(inst, perm, state.depot, timeParams.speed)) {
      routes[r].seq = perm; return { delta: -0.5, changed: true };
    }
  }

  return { delta: 0, changed: false };
}

// -------------- Monte Carlo --------------
function sampleCorrelated(inst) {
  const { mu, L, D, p } = inst;
  const zr = new Array(L[0].length).fill(0).map(() => randn());
  const X = new Array(p);
  for (let i = 0; i < p; i++) {
    let val = mu[i];
    for (let k = 0; k < L[0].length; k++) val += L[i][k] * zr[k];
    val += Math.sqrt(Math.max(D[i], 1e-6)) * randn();
    X[i] = Math.max(0, val);
  }
  return X;
}

function mcOverflowProb(inst, routeSeq, C, M = 300) {
  let count = 0;
  for (let m = 0; m < M; m++) {
    const X = sampleCorrelated(inst);
    let load = 0; for (let j of routeSeq) load += X[j];
    if (load > C) count++;
  }
  return count / M;
}

// -------------- Main Component --------------
export default function App() {
  const [p, setP] = useState(120);
  const [r, setR] = useState(10);
  const [surge, setSurge] = useState(1.2);
  const [capacity, setCapacity] = useState(750);
  const [eps, setEps] = useState(0.05);
  const [budget, setBudget] = useState(0.22);
  const [skipPenalty, setSkipPenalty] = useState(35);
  const [twWidth, setTwWidth] = useState(180);
  const [speed, setSpeed] = useState(35);

  const [instance, setInstance] = useState(null);
  const [routes, setRoutes] = useState([]);
  const [logs, setLogs] = useState([]);
  const [mc, setMc] = useState({ prob: null, weekly: null, M: 300, days: 5 });

  const depot = useMemo(() => ({ x: 500, y: 500 }), []);
  const timeParams = { depot, speed };

  function log(msg) {
    setLogs((L) => [{ t: new Date().toLocaleTimeString(), msg }, ...L].slice(0, 400));
  }

  function onGenerate() {
    const inst = generateInstance({ p, r, surge, meanMin: 6, meanMax: 22, dMin: 1.2, dMax: 6.5, twWidth, gridSize: 1000 });
    setInstance(inst);
    setRoutes([makeRouteCaches(p, r)]);
    setLogs([]);
    setMc(v => ({ ...v, prob: null, weekly: null }));
    log(`Generated instance: p=${p}, r=${r}, surge=${surge.toFixed(2)}`);
  }

  function tryServeOrSkip(caches, j) {
    const tryRes = tryInsert({ inst: instance, caches, j, C: capacity, eps, timeParams });
    if (tryRes.ok) { log(`Insert site ${j} → PASS`); return true; }
    const lastIdx = caches.seq[caches.seq.length - 1];
    const lastPt = typeof lastIdx === "number" ? instance.coords[lastIdx] : depot;
    const marginalKm = dist(lastPt, instance.coords[j]) / 1000 * 50;
    if (skipPenalty < 0.5 * marginalKm || /Gate FAIL/.test(tryRes.reason)) {
      log(`Skip site ${j} (reason: ${tryRes.reason})`);
      return false;
    }
    // Retry by position-only relaxation
    const pos = findFeasiblePosition(instance, caches.seq, j, timeParams);
    if (pos >= 0) {
      const tmp = rebuildCaches(instance, caches.seq);
      const res = tryInsert({ inst: instance, caches: tmp, j, C: capacity, eps, timeParams });
      if (res.ok) { Object.assign(caches, tmp); log(`Insert site ${j} at pos ${pos} → PASS on retry`); return true; }
    }
    log(`Skip site ${j} after retry (reason: ${tryRes.reason})`);
    return false;
  }

  function greedyBuild() {
    if (!instance) return;
    const caches = makeRouteCaches(instance.p, instance.r);
    const idx = Array.from({ length: instance.p }, (_, i) => i);
    idx.sort((a, b) => (instance.mu[b] - instance.mu[a]) + 0.001 * (dist(instance.coords[b], depot) - dist(instance.coords[a], depot)));
    let served = 0;
    for (let j of idx) if (tryServeOrSkip(caches, j)) served++;
    setRoutes([caches]);
    log(`Initial build complete: served=${served}, routeLen=${caches.seq.length}`);
  }

  function runALNS(iter = 200) {
    if (!instance || routes.length === 0) return;
    const state = { inst: instance, routes: [...routes], C: capacity, eps, timeParams, depot };
    const stats = { mean: Array(OPERATORS.length).fill(0), count: Array(OPERATORS.length).fill(0) };
    for (let t = 1; t <= iter; t++) {
      const k = pickOperator(stats, t);
      const { delta, changed } = applyOperator(k, state);
      if (changed) updateBandit(stats, k, -delta);
    }
    setRoutes(state.routes);
    log(`ALNS complete: iters=${iter}`);
  }

  function runMonteCarlo() {
    if (!instance || routes.length === 0) return;
    const perRouteProb = mcOverflowProb(instance, routes[0].seq, capacity, mc.M);
    const weeklyBound = clamp((mc.days || 5) * perRouteProb, 0, 0.99); // Boole bound
    setMc(v => ({ ...v, prob: perRouteProb, weekly: weeklyBound }));
    log(`Monte Carlo: per-route overflow ≈ ${perRouteProb.toFixed(3)}, weekly bound (days=${mc.days}) ≈ ${weeklyBound.toFixed(3)}`);
  }

  // KPIs
  const kpis = useMemo(() => {
    if (!instance || routes.length === 0) return null;
    const seq = routes[0].seq;
    let km = 0; let last = depot;
    for (let j of seq) { km += (dist(last, instance.coords[j]) / 1000) * 50; last = instance.coords[j]; }
    km += (dist(last, depot) / 1000) * 50;
    const hours = km / 30; // avg 30 km/h
    const co2 = km * 0.25; // simple proxy kg
    return { km, hours, co2, mcOv: mc.prob, weekly: mc.weekly };
  }, [instance, routes, depot, mc.prob, mc.weekly]);

  // -------------- Self-checks (light tests) --------------
  useEffect(() => {
    // Test: gate monotonicity in eps (larger eps -> larger ln(1/eps) decrease -> smaller q)
    const q1 = bernsteinQuantile(10, 5, 2, 0.05);
    const q2 = bernsteinQuantile(10, 5, 2, 0.10);
    if (!(q2 <= q1)) console.warn("Test failed: quantile monotonicity");
  }, []);

  return (
    <TooltipProvider>
      <div className="w-full min-h-screen bg-gray-50 p-4">
        <div className="max-w-[1200px] mx-auto grid grid-cols-12 gap-4">
          {/* Sidebar Controls */}
          <div className="col-span-12 lg:col-span-4 space-y-4">
            <Card className="shadow-sm">
              <CardContent className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Instance</h2>
                  <Badge>Q-Route+</Badge>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div><Label>Sites p</Label><Input type="number" value={p} onChange={e=>setP(parseInt(e.target.value||"0"))} /></div>
                  <div><Label>Rank r</Label><Input type="number" value={r} onChange={e=>setR(parseInt(e.target.value||"0"))} /></div>
                  <div>
                    <Label>Surge / Corr</Label>
                    <Slider min={0.4} max={2.0} step={0.1} value={[surge]} onValueChange={v=>setSurge(v[0])} />
                    <div className="text-xs text-gray-500 mt-1">{surge.toFixed(2)}</div>
                  </div>
                  <div><Label>TW Width (min)</Label><Input type="number" value={twWidth} onChange={e=>setTwWidth(parseInt(e.target.value||"0"))} /></div>
                  <div><Label>Capacity C</Label><Input type="number" value={capacity} onChange={e=>setCapacity(parseFloat(e.target.value||"0"))} /></div>
                  <div><Label>Risk per-route ε</Label><Input type="number" step="0.005" value={eps} onChange={e=>setEps(parseFloat(e.target.value||"0"))} /></div>
                  <div><Label>Weekly budget ϱ</Label><Input type="number" step="0.01" value={budget} onChange={e=>setBudget(parseFloat(e.target.value||"0"))} /></div>
                  <div><Label>Skip penalty pᵢ</Label><Input type="number" value={skipPenalty} onChange={e=>setSkipPenalty(parseFloat(e.target.value||"0"))} /></div>
                  <div><Label>Speed (km/h)</Label><Input type="number" value={speed} onChange={e=>setSpeed(parseFloat(e.target.value||"0"))} /></div>
                </div>
                <div className="flex gap-2 pt-2">
                  <Button onClick={onGenerate}>Generate</Button>
                  <Button variant="secondary" onClick={greedyBuild} disabled={!instance}>Build</Button>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={()=>runALNS(200)} disabled={!instance}>ALNS (200)</Button>
                  <Button variant="ghost" onClick={runMonteCarlo} disabled={!instance || routes.length===0}>Monte Carlo</Button>
                </div>
                {kpis && (
                  <div className="text-sm grid grid-cols-3 gap-2 pt-2">
                    <div><span className="font-semibold">Distance</span><div>{kpis.km.toFixed(1)} km</div></div>
                    <div><span className="font-semibold">Driver Hours</span><div>{kpis.hours.toFixed(1)} h</div></div>
                    <div><span className="font-semibold">Overflow (MC)</span><div>{kpis.mcOv==null?"—":kpis.mcOv.toFixed(3)}</div></div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="shadow-sm">
              <CardContent className="p-4 space-y-2 text-sm text-gray-700">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 mt-0.5" />
                  <div>
                    <b>Gate (Bernstein):</b> q = μ<sub>R</sub> + √(2 σ<sub>R</sub><sup>2</sup> ln(1/ε)) + (b<sub>R</sub>/3) ln(1/ε) ≤ C.
                    Low-rank updates keep it O(r). Shrinkage stabilizes variance.
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Panel */}
          <div className="col-span-12 lg:col-span-8 space-y-4">
            <Tabs defaultValue="routes">
              <TabsList>
                <TabsTrigger value="routes">Routes</TabsTrigger>
                <TabsTrigger value="map">Map</TabsTrigger>
                <TabsTrigger value="oracle">Oracle Log</TabsTrigger>
                <TabsTrigger value="mc">Monte Carlo</TabsTrigger>
              </TabsList>

              <TabsContent value="routes">
                <Card className="shadow-sm">
                  <CardContent className="p-4">
                    {!instance ? (
                      <div className="text-gray-500">Generate an instance to view routes.</div>
                    ) : (
                      <div className="grid grid-cols-1 gap-3">
                        <div className="text-sm">Route length: <b>{routes[0]?.seq.length||0}</b></div>
                        <div className="flex flex-wrap gap-2 max-h-56 overflow-auto">
                          {routes[0]?.seq.map((j, idx) => (
                            <Badge key={idx} variant="secondary">{j}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="map">
                <Card className="shadow-sm">
                  <CardContent className="p-4">
                    {!instance ? (
                      <div className="text-gray-500">Generate an instance to view map.</div>
                    ) : (
                      <div className="bg-white rounded-xl border p-2">
                        <svg viewBox="0 0 1000 1000" className="w-full h-[420px]">
                          {/* depot */}
                          <circle cx={depot.x} cy={depot.y} r={8} fill="#111" />
                          {/* sites */}
                          {instance.coords.map((c, i) => (
                            <circle key={i} cx={c.x} cy={c.y} r={3} fill="#64748b" opacity={0.8} />
                          ))}
                          {/* route polyline */}
                          {routes[0]?.seq.length>0 && (
                            <polyline
                              fill="none"
                              stroke="#0ea5e9"
                              strokeWidth={2}
                              points={[depot, ...routes[0].seq.map(i=>instance.coords[i]), depot].map(pt=>`${pt.x},${pt.y}`).join(" ")}
                            />
                          )}
                        </svg>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="oracle">
                <Card className="shadow-sm">
                  <CardContent className="p-2">
                    <div className="h-[420px] overflow-auto text-xs font-mono bg-gray-50 rounded-lg p-2 border">
                      {logs.map((l, i) => (
                        <div key={i} className="py-0.5"><span className="text-gray-400">[{l.t}]</span> {l.msg}</div>
                      ))}
                      {logs.length===0 && <div className="text-gray-400">No events yet. Build a route to see PASS/FAIL decisions.</div>}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="mc">
                <Card className="shadow-sm">
                  <CardContent className="p-4 space-y-3">
                    <div className="grid grid-cols-2 gap-3 items-center">
                      <div>
                        <Label>Scenarios M</Label>
                        <Input type="number" value={mc.M} onChange={e=>setMc({ ...mc, M: parseInt(e.target.value||"0") })} />
                      </div>
                      <div>
                        <Label>Days in week</Label>
                        <Input type="number" value={mc.days} onChange={e=>setMc({ ...mc, days: parseInt(e.target.value||"0") })} />
                      </div>
                    </div>
                    {kpis?.mcOv!=null ? (
                      <div className="grid grid-cols-4 gap-4 text-sm">
                        <div className="bg-gray-50 rounded-lg p-3 border"><div className="text-gray-500">Overflow (MC)</div><div className="text-2xl font-semibold">{kpis.mcOv.toFixed(3)}</div></div>
                        <div className="bg-gray-50 rounded-lg p-3 border"><div className="text-gray-500">Weekly (Boole) ≲</div><div className="text-2xl font-semibold">{kpis.weekly?.toFixed(3)}</div></div>
                        <div className="bg-gray-50 rounded-lg p-3 border"><div className="text-gray-500">Distance (km)</div><div className="text-2xl font-semibold">{kpis.km.toFixed(1)}</div></div>
                        <div className="bg-gray-50 rounded-lg p-3 border"><div className="text-gray-500">Driver Hours</div><div className="text-2xl font-semibold">{kpis.hours.toFixed(1)}</div></div>
                      </div>
                    ) : (
                      <div className="text-gray-500">Run Monte Carlo to estimate per-route overflow and a weekly bound; compare to your budget ϱ.</div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
