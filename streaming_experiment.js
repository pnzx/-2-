const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { performance } = require('perf_hooks');

const OUT = path.join(__dirname, 'clickstream.csv');
const RESULT = path.join(__dirname, 'results.json');
const N = 500_000;
const VOCAB = 25_000;
const ZIPF = 1.07;

function rng(seed = 20261123) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function buildCdf() {
  const weights = new Float64Array(VOCAB);
  let total = 0;
  for (let i = 1; i <= VOCAB; i++) { weights[i - 1] = 1 / Math.pow(i, ZIPF); total += weights[i - 1]; }
  let c = 0;
  for (let i = 0; i < VOCAB; i++) { c += weights[i] / total; weights[i] = c; }
  weights[VOCAB - 1] = 1;
  return weights;
}
function generate() {
  const random = rng();
  const cdf = buildCdf();
  const fd = fs.openSync(OUT, 'w');
  fs.writeSync(fd, 'event_id,product_id\n');
  let buf = '';
  for (let i = 0; i < N; i++) {
    const x = random();
    let lo = 0, hi = VOCAB - 1;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (cdf[mid] < x) lo = mid + 1; else hi = mid; }
    buf += `${i + 1},P${String(lo + 1).padStart(5, '0')}\n`;
    if (buf.length >= 1 << 20) { fs.writeSync(fd, buf); buf = ''; }
  }
  if (buf) fs.writeSync(fd, buf);
  fs.closeSync(fd);
}

function mix32(x) {
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}
function itemNumber(s) { return Number(s.slice(1)); }
function hashPair(x) { return [mix32(x ^ 0x9e3779b9), mix32(x ^ 0x85ebca6b)]; }

class CountMinSketch {
  constructor(width, depth) { this.width = width; this.depth = depth; this.rows = Array.from({length: depth}, () => new Uint32Array(width)); }
  add(item) {
    const x = itemNumber(item);
    for (let r = 0; r < this.depth; r++) this.rows[r][mix32(x + Math.imul(r + 1, 0x9e3779b9)) % this.width]++;
  }
  estimate(item) {
    const x = itemNumber(item); let v = 0xffffffff;
    for (let r = 0; r < this.depth; r++) v = Math.min(v, this.rows[r][mix32(x + Math.imul(r + 1, 0x9e3779b9)) % this.width]);
    return v;
  }
  get bytes() { return this.width * this.depth * 4; }
}

class HyperLogLog {
  constructor(p) { this.p = p; this.m = 1 << p; this.reg = new Uint8Array(this.m); }
  add(item) {
    const x = itemNumber(item); const [hi, lo] = hashPair(x);
    const index = hi >>> (32 - this.p);
    const shifted = hi << this.p;
    const rank = shifted !== 0 ? Math.clz32(shifted) + 1 : (32 - this.p) + Math.clz32(lo) + 1;
    if (rank > this.reg[index]) this.reg[index] = rank;
  }
  count() {
    let sum = 0, zeros = 0;
    for (const r of this.reg) { sum += Math.pow(2, -r); if (r === 0) zeros++; }
    const alpha = this.m === 16 ? 0.673 : this.m === 32 ? 0.697 : this.m === 64 ? 0.709 : 0.7213 / (1 + 1.079 / this.m);
    let estimate = alpha * this.m * this.m / sum;
    if (estimate <= 2.5 * this.m && zeros > 0) estimate = this.m * Math.log(this.m / zeros);
    const two32 = 4294967296;
    if (estimate > two32 / 30) estimate = -two32 * Math.log(1 - estimate / two32);
    return estimate;
  }
  get bytes() { return this.m; }
}

async function forEachEvent(fn) {
  const rl = readline.createInterface({ input: fs.createReadStream(OUT), crlfDelay: Infinity });
  let header = true;
  for await (const line of rl) {
    if (header) { header = false; continue; }
    const comma = line.indexOf(',');
    fn(line.slice(comma + 1));
  }
}
async function timeOne(fn) {
  const t0 = performance.now(); await forEachEvent(fn); return performance.now() - t0;
}

async function main() {
  generate();
  if (global.gc) global.gc();
  const heapBeforeMap = process.memoryUsage().heapUsed;
  const exact = new Map();
  let exactMs = await timeOne((item) => exact.set(item, (exact.get(item) || 0) + 1));
  const exactDistinct = exact.size;
  const topItems = [...exact.entries()].sort((a,b)=>b[1]-a[1]).slice(0, 100).map(x=>x[0]);
  const freqEval = [...exact.keys()];
  const configs = [
    {width:256, depth:3}, {width:512, depth:4}, {width:1024, depth:5}, {width:2048, depth:5}
  ];
  const cmsResults = [];
  for (const cfg of configs) {
    const cms = new CountMinSketch(cfg.width, cfg.depth);
    const elapsedMs = await timeOne(item => cms.add(item));
    let abs = 0, rel = 0, maxOver = 0, topAbs = 0, topRel = 0;
    for (const item of freqEval) {
      const truth = exact.get(item); const est = cms.estimate(item); const err = est - truth;
      abs += Math.abs(err); rel += Math.abs(err) / truth; maxOver = Math.max(maxOver, err);
    }
    for (const item of topItems) { const truth=exact.get(item), est=cms.estimate(item); topAbs+=Math.abs(est-truth); topRel+=Math.abs(est-truth)/truth; }
    cmsResults.push({ ...cfg, bytes: cms.bytes, elapsedMs, throughput: N/(elapsedMs/1000), meanAbsoluteError: abs/freqEval.length, meanRelativeErrorPct: rel/freqEval.length*100, maxOverestimate: maxOver, top100MeanAbsoluteError: topAbs/topItems.length, top100MeanRelativeErrorPct: topRel/topItems.length*100 });
  }
  const hllResults = [];
  for (const p of [8,10,12,14]) {
    const hll = new HyperLogLog(p);
    const elapsedMs = await timeOne(item => hll.add(item));
    const estimate = hll.count();
    hllResults.push({p, buckets:hll.m, bytes:hll.bytes, elapsedMs, throughput:N/(elapsedMs/1000), estimate, exact:exactDistinct, absoluteError:Math.abs(estimate-exactDistinct), relativeErrorPct:Math.abs(estimate-exactDistinct)/exactDistinct*100, theoreticalRsePct:1.04/Math.sqrt(hll.m)*100});
  }
  const result = {
    dataset:{name:'Synthetic Zipfian e-commerce clickstream', records:N, vocabulary:VOCAB, zipfExponent:ZIPF, seed:20261123, fileBytes:fs.statSync(OUT).size, note:'Deterministically generated, one CSV row per event; product popularity follows a Zipf distribution.'},
    environment:{runtime:process.version, platform:process.platform, arch:process.arch},
    groundTruth:{distinctProducts:exactDistinct, exactCountingElapsedMs:exactMs, exactMapHeapBytes:process.memoryUsage().heapUsed-heapBeforeMap},
    countMin:cmsResults, hyperLogLog:hllResults
  };
  fs.writeFileSync(RESULT, JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
