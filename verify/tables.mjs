// Every table cell, checked where it sits rather than checked for existing.
//
// scripts/check_numbers.py recomputes each published figure and then searches
// the whole document for the text of it. That catches a number going stale. It
// cannot catch a number being in the wrong cell, and the results table is three
// pairs by five methods, so two rows swapped would leave every value present
// and every claim wrong. The IPF table in notes/METHODS.md is worse, ten
// columns wide, and its columns are only labelled by number.
//
// This parses the markdown tables, resolves each row and column by its heading,
// and requires the cell to equal the value recomputed from
// results/transport.csv for that pair and that method.

import { readFileSync } from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? ".";
const read = (p) => readFileSync(path.join(root, p), "utf8");

// --- results/transport.csv ------------------------------------------------
const lines = read("results/transport.csv").trim().split("\n");
const head = lines[0].split(",");
const idx = (name) => {
  const i = head.indexOf(name);
  if (i < 0) throw new Error(`transport.csv has no column ${name}`);
  return i;
};
const [cPair, cSeed, cMethod, cStage, cW2, cWall] =
  ["pair", "seed", "method", "stage", "w2", "wall_s"].map(idx);
const rows = lines.slice(1).map((l) => {
  const f = l.split(",");
  return {
    pair: f[cPair], seed: +f[cSeed], method: f[cMethod], stage: +f[cStage],
    w2: Number(f[cW2]), wall: Number(f[cWall]),
  };
});
const finite = rows.filter((r) => Number.isFinite(r.w2));

const median = (v) => {
  if (v.length === 0) throw new Error("median of nothing");
  const s = [...v].sort((a, b) => a - b);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};
const medW2 = (pair, method, stage) =>
  median(finite.filter((r) => r.pair === pair && r.method === method &&
                              (stage === undefined || r.stage === stage)).map((r) => r.w2));
const dsbBest = (pair) =>
  Math.min(...finite.filter((r) => r.pair === pair && r.method === "dsb").map((r) => r.w2));

// --- markdown tables ------------------------------------------------------
function tables(md) {
  const out = [];
  const ls = md.split("\n");
  for (let i = 0; i < ls.length; i++) {
    if (!ls[i].trim().startsWith("|")) continue;
    if (!(ls[i + 1] ?? "").trim().startsWith("|-")) continue;
    const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|")
      .map((c) => c.replace(/[*`]/g, "").trim());
    const header = cells(ls[i]);
    const body = [];
    let j = i + 2;
    for (; j < ls.length && ls[j].trim().startsWith("|"); j++) body.push(cells(ls[j]));
    out.push({ header, body });
    i = j;
  }
  return out;
}

const readme = read("README.md");
const methods = read("notes/METHODS.md");
let bad = 0;
let checked = 0;
const cell = (name, got, want) => {
  checked++;
  if (got === want) return true;
  bad++;
  console.log(`  FAIL ${name.padEnd(40)} table says ${got}, data says ${want}`);
  return false;
};
const known = new Set(rows.map((r) => r.pair));
// "gaussian to 8 gaussians" in the prose is "gaussian->8gaussians" in the data.
const pairOf = (label) => {
  const p = label.replace(/ to /, "->").replace(/ /g, "");
  if (!known.has(p)) throw new Error(`no pair ${p} in transport.csv, from row label "${label}"`);
  return p;
};

// results table
const results = tables(readme).find((t) => t.header[1] === "doing nothing");
if (!results) throw new Error("no results table in README.md");
const column = {
  "doing nothing": (p) => medW2(p, "untransported"),
  "Sinkhorn (eps 0.03)": (p) => medW2(p, "sinkhorn-eps0.03"),
  "DSB (best run)": (p) => dsbBest(p),
  "bridge match (ODE)": (p) => medW2(p, "bridge-matching-ode", 100),
  "bridge match (SDE)": (p) => medW2(p, "bridge-matching-sde", 100),
};
for (const row of results.body) {
  const pair = pairOf(row[0]);
  for (const [name, fn] of Object.entries(column)) {
    const c = results.header.indexOf(name);
    if (c < 0) { console.log(`  FAIL results table has no column ${name}`); bad++; continue; }
    cell(`${pair} ${name}`, row[c], fn(pair).toFixed(3));
  }
}

// wall clock table
const wall = tables(readme).find((t) => t.header[0] === "method" && t.header[1] === "time");
if (!wall) throw new Error("no wall clock table in README.md");
const dsbElapsed = [...new Set(rows.filter((r) => r.method === "dsb")
  .map((r) => `${r.pair}|${r.seed}`))].map((k) =>
    Math.max(...rows.filter((r) => r.method === "dsb" && `${r.pair}|${r.seed}` === k)
      .map((r) => r.wall)));
const wallWant = {
  Sinkhorn: median(rows.filter((r) => r.method === "sinkhorn-eps0.03").map((r) => r.wall)).toFixed(1),
  "bridge matching": median(rows.filter((r) => r.method === "bridge-matching-ode" && r.stage === 100)
    .map((r) => r.wall)).toFixed(1),
  DSB: median(dsbElapsed).toFixed(0),
};
const cTime = wall.header.indexOf("time");
for (const row of wall.body) {
  const want = wallWant[row[0]];
  if (want === undefined) { console.log(`  FAIL unknown method row ${row[0]}`); bad++; continue; }
  cell(`wall clock ${row[0]}`, row[cTime].replace(/\s*s$/, ""), want);
}

// per IPF iteration table in notes/METHODS.md
const ipf = tables(methods).find((t) => t.header[1] === "ipf 0");
if (!ipf) throw new Error("no IPF table in notes/METHODS.md");
const stages = ipf.header.slice(1).map((h) => Number(h.replace("ipf ", "")));
for (const row of ipf.body) {
  const pair = pairOf(row[0]);
  for (let k = 0; k < stages.length; k++) {
    cell(`${pair} ipf ${stages[k]}`, row[k + 1], medW2(pair, "dsb", stages[k]).toFixed(3));
  }
}

if (bad > 0) {
  console.log(`  ${bad} table cells disagree with results/transport.csv`);
  process.exit(1);
}
console.log(`  all ${checked} cells of the three published tables sit where the data puts them`);
