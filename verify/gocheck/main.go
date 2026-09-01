// Structural validation of every results file the repo publishes, plus the
// shape claims the README makes about them.
//
// The other checks in verify/ recompute numbers. This one asks whether the
// files those numbers come out of are the shape everything downstream assumes:
// no ragged row, no duplicate column name, no silent non finite value outside
// the two runs that are documented as having gone non finite, and a row count
// per pair and seed that matches results/run-meta.json rather than whatever
// happened to be written last.
//
// A missing row is the failure mode a recompute cannot see. A median over two
// seeds instead of three is still a median, and it is still wrong.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

type table struct {
	name string
	head []string
	rows [][]string
}

func (t *table) col(name string) int {
	for i, h := range t.head {
		if h == name {
			return i
		}
	}
	fail("%s: no column named %q", t.name, name)
	return -1
}

func (t *table) get(r []string, name string) string { return r[t.col(name)] }

var problems []string

func fail(format string, a ...any) {
	problems = append(problems, fmt.Sprintf(format, a...))
}

// readCSV rejects ragged rows and duplicate column names on the way in.
func readCSV(path string) *table {
	fh, err := os.Open(path)
	if err != nil {
		fmt.Println("  cannot open", path, err)
		os.Exit(1)
	}
	defer fh.Close()
	r := csv.NewReader(fh)
	r.FieldsPerRecord = 0 // every record must match the header width
	recs, err := r.ReadAll()
	if err != nil {
		fmt.Printf("  %s: %v\n", path, err)
		os.Exit(1)
	}
	if len(recs) < 2 {
		fmt.Printf("  %s: no data rows\n", path)
		os.Exit(1)
	}
	t := &table{name: filepath.Base(path), head: recs[0], rows: recs[1:]}
	seen := map[string]bool{}
	for _, h := range t.head {
		if seen[h] {
			fail("%s: duplicate column name %q", t.name, h)
		}
		seen[h] = true
	}
	return t
}

func mustFloat(t *table, r []string, name string) float64 {
	v, err := strconv.ParseFloat(t.get(r, name), 64)
	if err != nil {
		fail("%s: %q is not a number in column %s", t.name, t.get(r, name), name)
		return math.NaN()
	}
	return v
}

func main() {
	root := flag.String("root", "..", "repository root")
	flag.Parse()
	p := func(rel string) string { return filepath.Join(*root, rel) }

	tr := readCSV(p("results/transport.csv"))
	want := []string{"pair", "seed", "method", "stage", "w2", "wall_s", "nfe", "params"}
	if len(tr.head) != len(want) {
		fail("transport.csv: %d columns, expected %d", len(tr.head), len(want))
	}
	for _, w := range want {
		found := false
		for _, h := range tr.head {
			if h == w {
				found = true
			}
		}
		if !found {
			fail("transport.csv: missing column %q", w)
		}
	}
	if len(problems) > 0 {
		report()
	}

	// run-meta.json is the record of how the experiment was invoked. Everything
	// below is checked against it rather than against a constant written here.
	var meta struct {
		Pairs      []string `json:"pairs"`
		Seeds      []int    `json:"seeds"`
		N          int      `json:"n"`
		DSBIPF     int      `json:"dsb_ipf"`
		BMSteps    int      `json:"bm_steps"`
		WallClockS float64  `json:"wall_clock_s"`
	}
	raw, err := os.ReadFile(p("results/run-meta.json"))
	if err != nil {
		fmt.Println("  cannot read run-meta.json:", err)
		os.Exit(1)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		fmt.Println("  run-meta.json:", err)
		os.Exit(1)
	}

	type key struct {
		pair string
		seed int
	}
	perRun := map[key]int{}
	dsbStages := map[key]map[int]bool{}
	methods := map[string]int{}
	nonFinite := []string{}

	for i, r := range tr.rows {
		pair := tr.get(r, "pair")
		seed, err := strconv.Atoi(tr.get(r, "seed"))
		if err != nil {
			fail("transport.csv row %d: seed %q is not an integer", i+2, tr.get(r, "seed"))
			continue
		}
		stage, err := strconv.Atoi(tr.get(r, "stage"))
		if err != nil {
			fail("transport.csv row %d: stage %q is not an integer", i+2, tr.get(r, "stage"))
			continue
		}
		method := tr.get(r, "method")
		w2 := mustFloat(tr, r, "w2")
		wall := mustFloat(tr, r, "wall_s")
		if math.IsInf(w2, 0) || math.IsNaN(w2) {
			nonFinite = append(nonFinite, fmt.Sprintf("%s seed %d %s stage %d", pair, seed, method, stage))
		} else if w2 < 0 {
			fail("transport.csv row %d: negative w2 %g", i+2, w2)
		}
		if math.IsNaN(wall) || math.IsInf(wall, 0) || wall < 0 {
			fail("transport.csv row %d: wall_s is %v", i+2, tr.get(r, "wall_s"))
		}
		k := key{pair, seed}
		perRun[k]++
		methods[method]++
		if method == "dsb" {
			if dsbStages[k] == nil {
				dsbStages[k] = map[int]bool{}
			}
			if dsbStages[k][stage] {
				fail("transport.csv: %s seed %d has two dsb rows at IPF %d", pair, seed, stage)
			}
			dsbStages[k][stage] = true
		}
	}

	// Every pair and seed in the meta must be present, with the same number of
	// rows as every other run.
	wantRuns := len(meta.Pairs) * len(meta.Seeds)
	if len(perRun) != wantRuns {
		fail("transport.csv: %d pair and seed combinations, run-meta says %d", len(perRun), wantRuns)
	}
	for _, pair := range meta.Pairs {
		for _, seed := range meta.Seeds {
			k := key{pair, seed}
			if perRun[k] == 0 {
				fail("transport.csv: nothing for %s seed %d", pair, seed)
				continue
			}
			if got := len(dsbStages[k]); got != meta.DSBIPF {
				fail("transport.csv: %s seed %d has %d dsb IPF rows, run-meta says %d",
					pair, seed, got, meta.DSBIPF)
			}
		}
	}
	counts := map[int]int{}
	for _, c := range perRun {
		counts[c]++
	}
	if len(counts) != 1 {
		fail("transport.csv: runs do not all have the same number of rows: %v", counts)
	}
	if got := methods["dsb"]; got != wantRuns*meta.DSBIPF {
		fail("transport.csv: %d dsb rows, expected %d runs times %d IPF iterations",
			got, wantRuns, meta.DSBIPF)
	}

	// The two non finite DSB runs are documented in notes/METHODS.md. Any other
	// one is a new failure that nobody has written down.
	allowed := map[string]bool{
		"moons->spiral seed 0 dsb stage 8": true,
		"moons->spiral seed 0 dsb stage 9": true,
	}
	sort.Strings(nonFinite)
	for _, nf := range nonFinite {
		if !allowed[nf] {
			fail("transport.csv: undocumented non finite w2 at %s", nf)
		}
	}
	if len(nonFinite) != len(allowed) {
		fail("transport.csv: %d non finite rows, notes/METHODS.md documents %d",
			len(nonFinite), len(allowed))
	}

	// The golden export the C and Rust checks read. Same treatment: shape first.
	pts := func(rel string) int {
		t := readCSV(p(rel))
		t.col("x")
		t.col("y")
		for i, r := range t.rows {
			for _, c := range []string{"x", "y"} {
				v := mustFloat(t, r, c)
				if math.IsNaN(v) || math.IsInf(v, 0) {
					fail("%s row %d: %s is not finite", t.name, i+2, c)
				}
			}
		}
		return len(t.rows)
	}
	ms := readCSV(p("verify/golden/metric_summary.csv"))
	nT, _ := strconv.Atoi(ms.get(ms.rows[0], "n_transported"))
	nG, _ := strconv.Atoi(ms.get(ms.rows[0], "n_target"))
	if got := pts("verify/golden/metric_transported.csv"); got != nT {
		fail("metric_transported.csv has %d points, the summary says %d", got, nT)
	}
	if got := pts("verify/golden/metric_target.csv"); got != nG {
		fail("metric_target.csv has %d points, the summary says %d", got, nG)
	}
	if nG != meta.N {
		fail("the golden target cloud has %d points, run-meta says the experiment used %d", nG, meta.N)
	}
	kn := pts("verify/golden/kernel_source.csv")
	if got := pts("verify/golden/kernel_target.csv"); got != kn {
		fail("kernel_source.csv has %d points and kernel_target.csv has %d", kn, got)
	}
	ks := readCSV(p("verify/golden/kernel_summary.csv"))
	pot := readCSV(p("verify/golden/kernel_potentials.csv"))
	seenPot := map[string]bool{}
	for _, r := range pot.rows {
		k := pot.get(r, "eps") + "/" + pot.get(r, "index")
		if seenPot[k] {
			fail("kernel_potentials.csv: two rows for eps %s index %s",
				pot.get(r, "eps"), pot.get(r, "index"))
		}
		seenPot[k] = true
		for _, c := range []string{"u", "v"} {
			if v := mustFloat(pot, r, c); math.IsNaN(v) || math.IsInf(v, 0) {
				fail("kernel_potentials.csv: %s is not finite at eps %s index %s",
					c, pot.get(r, "eps"), pot.get(r, "index"))
			}
		}
	}
	for _, r := range ks.rows {
		n, _ := strconv.Atoi(ks.get(r, "n"))
		if n != kn {
			fail("kernel_summary.csv says n=%d, the point files have %d", n, kn)
		}
		for i := 0; i < n; i++ {
			if !seenPot[ks.get(r, "eps")+"/"+strconv.Itoa(i)] {
				fail("kernel_potentials.csv: nothing for eps %s index %d", ks.get(r, "eps"), i)
				break
			}
		}
		it, _ := strconv.Atoi(ks.get(r, "iters"))
		iterCap, _ := strconv.Atoi(ks.get(r, "iters_cap"))
		if it < 1 || it > iterCap {
			fail("kernel_summary.csv: %d iterations with a cap of %d", it, iterCap)
		}
	}
	if len(pot.rows) != len(ks.rows)*kn {
		fail("kernel_potentials.csv has %d rows, expected %d regularisation levels times %d points",
			len(pot.rows), len(ks.rows), kn)
	}

	fmt.Printf("  %d rows over %d runs, %d dsb rows, %d documented non finite\n",
		len(tr.rows), len(perRun), methods["dsb"], len(nonFinite))
	report()
}

func report() {
	if len(problems) > 0 {
		fmt.Printf("  %d structural problems:\n", len(problems))
		for _, p := range problems {
			fmt.Println("   -", p)
		}
		os.Exit(1)
	}
	fmt.Println("  results/ and verify/golden/ are the shape run-meta.json and the notes say")
	os.Exit(0)
}
