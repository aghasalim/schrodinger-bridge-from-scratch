// The coupling itself: is it a coupling, and does averaging it give the map the
// experiment used.
//
// verify/sinkhorn.c reruns the fixed point iteration and compares the plan it
// converges to. This starts from the other end. It takes the plan out of the
// golden potentials without iterating at all, and asks the two questions the
// iteration is supposed to have answered: do both marginals come out uniform,
// which is the constraint the whole Schrodinger bridge problem is defined by,
// and does the barycentric projection of that plan land on the transported
// points the experiment then measured.
//
// The barycentric step is worth checking on its own because notes/METHODS.md
// blames it for the one row where Sinkhorn does worse than doing nothing. A
// claim like that should not rest on one implementation of the averaging.
//
// Columns are resolved by name. Plain JDK, no build file.

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class PlanCheck {

    static final class Table {
        final String name;
        final List<String> head = new ArrayList<>();
        final List<String[]> rows = new ArrayList<>();

        Table(Path p) throws IOException {
            name = p.getFileName().toString();
            try (BufferedReader r = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
                String line = r.readLine();
                if (line == null) throw new IOException(name + " is empty");
                for (String h : line.trim().split(",")) head.add(h.trim());
                while ((line = r.readLine()) != null) {
                    if (line.isBlank()) continue;
                    rows.add(line.trim().split(",", -1));
                }
            }
            if (rows.isEmpty()) throw new IOException(name + " has no data rows");
        }

        int col(String want) {
            int i = head.indexOf(want);
            if (i < 0) throw new IllegalStateException(name + " has no column " + want);
            return i;
        }

        double num(String[] row, String want) {
            return Double.parseDouble(row[col(want)]);
        }
    }

    static double[][] points(Table t) {
        int cx = t.col("x"), cy = t.col("y");
        double[][] out = new double[t.rows.size()][2];
        for (int i = 0; i < out.length; i++) {
            out[i][0] = Double.parseDouble(t.rows.get(i)[cx]);
            out[i][1] = Double.parseDouble(t.rows.get(i)[cy]);
        }
        return out;
    }

    public static void main(String[] args) throws IOException {
        Path root = Path.of(args.length > 0 ? args[0] : ".");
        Path g = root.resolve("verify/golden");
        double[][] x = points(new Table(g.resolve("kernel_source.csv")));
        double[][] y = points(new Table(g.resolve("kernel_target.csv")));
        if (x.length != y.length) {
            System.out.println("  source and target sizes differ");
            System.exit(1);
        }
        int n = x.length;

        Table pot = new Table(g.resolve("kernel_potentials.csv"));
        Table bar = new Table(g.resolve("kernel_transported.csv"));
        Table sum = new Table(g.resolve("kernel_summary.csv"));

        Map<String, double[][]> uv = new LinkedHashMap<>();
        for (String[] r : pot.rows) {
            String eps = r[pot.col("eps")];
            int i = Integer.parseInt(r[pot.col("index")]);
            double[][] a = uv.computeIfAbsent(eps, k -> new double[2][n]);
            a[0][i] = pot.num(r, "u");
            a[1][i] = pot.num(r, "v");
        }
        Map<String, double[][]> golden = new LinkedHashMap<>();
        for (String[] r : bar.rows) {
            String eps = r[bar.col("eps")];
            int i = Integer.parseInt(r[bar.col("index")]);
            double[][] a = golden.computeIfAbsent(eps, k -> new double[n][2]);
            a[i][0] = bar.num(r, "x");
            a[i][1] = bar.num(r, "y");
        }

        int failures = 0, checked = 0;
        for (String[] s : sum.rows) {
            String key = s[sum.col("eps")];
            double eps = Double.parseDouble(key);
            double wantRow = sum.num(s, "row_marginal_max_err");
            double wantCol = sum.num(s, "col_marginal_max_err");
            double wantCost = sum.num(s, "cost");
            double[][] a = uv.get(key);
            double[][] gold = golden.get(key);
            if (a == null || gold == null) {
                System.out.println("  eps " + key + " FAIL: no potentials or no transported points");
                failures++;
                continue;
            }
            double[] u = a[0], v = a[1];

            // log P_ij = -C_ij/eps + (u_i + v_j - u_0)/eps, which is the gauge
            // the export fixed: u is the first column and v the first row of
            // eps*(log P + C/eps), so both contain f_0 + g_0 once.
            double[] rowSum = new double[n];
            double[] colSum = new double[n];
            double[][] mapped = new double[n][2];
            double cost = 0.0;
            for (int i = 0; i < n; i++) {
                double[] p = new double[n];
                for (int j = 0; j < n; j++) {
                    double dx = x[i][0] - y[j][0], dy = x[i][1] - y[j][1];
                    double c = dx * dx + dy * dy;
                    p[j] = Math.exp((-c + u[i] + v[j] - u[0]) / eps);
                    rowSum[i] += p[j];
                    colSum[j] += p[j];
                    cost += p[j] * c;
                }
                double s2 = Math.max(rowSum[i], 1e-30);
                for (int j = 0; j < n; j++) {
                    mapped[i][0] += p[j] / s2 * y[j][0];
                    mapped[i][1] += p[j] / s2 * y[j][1];
                }
            }

            double rowErr = 0, colErr = 0, mapErr = 0;
            for (int i = 0; i < n; i++) {
                rowErr = Math.max(rowErr, Math.abs(rowSum[i] - 1.0 / n));
                colErr = Math.max(colErr, Math.abs(colSum[i] - 1.0 / n));
                mapErr = Math.max(mapErr, Math.max(Math.abs(mapped[i][0] - gold[i][0]),
                                                   Math.abs(mapped[i][1] - gold[i][1])));
            }
            double relCost = Math.abs(cost - wantCost) / Math.abs(wantCost);

            // The plan is required to be a coupling to the accuracy the export
            // recorded, with a factor of ten of headroom. The map and cost
            // tolerances are the measured cost of exponentiating a float32
            // potential in double: the plan divides u and v by eps, so at eps
            // 0.03 a single precision potential is amplified by a factor of
            // thirty before exp sees it. 5e-5 on the map and 1e-5 on the cost
            // are what that came out at, not targets chosen in advance.
            boolean bad = rowErr > 10 * Math.max(wantRow, 1e-9)
                       || colErr > 10 * Math.max(wantCol, 1e-9)
                       || mapErr > 5e-5
                       || relCost > 1e-5;
            System.out.printf("  eps %-5s marginals %.2e and %.2e (export %.2e and %.2e)"
                            + "  max map gap %.2e  cost rel %.2e  %s%n",
                    key, rowErr, colErr, wantRow, wantCol, mapErr, relCost, bad ? "FAIL" : "ok");
            if (bad) failures++;
            checked++;
        }
        if (checked == 0) {
            System.out.println("  nothing checked");
            System.exit(1);
        }
        if (failures > 0) {
            System.out.println("  " + failures + " of " + checked
                    + " plans are not the coupling the export says they are");
            System.exit(1);
        }
        System.out.println("  the golden plan is a coupling and its barycentric map is"
                + " the transported cloud, at all " + checked + " regularisation levels");
    }
}
