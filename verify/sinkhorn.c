/* Log domain Sinkhorn, reimplemented in C, checked against the golden export.
 *
 * sb/sinkhorn/solver.py is the ground truth the rest of the repo is compared
 * against, and until now the only thing that had ever run it was itself. This
 * recomputes the same fixed point from the same 256 by 256 problem and requires
 * the plan back out to match verify/golden/kernel_potentials.csv.
 *
 * Two deliberate differences from the Python. This works in double where torch
 * defaults to float, and it builds the cost matrix as a sum of squared
 * differences where torch.cdist takes a square root and squares it again. Both
 * are there so that agreement means the algorithm agrees, not that the same
 * floating point path was retraced. The tolerances below are the measured
 * consequence of that, not a guess.
 *
 * Every column is resolved by name from the header, so reordering a golden file
 * cannot silently feed the wrong numbers in.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXLINE 4096

static void die(const char *msg, const char *what) {
    fprintf(stderr, "sinkhorn.c: %s: %s\n", msg, what);
    exit(1);
}

/* Index of a named column in a CSV header line, or -1. */
static int column_of(char *header, const char *want) {
    int idx = 0;
    char *save = NULL;
    for (char *t = strtok_r(header, ",\r\n", &save); t; t = strtok_r(NULL, ",\r\n", &save)) {
        if (strcmp(t, want) == 0) return idx;
        idx++;
    }
    return -1;
}

static char *field(char *line, int col) {
    static char buf[MAXLINE];
    snprintf(buf, sizeof buf, "%s", line);
    int idx = 0;
    char *save = NULL;
    for (char *t = strtok_r(buf, ",\r\n", &save); t; t = strtok_r(NULL, ",\r\n", &save)) {
        if (idx == col) return t;
        idx++;
    }
    return NULL;
}

/* Read a two column point file with headers x and y. */
static int read_points(const char *path, double **out) {
    FILE *fh = fopen(path, "r");
    if (!fh) die("cannot open", path);
    char line[MAXLINE], header[MAXLINE];
    if (!fgets(header, MAXLINE, fh)) die("empty file", path);
    char hdr[MAXLINE];
    snprintf(hdr, sizeof hdr, "%s", header);
    int cx = column_of(hdr, "x");
    snprintf(hdr, sizeof hdr, "%s", header);
    int cy = column_of(hdr, "y");
    if (cx < 0 || cy < 0) die("missing x or y column", path);
    int cap = 1024, n = 0;
    double *p = malloc((size_t)cap * 2 * sizeof(double));
    while (fgets(line, MAXLINE, fh)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (n == cap) { cap *= 2; p = realloc(p, (size_t)cap * 2 * sizeof(double)); }
        char *a = field(line, cx), *b = field(line, cy);
        if (!a || !b) die("short row in", path);
        p[2 * n] = atof(a);
        p[2 * n + 1] = atof(b);
        n++;
    }
    fclose(fh);
    *out = p;
    return n;
}

static double logsumexp(const double *v, int n, int stride) {
    double m = -INFINITY;
    for (int i = 0; i < n; i++) if (v[i * stride] > m) m = v[i * stride];
    if (!isfinite(m)) return m;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp(v[i * stride] - m);
    return m + log(s);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024];
    double *x, *y;
    snprintf(path, sizeof path, "%s/verify/golden/kernel_source.csv", root);
    int n = read_points(path, &x);
    snprintf(path, sizeof path, "%s/verify/golden/kernel_target.csv", root);
    int m = read_points(path, &y);
    if (n != m) die("source and target sizes differ", "kernel_*.csv");
    printf("  read %d source and %d target points\n", n, m);

    double *C = malloc((size_t)n * m * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double dx = x[2 * i] - y[2 * j], dy = x[2 * i + 1] - y[2 * j + 1];
            C[(size_t)i * m + j] = dx * dx + dy * dy;
        }

    /* golden summary */
    snprintf(path, sizeof path, "%s/verify/golden/kernel_summary.csv", root);
    FILE *sm = fopen(path, "r");
    if (!sm) die("cannot open", path);
    char header[MAXLINE], line[MAXLINE], hdr[MAXLINE];
    if (!fgets(header, MAXLINE, sm)) die("empty file", path);
    snprintf(hdr, sizeof hdr, "%s", header); int c_eps = column_of(hdr, "eps");
    snprintf(hdr, sizeof hdr, "%s", header); int c_it = column_of(hdr, "iters");
    snprintf(hdr, sizeof hdr, "%s", header); int c_cap = column_of(hdr, "iters_cap");
    snprintf(hdr, sizeof hdr, "%s", header); int c_cost = column_of(hdr, "cost");
    if (c_eps < 0 || c_it < 0 || c_cap < 0 || c_cost < 0) die("missing column in", path);

    double *f = malloc((size_t)n * sizeof(double));
    double *g = malloc((size_t)m * sizeof(double));
    double *tmp = malloc((size_t)(n > m ? n : m) * sizeof(double));
    int failures = 0, checked = 0;

    while (fgets(line, MAXLINE, sm)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        double eps = atof(field(line, c_eps));
        int want_iters = atoi(field(line, c_it));
        int cap = atoi(field(line, c_cap));
        double want_cost = atof(field(line, c_cost));

        for (int i = 0; i < n; i++) f[i] = 0.0;
        for (int j = 0; j < m; j++) g[j] = 0.0;
        double log_a = -log((double)n), log_b = -log((double)m);
        int it = 0;
        double err = NAN;
        for (it = 0; it < cap; it++) {
            double maxd = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) tmp[j] = -C[(size_t)i * m + j] / eps + g[j] / eps;
                double nf = eps * (log_a - logsumexp(tmp, m, 1));
                double d = fabs(nf - f[i]);
                if (d > maxd) maxd = d;
                f[i] = nf;
            }
            for (int j = 0; j < m; j++) {
                for (int i = 0; i < n; i++) tmp[i] = -C[(size_t)i * m + j] / eps + f[i] / eps;
                g[j] = eps * (log_b - logsumexp(tmp, n, 1));
            }
            err = maxd;
            if (err < 1e-9) break;
        }
        int iters = it + 1 > cap ? cap : it + 1;

        /* The Python updates every f from the previous g in one shot. Doing the
         * rows in place, as above, would be a different algorithm, so the row
         * pass must not see its own updates: it does not, because tmp is built
         * from g only. The column pass does see the new f, exactly as in the
         * Python. */

        double cost = 0.0, plan_sum = 0.0;
        double du = 0.0, dv = 0.0;
        double *u = malloc((size_t)n * sizeof(double));
        double *v = malloc((size_t)m * sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double lp = -C[(size_t)i * m + j] / eps + f[i] / eps + g[j] / eps;
                double p = exp(lp);
                cost += p * C[(size_t)i * m + j];
                plan_sum += p;
                if (j == 0) u[i] = eps * (lp + C[(size_t)i * m + j] / eps);
                if (i == 0) v[j] = eps * (lp + C[(size_t)i * m + j] / eps);
            }
        }
        /* golden potentials for this eps */
        snprintf(path, sizeof path, "%s/verify/golden/kernel_potentials.csv", root);
        FILE *pf = fopen(path, "r");
        if (!pf) die("cannot open", path);
        char ph[MAXLINE], pl[MAXLINE], phc[MAXLINE];
        if (!fgets(ph, MAXLINE, pf)) die("empty file", path);
        snprintf(phc, sizeof phc, "%s", ph); int p_eps = column_of(phc, "eps");
        snprintf(phc, sizeof phc, "%s", ph); int p_ix = column_of(phc, "index");
        snprintf(phc, sizeof phc, "%s", ph); int p_u = column_of(phc, "u");
        snprintf(phc, sizeof phc, "%s", ph); int p_v = column_of(phc, "v");
        if (p_eps < 0 || p_ix < 0 || p_u < 0 || p_v < 0) die("missing column in", path);
        int seen = 0;
        while (fgets(pl, MAXLINE, pf)) {
            if (pl[0] == '\n' || pl[0] == '\r') continue;
            if (fabs(atof(field(pl, p_eps)) - eps) > 1e-12) continue;
            int i = atoi(field(pl, p_ix));
            if (i < 0 || i >= n) die("index out of range in", path);
            double gu = atof(field(pl, p_u)), gv = atof(field(pl, p_v));
            double a = fabs(gu - u[i]), b = fabs(gv - v[i]);
            if (a > du) du = a;
            if (b > dv) dv = b;
            seen++;
        }
        fclose(pf);
        if (seen != n) {
            printf("  eps %-5g FAIL: %d potential rows, expected %d\n", eps, seen, n);
            failures++;
            free(u); free(v);
            continue;
        }

        double rel_cost = fabs(cost - want_cost) / fabs(want_cost);
        /* Tolerances. The golden side is float32 torch, this side is double, so
         * the two cannot agree to better than single precision. 3e-5 on the
         * potentials and 1e-6 on the transport cost are what was measured. */
        int bad = 0;
        if (rel_cost > 1e-6) bad = 1;
        if (du > 3e-5 || dv > 3e-5) bad = 1;
        if (fabs(plan_sum - 1.0) > 1e-4) bad = 1;
        /* The iteration count is printed but not asserted. The stopping rule is
         * max|f - f_prev| < 1e-9, and a float32 run reaches that sooner than a
         * double one because its updates stall at the single precision noise
         * floor: at eps 0.5 the golden stops at 108 and this stops at 131. The
         * fixed point they both land on is the same, which is what is checked. */
        printf("  eps %-5g iters %3d (golden %3d)  cost %.9f rel %.2e  max|du| %.2e  max|dv| %.2e  %s\n",
               eps, iters, want_iters, cost, rel_cost, du, dv, bad ? "FAIL" : "ok");
        failures += bad;
        checked++;
        free(u); free(v);
    }
    fclose(sm);
    free(C); free(x); free(y); free(f); free(g); free(tmp);
    if (checked == 0) { printf("  nothing checked\n"); return 1; }
    if (failures) { printf("  %d of %d Sinkhorn runs disagree with the golden export\n", failures, checked); return 1; }
    printf("  C reproduces the Python plan for all %d regularisation levels\n", checked);
    return 0;
}
