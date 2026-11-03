"""Build the LaTeX tables of the revised paper from the results/ CSV files.

Writes to results/tex/:
  table1_body.tex          body of main Table 1 (ML vs MML87, matched rank)
  compare_summary_body.tex body of main Table 2 (comparison with alternative estimators)
  modelsel_rho2_body.tex   body of main Table 3 (pure noise and rho = 2)
  modelsel_rho5_body.tex   body of main Table 4 (rho = 5)
  asymptotic_body.tex      body of supplementary Table S1 (asymptotic checks)
  supp_tables_S2_S10.tex   supplementary Tables S2-S10 (complete floats)
Prints the Monte Carlo standard errors and the statistics quoted in the text.
Requires only the Python standard library.
"""
import csv
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results') + os.sep      # simulation CSV files
OUT = os.path.join(RES, 'tex') + os.sep                 # generated LaTeX

# Estimators shown in the comparison tables, as named in the CSV files (the
# DGJ shrinkers use the PLY residual variance, hence the 'P' suffix), and the
# matching column headings in the paper.
METHODS = ['ML', 'MML', 'BC', 'PLY', 'DGJentP', 'DGJstP']
HEAD = ['MLE', 'MML87', 'BC', 'PLY', 'DGJ-E', 'DGJ-S']
# CSV column -> (LaTeX name, number format) for the supplementary tables
METRICS = {'S2': ('$S_2$', '%.4f'),
           'KL': ('$\\operatorname{KL}(\\boldsymbol{\\Sigma}_0,\\widehat{\\boldsymbol{\\Sigma}})$', '%.3f'),
           'rKL': ('$\\operatorname{KL}(\\widehat{\\boldsymbol{\\Sigma}},\\boldsymbol{\\Sigma}_0)$', '%.3f')}


def load(fname, keys):
    """Read a comparison CSV into {setting: {method: row}}.

    A setting is the tuple of the values of KEYS (e.g. N, SNR, J); each CSV
    row holds one method's averages and Monte Carlo standard errors.
    """
    cells = defaultdict(dict)
    for r in csv.DictReader(open(RES + fname)):
        cells[tuple(float(r[k]) for k in keys)][r['Method']] = r
    return cells


def bold_min(vals, fmt):
    """Format VALS and bold every entry equal to the smallest after rounding."""
    s = [fmt % v for v in vals]
    best = min(s, key=float)
    return ['{\\bf %s}' % x if x == best else x for x in s]


def fmt_se(x):
    """Two significant figures in plain decimal notation (no exponent)."""
    return ('%.2g' % x) if x >= 1e-4 else '%.*f' % (1 - int(math.floor(math.log10(x))), x)


def num(x):
    """Shortest plain formatting of a setting value (e.g. 25, 0.5)."""
    return '%g' % x


# ---------------------------------------------------------------- main Table 1
def table1_body(cells):
    """Body of main Table 1: ML against MML87 for each (N, rho, J).

    Columns: regular-solution percentage, then S1, S2, forward KL and reverse
    KL for ML and MML87. In each pair the better value is bold (smaller |S1|,
    smaller S2 and KL), unless the two agree after rounding. N and rho are
    printed once per block with \\multirow.
    """
    out = []
    Ns = sorted({c[0] for c in cells}); rhos = sorted({c[1] for c in cells}); Js = sorted({c[2] for c in cells})
    for i, N in enumerate(Ns):
        for k, rho in enumerate(rhos):
            for j, J in enumerate(Js):
                c = (N, rho, J); g = lambda m, x: float(cells[c][m][x])
                lead = ('\\multirow{%d}{*}{%s}' % (len(rhos) * len(Js), num(N)) if k == 0 and j == 0 else '') + ' & ' + \
                       ('\\multirow{%d}{*}{%s}' % (len(Js), num(rho)) if j == 0 else '')
                parts = []
                # (metric, format, compare absolute values?) -- S1 is a signed bias
                for m, fmt, absv in [('S1', '%.3f', True), ('S2', '%.4f', False), ('KL', '%.3f', False), ('rKL', '%.3f', False)]:
                    a, b = g('ML', m), g('MML', m)
                    sa, sb = fmt % a, fmt % b
                    ka, kb = (abs(a), abs(b)) if absv else (a, b)
                    if sa.lstrip('-') != sb.lstrip('-'):
                        if ka < kb: sa = '{\\bf %s}' % sa
                        elif kb < ka: sb = '{\\bf %s}' % sb
                    parts.append('%s & %s' % (sa, sb))
                out.append('%s & %d & %.0f & %s \\\\' % (lead, J, g('ML', 'regular_pct'), ' & & '.join(parts)))
        if i < len(Ns) - 1:
            out += ['\\vspace{-2mm} \\\\', '\\cmidrule{2-15}', '\\vspace{-2mm} \\\\']
    return '\n'.join(out) + '\n'


def max_se(cells, methods, metrics):
    """Largest Monte Carlo standard error of each metric over all settings."""
    return {m: max(float(cells[c][k]['se_' + m]) for c in cells for k in methods) for m in metrics}


# ------------------------------------------------------- supplement tables
def supp_table(cells, keys, metric, caption, label):
    """One complete supplementary table (Tables S2-S10) for one metric.

    One row per setting, sorted by the setting keys; a key value is printed
    only when it changes from the previous row, and a rule separates the
    blocks of the first key. The smallest value in each row is bold, and the
    caption reports the largest Monte Carlo standard error in the table.
    """
    name, fmt = METRICS[metric]
    # DGJ shrinkers use the PLY residual variance, so their S2 equals PLY's
    meths = METHODS[:4] if metric == 'S2' else METHODS
    head = HEAD[:len(meths)]
    nk = len(keys)
    cols = 'c' * (nk + 1 + len(meths))
    lines = ['\\begin{table}[p]', '\\scriptsize', '\\begin{center}', '\\begin{tabular}{%s}' % cols, '\\toprule',
             ' & '.join(['$%s$' % k for k in keys] + ['Reg.\\ (\\%)'] + head) + ' \\\\',
             '\\cmidrule{1-%d}' % (nk + 1 + len(meths))]
    srt = sorted(cells, key=lambda c: c)
    prev = [None] * nk
    for i, c in enumerate(srt):
        if i > 0 and c[0] != srt[i - 1][0]:
            lines.append('\\midrule')
        lead = []
        for j in range(nk):
            # print key j if it or any key to its left changed since the last row
            show = any(c[t] != (prev[t] if prev[t] is not None else None) for t in range(j + 1)) or prev[0] is None
            lead.append(num(c[j]) if show else '')
        prev = list(c)
        vals = [float(cells[c][m][metric]) for m in meths]
        lines.append(' & '.join(lead + ['%.0f' % float(cells[c]['ML']['regular_pct'])] + bold_min(vals, fmt)) + ' \\\\')
    se = max(float(cells[c][m]['se_' + metric]) for c in cells for m in meths)
    lines += ['\\bottomrule', '\\end{tabular}',
              '\\caption{%s Bold marks the smallest value in each row. The largest Monte Carlo standard error is $%s$.}' % (caption, fmt_se(se)),
              '\\label{%s}' % label, '\\end{center}', '\\end{table}', '']
    return '\n'.join(lines)



# ------------------------------------------------------------ main Table 2
def compare_summary_body(designs):
    """Body of main Table 2, and the MML87 versus DGJ-S statistics in the text.

    For each design and criterion, counts the settings in which each
    estimator has the smallest average loss ('--' where a method is not
    compared: the DGJ shrinkers share PLY's S2). Also prints the median and
    largest relative gap |MML87/DGJ-S - 1| in reverse KL over the regular
    settings (a regular J-component MML87 fit in at least 90% of data sets).
    """
    out = []
    for d, (label, cells) in enumerate(designs):
        ks = sorted(cells)
        g = lambda k, m, x: float(cells[k][m][x])
        for r, (metric, mname) in enumerate([('S2', '$S_2$'), ('KL', 'Forward KL'), ('rKL', 'Reverse KL')]):
            meths = METHODS[:4] if metric == 'S2' else METHODS
            wins = {m: 0 for m in meths}
            for k in ks:
                wins[min(meths, key=lambda m: g(k, m, metric))] += 1
            cnt = [str(wins[m]) if m in wins else '--' for m in METHODS]
            lead = ('\\multirow{3}{*}{%s} & \\multirow{3}{*}{%d}' % (label, len(ks))) if r == 0 else ' & '
            out.append('%s & %s & %s \\\\' % (lead, mname, ' & '.join(cnt)))
        if d < len(designs) - 1:
            out.append('\\midrule')
        reg = [k for k in ks if g(k, 'ML', 'regular_pct') >= 90]
        gap = sorted(abs(100 * (g(k, 'MML', 'rKL') / g(k, 'DGJstP', 'rKL') - 1)) for k in reg)
        med = gap[len(gap) // 2] if len(gap) % 2 else (gap[len(gap) // 2 - 1] + gap[len(gap) // 2]) / 2
        print('Table 2, %s: %d settings, %d regular; |MML87/DGJ-S - 1| in reverse KL: median %.2f%%, max %.2f%%'
              % (label, len(ks), len(reg), med, gap[-1]))
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------- main Tables 3-4
MS_METHODS = ['MML', 'BIC', 'Bayes', 'GIC', 'SPA']


def modelsel_tables(fname='final_modelsel.csv'):
    """Bodies of main Tables 3 (pure noise and rho = 2) and 4 (rho = 5).

    For each N and generating J, one row per method with the forward and
    reverse KL divergences and the percentages of under-, exact and
    over-selection; the smallest KL values and the highest exact-recovery
    rate in each block are bold. Also prints the smallest percentage of data
    sets in which the MML87 rank equals the largest admissible rank, and the
    largest Monte Carlo standard error of the KL columns (SPA excluded).
    """
    cells = defaultdict(dict)
    for r in csv.DictReader(open(RES + fname)):
        cells[(int(r['N']), r['SNR'], int(r['J']))][r['Method']] = r
    M = MS_METHODS

    def body(rho, include0):
        # rows for signal-to-noise ratio RHO, preceded by the pure-noise
        # block (J = 0) when INCLUDE0
        out=[]; Ns=sorted({c[0] for c in cells})
        for i,N in enumerate(Ns):
            keys=[]
            if include0: keys.append((N,[c for c in cells if c[0]==N and c[2]==0][0][1],0))
            keys+=sorted([c for c in cells if c[0]==N and c[2]>0 and abs(float(c[1])-rho)<1e-9],key=lambda c:c[2])
            nrow=len(keys)*len(M)
            for k,c in enumerate(keys):
                g=lambda m,x: float(cells[c][m][x])
                # best values in this block, for bolding
                bk=min(g(m,'KL') for m in M); br=min(g(m,'rKL') for m in M); be=max(g(m,'exact') for m in M)
                for j,m in enumerate(M):
                    lead=(f'\\multirow{{{nrow}}}{{*}}{{{N}}}' if k==0 and j==0 else '')+' & '+(f'\\multirow{{{len(M)}}}{{*}}{{{c[2]}}}' if j==0 else '')
                    kl='%.3f'%g(m,'KL'); rk='%.3f'%g(m,'rKL'); ex='%.2f'%g(m,'exact')
                    if kl=='%.3f'%bk: kl='{\\bf '+kl+'}'
                    if rk=='%.3f'%br: rk='{\\bf '+rk+'}'
                    if ex=='%.2f'%be: ex='{\\bf '+ex+'}'
                    un='--' if c[2]==0 else '%.2f'%g(m,'under')
                    out.append(f"{lead} & {m} & {kl} & {rk} & & {un} & {ex} & {'%.2f'%g(m,'over')} \\\\")
            if i<len(Ns)-1: out+=['\\vspace{-2mm} \\\\','\\cmidrule{2-9}','\\vspace{-2mm} \\\\']
        return '\n'.join(out)+'\n'

    open(OUT + 'modelsel_rho2_body.tex', 'w').write(body(2, True))
    open(OUT + 'modelsel_rho5_body.tex', 'w').write(body(5, False))
    thr = [float(c['MML']['mml_is_threshold_pct']) for c in cells.values()]
    print('Model selection: MML rank = largest admissible rank in at least %.2f%% of data sets' % min(thr))
    se = max(float(cells[c][m][k]) for c in cells for m in M[:4] for k in ('se_KL', 'se_rKL'))
    print('Model selection: largest MCSE of KL excluding SPA = %.4f' % se)


# ------------------------------------------------- supplementary Table S1
# Rows of Table S1: (regime, quantity in final_asymptotic.csv, label, format)
ASYM_ROWS = [
    ('fixed', 'var_tau', r'$\operatorname{Var}\{\sqrt N(\widehat\tau-\tau)\}$', '%.3f'),
    ('fixed', 'var_x1', r'$\operatorname{Var}\{\sqrt N(\widehat x_1-x_1)\}$', '%.1f'),
    ('fixed', 'var_x2', r'$\operatorname{Var}\{\sqrt N(\widehat x_2-x_2)\}$', '%.1f'),
    ('fixed', 'cov_tau_x1', r'$\operatorname{Cov}\{\sqrt N(\widehat\tau-\tau),\sqrt N(\widehat x_1-x_1)\}$', '%.3f'),
    ('fixed', 'cov_x1_x2', r'$\operatorname{Cov}\{\sqrt N(\widehat x_1-x_1),\sqrt N(\widehat x_2-x_2)\}$', '%.3f'),
    ('fixed', 'coverage', r'Coverage of the 95\% interval for $\tau$', 'pct'),
    ('proportional', 'mean_Ntau', r'Mean of $N(\widehat\tau-\tau)$', '%.3f'),
    ('proportional', 'var_Ntau', r'$\operatorname{Var}\{N(\widehat\tau-\tau)\}$', '%.2f'),
    ('proportional', 'var_x1', r'$\operatorname{Var}\{\sqrt N(\widehat x_1-\tau z_1)\}$', '%.1f'),
    ('proportional', 'var_x2', r'$\operatorname{Var}\{\sqrt N(\widehat x_2-\tau z_2)\}$', '%.1f'),
]


def asymptotic_body(fname='final_asymptotic.csv'):
    """Body of supplementary Table S1: simulated moments against their limits.

    One row per quantity, with the limit from Theorems 2-3 and the estimate
    at each N. Prints the percentage of regular fits and the Monte Carlo
    standard errors quoted in the caption.
    """
    rows = list(csv.DictReader(open(RES + fname)))
    Ns = sorted({int(r['N']) for r in rows})
    get = {(r['regime'], r['quantity'], int(r['N'])): r for r in rows}
    fmt = lambda f, v: ('$%.0f\\%%$' % (100 * v) if v == 0.95 else '$%.1f\\%%$' % (100 * v)) if f == 'pct' else '$' + f % v + '$'
    head = {'fixed': r'\multicolumn{%d}{l}{\emph{Fixed dimension:} $K=6$, $J=2$, $\tau=1$, $(\rho_1,\rho_2)=(5,2)$}\\',
            'proportional': r'\multicolumn{%d}{l}{\emph{Proportional regime:} $K=N/2$, $J=2$, $\tau=1$, $(\rho_1,\rho_2)=(8,4)$}\\'}
    out, prev = [], None
    for reg, q, label, f in ASYM_ROWS:
        if reg != prev:
            if prev: out.append(r'\midrule')
            out.append(head[reg] % (len(Ns) + 2)); prev = reg
        lim = float(get[(reg, q, Ns[0])]['limit'])
        est = [fmt(f, float(get[(reg, q, N)]['estimate'])) for N in Ns]
        out.append(' & '.join([label, fmt(f, lim)] + est) + r' \\')
    for N in Ns:
        print('Table S1, N=%d: regular fits %s' % (N, ', '.join('%s %.2f%%' % (g, float(get[(g, 'var_x1', N)]['regular_pct'])) for g in ('fixed', 'proportional'))))
    for reg, q, _, _ in ASYM_ROWS:
        print('Table S1 MCSE %-12s %-10s %s' % (reg, q, ' '.join('%.4f' % float(get[(reg, q, N)]['mcse']) for N in Ns)))
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    eq = load('final_compare_equal.csv', ['N', 'SNR', 'J'])
    open(OUT + 'table1_body.tex', 'w').write(table1_body(eq))
    print('Table 1 max MCSE (ML, MML):', max_se(eq, ['ML', 'MML'], ['S1', 'S2', 'KL', 'rKL']))

    out = []
    geo = load('final_compare_geometric.csv', ['N', 'SNR', 'J'])
    hd = load('final_compare_highdim.csv', ['K', 'N', 'SNR', 'J'])
    open(OUT + 'compare_summary_body.tex', 'w').write(compare_summary_body(
        [('Equal', eq), ('Geometric', geo), ('Higher-dimensional', hd)]))
    for design, cells, keys, desc in [
            ('equal', eq, ['N', '\\rho', 'J'], 'equal per-component signal-to-noise ratio $\\rho$ and $K=10$'),
            ('geometric', geo, ['N', '\\rho', 'J'], 'geometric loadings with weakest-component signal-to-noise ratio $\\rho$, ratio $0.75$, and $K=10$'),
            ('highdim', hd, ['K', 'N', '\\rho', 'J'], 'equal per-component signal-to-noise ratio $\\rho$ and $N\\in\\{2K,10K\\}$')]:
        for metric, mname in [('S2', 'Squared log-scale error $S_2$ of the residual scale'),
                              ('KL', 'Forward KL divergence (half the entropy loss)'),
                              ('rKL', 'Reverse KL divergence (half the Stein loss)')]:
            cap = '%s at a matched rank, averaged over $10^5$ simulated data sets with %s.' % (mname, desc)
            out.append(supp_table(cells, keys, metric, cap, 'tab:supp:%s:%s' % (design, metric)))
    open(OUT + 'supp_tables_S2_S10.tex', 'w').write('\n'.join(out))
    print('supplement tables written')
    modelsel_tables()
    print('model-selection tables written to', OUT)
    if os.path.exists(RES + 'final_asymptotic.csv'):
        open(OUT + 'asymptotic_body.tex', 'w').write(asymptotic_body())
        print('Table S1 written')
