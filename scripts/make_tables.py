"""Build the LaTeX tables of the revised paper from the results/ CSV files.

Writes to results/tex/:
  table1_body.tex          body of main Table 1 (ML vs MML87, matched rank)
  modelsel_rho2_body.tex   body of main Table 2 (pure noise and rho = 2)
  modelsel_rho5_body.tex   body of main Table 3 (rho = 5)
  supp_tables_S1_S9.tex    supplementary Tables S1-S9 (complete floats)
Prints the Monte Carlo standard errors and the statistics quoted in the text.
Requires only the Python standard library.
"""
import csv
from collections import defaultdict

import os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results') + os.sep
OUT = os.path.join(RES, 'tex') + os.sep
METHODS = ['ML', 'MML', 'BC', 'PLY', 'DGJentP', 'DGJstP']
HEAD = ['MLE', 'MML87', 'BC', 'PLY', 'DGJ-E', 'DGJ-S']
METRICS = {'S2': ('$S_2$', '%.4f'),
           'KL': ('$\\operatorname{KL}(\\boldsymbol{\\Sigma}_0,\\widehat{\\boldsymbol{\\Sigma}})$', '%.3f'),
           'rKL': ('$\\operatorname{KL}(\\widehat{\\boldsymbol{\\Sigma}},\\boldsymbol{\\Sigma}_0)$', '%.3f')}


def load(fname, keys):
    cells = defaultdict(dict)
    for r in csv.DictReader(open(RES + fname)):
        cells[tuple(float(r[k]) for k in keys)][r['Method']] = r
    return cells


def bold_min(vals, fmt):
    s = [fmt % v for v in vals]
    best = min(s, key=float)
    return ['{\\bf %s}' % x if x == best else x for x in s]


def num(x):
    return '%g' % x


# ---------------------------------------------------------------- main Table 1
def table1_body(cells):
    out = []
    Ns = sorted({c[0] for c in cells}); rhos = sorted({c[1] for c in cells}); Js = sorted({c[2] for c in cells})
    for i, N in enumerate(Ns):
        for k, rho in enumerate(rhos):
            for j, J in enumerate(Js):
                c = (N, rho, J); g = lambda m, x: float(cells[c][m][x])
                lead = ('\\multirow{%d}{*}{%s}' % (len(rhos) * len(Js), num(N)) if k == 0 and j == 0 else '') + ' & ' + \
                       ('\\multirow{%d}{*}{%s}' % (len(Js), num(rho)) if j == 0 else '')
                parts = []
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
    return {m: max(float(cells[c][k]['se_' + m]) for c in cells for k in methods) for m in metrics}


# ------------------------------------------------------- supplement tables
def supp_table(cells, keys, metric, caption, label):
    name, fmt = METRICS[metric]
    # DGJ shrinkers use the PLY noise variance, so their S2 equals PLY's
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
            show = any(c[t] != (prev[t] if prev[t] is not None else None) for t in range(j + 1)) or prev[0] is None
            lead.append(num(c[j]) if show else '')
        prev = list(c)
        vals = [float(cells[c][m][metric]) for m in meths]
        lines.append(' & '.join(lead + ['%.0f' % float(cells[c]['ML']['regular_pct'])] + bold_min(vals, fmt)) + ' \\\\')
    se = max(float(cells[c][m]['se_' + metric]) for c in cells for m in meths)
    lines += ['\\bottomrule', '\\end{tabular}',
              '\\caption{%s Bold marks the smallest value in each row. The largest Monte Carlo standard error is $%s$.}' % (caption, '%.2g' % se),
              '\\label{%s}' % label, '\\end{center}', '\\end{table}', '']
    return '\n'.join(lines)



# ---------------------------------------------------------- main Tables 2-3
MS_METHODS = ['MML', 'BIC', 'Bayes', 'GIC', 'SPA']


def modelsel_tables(fname='final_modelsel.csv'):
    cells = defaultdict(dict)
    for r in csv.DictReader(open(RES + fname)):
        cells[(int(r['N']), r['SNR'], int(r['J']))][r['Method']] = r
    M = MS_METHODS

    def body(rho, include0):
        out=[]; Ns=sorted({c[0] for c in cells})
        for i,N in enumerate(Ns):
            keys=[]
            if include0: keys.append((N,[c for c in cells if c[0]==N and c[2]==0][0][1],0))
            keys+=sorted([c for c in cells if c[0]==N and c[2]>0 and abs(float(c[1])-rho)<1e-9],key=lambda c:c[2])
            nrow=len(keys)*len(M)
            for k,c in enumerate(keys):
                g=lambda m,x: float(cells[c][m][x])
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


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    eq = load('final_compare_equal.csv', ['N', 'SNR', 'J'])
    open(OUT + 'table1_body.tex', 'w').write(table1_body(eq))
    print('Table 1 max MCSE (ML, MML):', max_se(eq, ['ML', 'MML'], ['S1', 'S2', 'KL', 'rKL']))

    out = []
    geo = load('final_compare_geometric.csv', ['N', 'SNR', 'J'])
    hd = load('final_compare_highdim.csv', ['K', 'N', 'SNR', 'J'])
    for design, cells, keys, desc in [
            ('equal', eq, ['N', '\\rho', 'J'], 'equal per-component signal-to-noise ratio $\\rho$ and $K=10$'),
            ('geometric', geo, ['N', '\\rho', 'J'], 'geometric loadings with weakest-component signal-to-noise ratio $\\rho$, ratio $0.75$, and $K=10$'),
            ('highdim', hd, ['K', 'N', '\\rho', 'J'], 'equal per-component signal-to-noise ratio $\\rho$ and $N\\in\\{2K,10K\\}$')]:
        for metric, mname in [('S2', 'Squared log-scale error $S_2$ of the residual scale'),
                              ('KL', 'Forward KL divergence (half the entropy loss)'),
                              ('rKL', 'Reverse KL divergence (half the Stein loss)')]:
            cap = '%s at a matched rank, averaged over $10^5$ simulated data sets with %s.' % (mname, desc)
            out.append(supp_table(cells, keys, metric, cap, 'tab:supp:%s:%s' % (design, metric)))
    open(OUT + 'supp_tables_S1_S9.tex', 'w').write('\n'.join(out))
    print('supplement tables written')
    modelsel_tables()
    print('model-selection tables written to', OUT)
