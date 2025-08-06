# -*- coding: utf-8 -*-
"""
pull_best_static.py
"""
import pathlib
import MySQLdb
import pandas as pd

# ---------------- INPUT ---------------------------------
elements   = {'Al', 'Fe', 'Ni'}             # нужный набор
out_folder = pathlib.Path('unittests/poscars')
out_folder.mkdir(exist_ok=True)

# ---------------- DB CONNECT ----------------------------
db = MySQLdb.connect(
    host   = "localhost",
    user   = "oqmd_user",
    passwd = "MaDnEsS!!",     # пароль через env
    db     = "oqmd",
    charset= "utf8mb4"
)
cur = db.cursor()

# ---------------- SQL: все static -----------------------
elem_regex = '|'.join(elements)                        # Al|Fe|Ni
list_regex = fr'^({elem_regex})_(?:({elem_regex})_)*$' # ^(Al|Fe|Ni)_(...)*$

sql = f"""
SELECT
    e.id                       AS entry_id,
    c.formula,
    COALESCE(cal.natoms, s.natoms, 0) AS natoms,
    cal.energy_pa,
    cal.energy,
    s.x1, s.y1, s.z1,
    s.x2, s.y2, s.z2,
    s.x3, s.y3, s.z3,
    s.id AS structure_id
FROM calculations cal
JOIN entries       e  ON e.id = cal.entry_id
JOIN compositions   c ON c.formula = cal.composition_id
LEFT JOIN structures s ON s.entry_id = e.id
WHERE cal.configuration = 'static'
  AND c.element_list REGEXP '{list_regex}';
"""

cur.execute(sql)
rows  = cur.fetchall()
cols  = [d[0] for d in cur.description]
df    = pd.DataFrame(rows, columns=cols)

# ---------------- select best per formula ---------------
best_idx = df.groupby('formula')['energy_pa'].idxmin()
best     = df.loc[best_idx].reset_index(drop=True)

# ---------------- helper: POSCAR writer -----------------
def write_poscar(row, atom_rows):
    lines = [row.formula, '1.0']
    # ячейка
    cell = [(row.x1, row.y1, row.z1),
            (row.x2, row.y2, row.z2),
            (row.x3, row.y3, row.z3)]
    lines += ['{:12.8f} {:12.8f} {:12.8f}'.format(*v) for v in cell]

    # элементы и координаты
    elements_order, coords = [], []
    for el, x, y, z in atom_rows:
        coords.append((el, x, y, z))
        if el not in elements_order:
            elements_order.append(el)
    counts = [sum(1 for el, *_ in coords if el == sym) for sym in elements_order]

    lines.append(' '.join(elements_order))
    lines.append(' '.join(str(c) for c in counts))
    lines.append('Direct')
    for sym in elements_order:
        for el, x, y, z in coords:
            if el == sym:
                lines.append(f'{x:12.8f} {y:12.8f} {z:12.8f}')

    fname = out_folder / f"{row.formula.replace(' ', '')}_{row.entry_id}.POSCAR"
    fname.write_text('\n'.join(lines))

# ---------------- pull atoms & write POSCAR -------------
atom_sql = """
SELECT element_id, x, y, z
FROM atoms
WHERE structure_id = %s
ORDER BY id;
"""

for _, row in best.iterrows():
    # ← 1. нет структуры ‒ пропускаем
    if pd.isna(row.structure_id):
        continue

    cur = db.cursor()
    cur.execute(atom_sql, (int(row.structure_id),))
    atom_rows = cur.fetchall()
    cur.close()

    if atom_rows:
        write_poscar(row, atom_rows)

db.close()

# -------- вывод таблицы ------------
print(f"{'Formula':<15} {'N_at':>5} {'Energy (eV)':>15} {'Source':>8} {'EntryID':>8}")
for _, r in best.iterrows():
    n = 0 if pd.isna(r.natoms) else int(r.natoms)       # 2. NaN → 0
    print(f"{r.formula:<15} {n:5d} {r.energy:15.6f} {'OQMD':>8} {int(r.entry_id):8d}")

print(f"\nPOSCAR files written to: {out_folder.resolve()}")
