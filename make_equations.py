"""Equation images for the report (sections 4.3 and 4.4): docs/figures/eq_norm.png, eq_filter.png."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
OUT = Path(__file__).resolve().parent / "docs" / "figures"
EQS = {
    "eq_norm": (r"$\mathbf{q}_i \;=\; \dfrac{\mathbf{p}_i - \mathbf{p}_0}{\Vert\mathbf{p}_9 - \mathbf{p}_0\Vert},\qquad i = 0,\ldots,20$", "(1)"),
    "eq_filter": (r"$f_c \;=\; f_{c,\mathrm{min}} \;+\; \beta\,\left\vert\dfrac{dx}{dt}\right\vert$", "(2)"),
}
for name, (tex, num) in EQS.items():
    fig = plt.figure(figsize=(6.5, 0.9))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=15)
    fig.text(0.985, 0.5, num, ha="right", va="center", fontsize=13)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
print("ok")
