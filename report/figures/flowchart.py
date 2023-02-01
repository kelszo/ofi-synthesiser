import graphviz
import pandas as pd

swetrau = pd.read_csv("data/raw/combined_dataset.csv")

n_swetrau = len(swetrau)

# Remove not screened
screened_ofi = swetrau[swetrau["ofi"].notna()]
n_not_screened = n_swetrau - len(screened_ofi)
screened_ofi["ofi"] = screened_ofi["ofi"].replace("No", 0)
screened_ofi["ofi"] = screened_ofi["ofi"].replace("Yes", 1)

over_14 = screened_ofi[(screened_ofi["pt_age_yrs"] > 14) | (screened_ofi["pt_age_yrs"].isna())]
n_under_15 = len(screened_ofi) - len(over_14)

excluded = n_not_screened + n_under_15

eligible = over_14
n_eligible = len(eligible)

filter_col = [col for col in screened_ofi if col.startswith("VK_")]

eligible[filter_col] = eligible[filter_col].replace(
    {"Nej": False, "Ja": True, "nej": False, "ja": True, "nn": False, "nj\nNej": False}
)

eligible = eligible.fillna(False)

filter_col.remove("VK_annat")
filter_col.remove("VK_avslutad")

n_audit_flagged = (eligible[filter_col].sum(axis=1) >= 1).sum()

n_nurse_flagged = ((eligible[filter_col].sum(axis=1) == 0) & eligible["VK_annat"]).sum()

n_review = n_audit_flagged + n_nurse_flagged

allneg_but_ofi = ((eligible[filter_col].sum(axis=1) == 0) & eligible["ofi"]).sum()

dead = eligible[eligible["res_survival"] == 1]
alive = eligible[eligible["res_survival"] != 1]

n_dead = len(dead)
n_alive = len(alive)

n_dead_ofi = len(dead[dead["ofi"] == 1])
n_dead_no_ofi = len(dead[dead["ofi"] == 0])

n_ofi = len(screened_ofi[screened_ofi["ofi"] == 1])
n_no_ofi = len(screened_ofi[screened_ofi["ofi"] == 0])

dot = graphviz.Digraph(
    format="pdf",
    filename="flowchart",
    comment="Flowchart",
    directory="report/figures",
    graph_attr=[
        ("layout", "dot"),
        ("splines", "ortho"),
        ("nodesep", "0.6"),
    ],
    node_attr=[
        ("shape", "box"),
        ("fontsize", "12"),
        ("width", "2.75"),
    ],
    edge_attr=[
        ("arrowsize", "0.7"),
        ("arrowhead", "vee"),
        ("labelfontsize", "11"),
    ],
)

dot.node("swetrau", label=f"Trauma Registry (n={n_swetrau})")


with dot.subgraph() as c:
    c.attr(rank="same")
    c.node(
        "excluded",
        label=f"""<
Excluded (n={excluded})<br ALIGN='LEFT'/>
&#8226; Not screened for OFI (n={n_not_screened})<br ALIGN='LEFT'/>
&#8226; Under the age of 15 (n={n_under_15})<br ALIGN='LEFT'/>
>""",
    )
    c.node("1", shape="point", width="0", height="0")

    c.edge("1", "excluded", minlen="2")

dot.edge("swetrau", "1", arrowhead="none")
dot.edge("1", "eligible")

dot.node("alive", label=f"Survival* (n={n_alive})")
dot.node("death", label=f"Death* (n={n_dead})")
dot.node("eligible", label=f"Eligible (n={n_eligible})")
dot.node("af", label=f"Flagged By Audit filter (n={n_audit_flagged})")
dot.node("nurse", label=f"Flagged by nurse (n={n_nurse_flagged})")
dot.node("review", label=f"Review by two nurses (n={n_review})")
dot.node("mbc", label="Morbidity conference (n=?)")
dot.node("mtc", label=f"Mortality conference (n={n_dead})")
dot.node("ofi", label=f"OFI (n={n_ofi})")
dot.node("nofi", label=f"No OFI (n={n_no_ofi})")

dot.edges([("eligible", "alive"), ("alive", "af"), ("af", "review")])
dot.edges([("eligible", "death"), ("death", "mtc")])
dot.edges([("alive", "nurse"), ("nurse", "review")])
dot.edge("review", "nofi")


dot.edge("alive", "nofi")

dot.edge("review", "mbc")

dot.edge("mtc", "ofi")
dot.edge("mtc", "nofi")

dot.edge("mbc", "ofi")
dot.edge("mbc", "nofi")

dot.render()
