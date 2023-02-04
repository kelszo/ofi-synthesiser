import graphviz
import pandas as pd

dot = graphviz.Digraph(
    format="pdf",
    filename="model_flowchart",
    comment="Model Flowchart",
    directory="report/figures",
    graph_attr=[("layout", "dot"), ("splines", "ortho"), ("rankdir", "lr"), ("nodesep", "1")],
    node_attr=[("shape", "record"), ("fontsize", "12")],
    edge_attr=[
        ("arrowsize", "0.5"),
        ("arrowhead", "normal"),
        ("labelfontsize", "11"),
    ],
)

dot.attr(rankdir="LR")

dot.node("swetrau", label="Trauma Registry")
dot.node("invis1", height="0", width="0", shape="point")
dot.edge("swetrau", "invis1", arrowhead="none")

dot.node("tvaecombdata", label="Data* + training data")


with dot.subgraph() as c:
    c.attr(rank="same")
    c.node("evalmodel", label="Evaluate Classification Model")
    c.node("trainmodel", label="Train Classification Model")

with dot.subgraph() as c:
    c.attr(rank="same")
    c.node("traindata", label="Train data split")
    c.node("testdata", label="Test data split")

dot.edges(
    (
        ["invis1", "traindata"],
        ["invis1", "testdata"],
    )
)


with dot.subgraph() as c:
    c.node("ctgan", label="Train CTGAN")
    c.node("ctgansyndata", label="Produce Data*")


with dot.subgraph() as c:
    c.node("tvae", label="Train TVAE")
    c.node("tvaesyndata", label="Produce Data*")
dot.node("ctgancombdata", label="Data* + training data")


dot.node("invis2", height="0", width="0", shape="point")
dot.edge("traindata", "invis2", arrowhead="none")

dot.edge("invis2", "trainmodel")

dot.edge("ctgansyndata", "trainmodel")
dot.edge("tvaesyndata", "trainmodel")

dot.edge("invis2", "ctgan")
dot.edge("ctgan", "ctgansyndata")
dot.edge("ctgansyndata", "ctgancombdata")


dot.edge("invis2", "tvae")
dot.edge("tvae", "tvaesyndata")
dot.edge("tvaesyndata", "tvaecombdata")


dot.edge("ctgancombdata", "trainmodel")
dot.edge("tvaecombdata", "trainmodel")


dot.edge("trainmodel", "evalmodel")
dot.edge("testdata", "evalmodel")


dot.render()
