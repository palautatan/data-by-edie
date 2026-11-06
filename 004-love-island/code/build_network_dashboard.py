"""Build a local, offline dashboard from the saved notebook 07 and 08 tables."""
from pathlib import Path
import json
import duckdb
import pandas as pd
from plotly.offline import get_plotlyjs


def build_network_dashboard(project_root):
    project_root = Path(project_root)
    database_path = project_root / "data/love_island_reddit.duckdb"
    template_directory = project_root / "code/dashboard"
    output_directory = project_root / "viz/dashboard"
    output_directory.mkdir(parents=True, exist_ok=True)

    # 1. Read saved results only. Building the dashboard never refits models.
    table_names = {
        "overall_nodes": "overall_network_nodes",
        "episode_nodes": "episode_network_nodes",
        "overall_edges": "overall_network_edges",
        "episode_edges": "episode_network_edges",
        "probabilities": "logistic_episode_probabilities",
        "cox": "cox_survival_probabilities",
        "km": "km_survival_probabilities",
        "evaluation": "logistic_model_evaluation",
    }
    datasets = {}
    with duckdb.connect(str(database_path), read_only=True) as connection:
        for dataset_name, table_name in table_names.items():
            datasets[dataset_name] = connection.execute(f"SELECT * FROM dashboard.{table_name}").df()

    # 2. Keep the exact positions, communities, names and metrics saved in 07.
    lifecycle = pd.read_csv(project_root / "data/metadata/islander_metadata.csv")
    lifecycle_columns = ["islander", "entrance_episode", "exit_episode"]
    datasets["lifecycle"] = lifecycle[lifecycle_columns].copy()
    overall_nodes = datasets["overall_nodes"]
    assert overall_nodes["target_islander"].is_unique
    assert not datasets["episode_nodes"].duplicated(["target_islander", "episode"]).any()
    assert overall_nodes[["x", "y"]].notna().all().all()
    islander_names = set(overall_nodes["target_islander"])
    assert islander_names.issubset(set(lifecycle["islander"]))
    for dataset_name in ["overall_edges", "episode_edges"]:
        edge_names = set(datasets[dataset_name]["source"])
        edge_names.update(datasets[dataset_name]["target"])
        assert edge_names.issubset(islander_names)

    # 3. Confirm next-episode alignment. Episode k displays the forecast for k+1.
    probabilities = datasets["probabilities"]
    assert probabilities["target_episode"].eq(probabilities["feature_episode"] + 1).all()
    assert not probabilities.duplicated(["target_islander", "feature_episode"]).any()
    predicted_rows = probabilities["prediction_status"].eq("predicted")
    available_probabilities = probabilities.loc[predicted_rows, "booted_next_episode_probability"]
    assert available_probabilities.between(0, 1).all()
    training_episodes = probabilities.loc[predicted_rows, "training_last_outcome_episode"]
    target_episodes = probabilities.loc[predicted_rows, "target_episode"]
    assert training_episodes.lt(target_episodes).all()

    # 4. JSON converts missing values to null, never a fake zero probability.
    dashboard_data = {}
    for dataset_name, dataset in datasets.items():
        dashboard_data[dataset_name] = json.loads(dataset.to_json(orient="records", date_format="iso"))
    serialized_data = json.dumps(dashboard_data, ensure_ascii=False, allow_nan=False)
    serialized_data = serialized_data.replace("<", "\\u003c")

    # 5. Bundle scripts locally so the HTML also works without an internet connection.
    template = (template_directory / "index.html").read_text(encoding="utf-8")
    stylesheet = (template_directory / "dashboard.css").read_text(encoding="utf-8")
    javascript = (template_directory / "dashboard.js").read_text(encoding="utf-8")
    dashboard_html = template.replace("__STYLES__", stylesheet)
    dashboard_html = dashboard_html.replace("__PLOTLY__", get_plotlyjs())
    dashboard_html = dashboard_html.replace("__DATA__", serialized_data)
    dashboard_html = dashboard_html.replace("__SCRIPT__", javascript)
    dashboard_path = output_directory / "index.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")

    # Save a short audit beside the page so coverage is easy to check.
    audit = {
        "islanders": len(overall_nodes),
        "episodes": sorted(datasets["episode_nodes"]["episode"].unique().tolist()),
        "overall_edges": len(datasets["overall_edges"]),
        "episode_edges": len(datasets["episode_edges"]),
        "forecast_alignment": "feature_episode k -> target_episode k+1",
        "model_version": probabilities["model_version"].iloc[0],
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (output_directory / "validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Dashboard saved: {dashboard_path}")
    print(f"Preserved all {len(overall_nodes)} islanders from notebook 07.")
    return dashboard_path


if __name__ == "__main__":
    build_network_dashboard(Path(__file__).resolve().parents[1])
