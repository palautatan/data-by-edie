"""Build survival datasets for inspection. This script does not fit a model.

Call build_survival_dataset(project_root) with your original project's folder.
The function saves the same database tables and CSV files as the original.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def build_survival_dataset(project_root):
    # 1. Read the islander metadata.
    project_root = Path(project_root)
    database_path = project_root / "data/love_island_reddit.duckdb"
    metadata_path = project_root / "data/metadata/islander_metadata.csv"

    subjects = pd.read_csv(metadata_path)
    subjects = subjects.rename(columns={"islander": "target_islander"})

    # A blank final placement is treated as an observed permanent exit.
    final_placement = subjects["final_placement"].fillna("").str.strip()
    subjects["event_observed"] = (final_placement == "").astype(int)
    subjects["censored"] = 1 - subjects["event_observed"]
    subjects["duration_episodes"] = (
        subjects["exit_episode"] - subjects["entrance_episode"] + 1
    )

    assert subjects["target_islander"].is_unique
    assert (subjects["duration_episodes"] > 0).all()

    # 2. Read the comment windows, existing scores, and dashboard features.
    schedule_query = """
        SELECT episode, MIN(created_at) AS cutoff_proxy
        FROM comments
        WHERE episode IS NOT NULL
        GROUP BY episode
        ORDER BY episode
    """

    scores_query = """
        SELECT
            windows.window_id,
            windows.comment_id,
            windows.target_islander,
            windows.episode,
            windows.created_at,
            roberta.roberta_sentiment,
            vader.vader_compound,
            detoxify.detoxify_toxicity,
            detoxify.detoxify_insult
        FROM islander_sentiment_windows AS windows
        LEFT JOIN roberta_window_scores AS roberta
            ON windows.window_id = roberta.window_id
            AND windows.model_text = roberta.model_text
        LEFT JOIN vader_window_scores AS vader
            ON windows.window_id = vader.window_id
        LEFT JOIN detoxify_window_scores AS detoxify
            ON windows.window_id = detoxify.window_id
        WHERE windows.keep_analysis = 'Y'
    """

    dashboard_table_names = [
        "episode_relative_metrics",
        "episode_network_nodes",
        "episode_community_approval",
        "kalshi_episode_prices",
    ]
    dashboard_tables = {}

    with duckdb.connect(str(database_path), read_only=True) as connection:
        schedule = connection.execute(schedule_query).df()
        windows = connection.execute(scores_query).df()

        for table_name in dashboard_table_names:
            query = f"SELECT * FROM dashboard.{table_name}"
            dashboard_tables[table_name] = connection.execute(query).df()

    assert windows["window_id"].is_unique

    # 3. Keep prior-episode comments posted before the next episode's cutoff.
    # This cutoff is a discussion timestamp proxy, not verified airtime.
    cutoffs = schedule.rename(columns={"episode": "target_episode"})
    windows["target_episode"] = windows["episode"] + 1
    windows = windows.merge(
        cutoffs,
        on="target_episode",
        how="left",
        validate="many_to_one",
    )
    windows["available_before_cutoff"] = (
        windows["created_at"] < windows["cutoff_proxy"]
    )

    timing_audit = windows.groupby("episode", as_index=False).agg(
        total_windows=("window_id", "size"),
        eligible_windows=("available_before_cutoff", "sum"),
    )
    timing_audit["excluded_windows"] = (
        timing_audit["total_windows"] - timing_audit["eligible_windows"]
    )
    eligible_windows = windows.loc[windows["available_before_cutoff"]].copy()

    # 4. Average windows within a comment, then comments within an episode.
    comment_scores = eligible_windows.groupby(
        ["target_islander", "episode", "comment_id"], as_index=False
    ).agg(
        roberta_sentiment=("roberta_sentiment", "mean"),
        vader_compound=("vader_compound", "mean"),
        detoxify_toxicity=("detoxify_toxicity", "mean"),
        detoxify_insult=("detoxify_insult", "mean"),
        latest_timestamp=("created_at", "max"),
    )

    # Missing toxicity stays missing; it does not become a non-toxic score.
    comment_scores["toxic_flag"] = (
        comment_scores["detoxify_toxicity"] >= 0.5
    ).astype(float)
    missing_toxicity = comment_scores["detoxify_toxicity"].isna()
    comment_scores.loc[missing_toxicity, "toxic_flag"] = np.nan

    episode_features = comment_scores.groupby(
        ["target_islander", "episode"], as_index=False
    ).agg(
        asof_num_comments=("comment_id", "size"),
        asof_vader_mean=("vader_compound", "mean"),
        asof_vader_sd=("vader_compound", "std"),
        asof_roberta_mean=("roberta_sentiment", "mean"),
        asof_roberta_sd=("roberta_sentiment", "std"),
        asof_toxicity_mean=("detoxify_toxicity", "mean"),
        asof_insult_mean=("detoxify_insult", "mean"),
        asof_toxic_comment_share=("toxic_flag", "mean"),
        asof_latest_timestamp=("latest_timestamp", "max"),
    )

    total_comments_by_episode = episode_features.groupby("episode")[
        "asof_num_comments"
    ].transform("sum")
    episode_features["asof_attention_share"] = (
        episode_features["asof_num_comments"] / total_comments_by_episode
    )

    # 4b. Count distinct co-mentions using the same time-filtered comment population.
    mention_columns = ["episode", "comment_id", "target_islander"]
    eligible_mentions = eligible_windows[mention_columns].drop_duplicates()
    mentions_per_comment = eligible_mentions.groupby(["episode", "comment_id"])["target_islander"].transform("size")
    eligible_mentions["other_islanders_mentioned"] = mentions_per_comment - 1
    eligible_mentions["has_comention"] = eligible_mentions["other_islanders_mentioned"].gt(0)

    comention_features = eligible_mentions.groupby(["target_islander", "episode"], as_index=False).agg(
        comention_denominator=("comment_id", "size"),
        asof_comention_strength=("other_islanders_mentioned", "sum"),
        asof_comention_comment_share=("has_comention", "mean"),
    )
    comention_features["asof_comention_strength_per_comment"] = (
        comention_features["asof_comention_strength"] / comention_features["comention_denominator"]
    )
    episode_features = episode_features.merge(comention_features, on=["target_islander", "episode"], validate="one_to_one")
    assert episode_features["asof_num_comments"].eq(episode_features["comention_denominator"]).all()
    episode_features = episode_features.drop(columns="comention_denominator")
    assert episode_features["asof_comention_comment_share"].between(0, 1).all()
    assert episode_features["asof_comention_strength_per_comment"].ge(0).all()

    # Rank discussion connections within each episode, using only eligible comments.
    # A self-join finds everyone mentioned alongside each target in a comment.
    connection_mentions = eligible_mentions[mention_columns].copy()
    connection_mentions = connection_mentions.rename(columns={"target_islander": "connection_islander"})
    paired_mentions = eligible_mentions.merge(connection_mentions, on=["episode", "comment_id"])
    different_islanders = paired_mentions["target_islander"].ne(paired_mentions["connection_islander"])
    paired_mentions = paired_mentions.loc[different_islanders].copy()
    connection_groups = paired_mentions.groupby(["target_islander", "episode"])
    connection_rows = []

    for group_key, target_mentions in eligible_mentions.groupby(["target_islander", "episode"]):
        target_islander, episode = group_key
        total_comments = target_mentions["comment_id"].nunique()
        ranked_connections = []

        if group_key in connection_groups.groups:
            target_connections = connection_groups.get_group(group_key)
            for connection_name, shared_mentions in target_connections.groupby("connection_islander"):
                shared_comment_ids = set(shared_mentions["comment_id"])
                ranked_connections.append((len(shared_comment_ids), connection_name, shared_comment_ids))

        # Largest count first; alphabetical names make ties reproducible.
        ranked_connections.sort(key=lambda connection: (-connection[0], connection[1]))
        first_name = None
        second_name = None
        first_comments = set()
        second_comments = set()
        first_place_ties = 0

        if len(ranked_connections) >= 1:
            first_count, first_name, first_comments = ranked_connections[0]
            for connection_count, connection_name, shared_comments in ranked_connections:
                if connection_count == first_count:
                    first_place_ties += 1

        if len(ranked_connections) >= 2:
            second_count, second_name, second_comments = ranked_connections[1]

        # Union prevents double-counting a comment that mentions both connections.
        top_two_comments = first_comments.union(second_comments)
        first_share = len(first_comments) / total_comments
        second_share = len(second_comments) / total_comments
        connection_rows.append({
            "target_islander": target_islander,
            "episode": episode,
            "asof_top1_connection": first_name,
            "asof_top2_connection": second_name,
            "asof_top1_comention_count": len(first_comments),
            "asof_top1_comention_share": first_share,
            "asof_second_comention_share": second_share,
            "asof_top2_comention_share": len(top_two_comments) / total_comments,
            "asof_connection_dominance_gap": first_share - second_share,
            "asof_top1_tie_count": first_place_ties,
        })

    connection_features = pd.DataFrame(connection_rows)
    episode_features = episode_features.merge(connection_features, on=["target_islander", "episode"], validate="one_to_one")
    assert episode_features["asof_top1_comention_share"].between(0, 1).all()
    assert episode_features["asof_top2_comention_share"].between(0, 1).all()
    assert episode_features["asof_top2_comention_share"].ge(episode_features["asof_top1_comention_share"]).all()
    assert episode_features["asof_connection_dominance_gap"].ge(0).all()

    # 5. Make one row for every episode each islander was observed.
    interval_rows = []

    for islander in subjects.itertuples(index=False):
        entrance_episode = int(islander.entrance_episode)
        exit_episode = int(islander.exit_episode)

        for episode in range(entrance_episode, exit_episode + 1):
            is_entry_episode = episode == entrance_episode
            is_last_episode = episode == exit_episode
            observed_exit = bool(islander.event_observed)
            time_since_entry = episode - entrance_episode

            interval_row = {
                "target_islander": islander.target_islander,
                "target_episode": episode,
                "feature_episode": episode - 1,
                "start": time_since_entry,
                "stop": time_since_entry + 1,
                "calendar_start": episode - 1,
                "calendar_stop": episode,
                "event": int(observed_exit and is_last_episode),
                "entry_interval": is_entry_episode,
                "censored_at_stop": int(not observed_exit and is_last_episode),
            }
            interval_rows.append(interval_row)

    intervals = pd.DataFrame(interval_rows)
    intervals = intervals.merge(
        subjects, on="target_islander", validate="many_to_one"
    )
    intervals = intervals.merge(
        cutoffs, on="target_episode", how="left", validate="many_to_one"
    )

    # 6. Attach the previous episode's features to each outcome interval.
    episode_features = episode_features.rename(
        columns={"episode": "feature_episode"}
    )
    intervals = intervals.merge(
        episode_features,
        on=["target_islander", "feature_episode"],
        how="left",
        validate="one_to_one",
    )

    asof_columns = []
    for column in intervals.columns:
        if column.startswith("asof_"):
            asof_columns.append(column)

    # There is no prior in-study episode for a newly entering islander.
    entry_rows = intervals["entry_interval"]
    intervals.loc[entry_rows, asof_columns] = np.nan
    intervals["asof_has_comments"] = intervals["asof_num_comments"].notna()
    intervals["asof_num_comments"] = (
        intervals["asof_num_comments"].fillna(0).astype(int)
    )
    intervals["asof_log_comments"] = np.log1p(intervals["asof_num_comments"])
    intervals["prediction_eligible"] = ~intervals["entry_interval"]
    intervals["outcome_next_episode"] = intervals["event"]

    # 7. Attach exploratory dashboard features. Their timing is unverified.
    dashboard_columns = {
        "episode_relative_metrics": list(
            dashboard_tables["episode_relative_metrics"].columns
        ),
        "episode_network_nodes": [
            "episode", "target_islander", "num_comments", "degree_centrality",
            "weighted_degree", "betweenness_centrality",
            "eigenvector_centrality", "community", "top_emojis",
            "standout_emojis",
        ],
        "episode_community_approval": [
            "episode", "target_islander", "approval", "approval_label",
            "sampled_comments", "confidence",
        ],
        "kalshi_episode_prices": ["episode", "target_islander", "kalshi_price"],
    }
    dashboard_prefixes = {
        "episode_relative_metrics": "explore_sentiment_",
        "episode_network_nodes": "explore_network_",
        "episode_community_approval": "explore_llm_",
        "kalshi_episode_prices": "explore_market_",
    }
    join_columns = ["target_islander", "feature_episode"]

    for table_name, selected_columns in dashboard_columns.items():
        dashboard_features = dashboard_tables[table_name][selected_columns].copy()
        dashboard_features = dashboard_features.rename(
            columns={"episode": "feature_episode"}
        )

        renamed_columns = {}
        for column in dashboard_features.columns:
            if column not in join_columns:
                renamed_columns[column] = dashboard_prefixes[table_name] + column

        dashboard_features = dashboard_features.rename(columns=renamed_columns)
        intervals = intervals.merge(
            dashboard_features,
            on=join_columns,
            how="left",
            validate="one_to_one",
        )

    # 8. Compare sentiment with other recorded at-risk islanders that episode.
    for sentiment_model in ["roberta", "vader"]:
        sentiment_column = f"asof_{sentiment_model}_mean"
        episode_groups = intervals.groupby("target_episode")[sentiment_column]
        episode_mean = episode_groups.transform("mean")
        episode_sd = episode_groups.transform("std", ddof=0)

        # A z-score is undefined when there is no meaningful variation.
        episode_sd = episode_sd.where(episode_sd > 1e-12)
        intervals[f"asof_{sentiment_model}_z"] = (
            intervals[sentiment_column] - episode_mean
        ) / episode_sd

    # 9. Calculate each islander's change from the preceding interval.
    intervals = intervals.sort_values(["target_islander", "target_episode"])
    intervals = intervals.reset_index(drop=True)

    for metric in ["roberta_mean", "vader_mean", "toxicity_mean"]:
        feature_column = f"asof_{metric}"
        intervals[f"{feature_column}_change"] = intervals.groupby(
            "target_islander"
        )[feature_column].diff()

    # 10. Prepare one early snapshot per islander for a possible static Cox fit.
    # This snapshot exists only for islanders who survived their entry interval.
    baseline_columns = ["target_islander", "target_episode", "cutoff_proxy"]
    for column in intervals.columns:
        if column.startswith("asof_"):
            baseline_columns.append(column)

    first_post_entry = intervals["start"] == 1
    baseline_features = intervals.loc[first_post_entry, baseline_columns].copy()
    baseline_features = baseline_features.rename(columns={
        "target_episode": "baseline_target_episode",
        "cutoff_proxy": "baseline_cutoff",
    })
    baseline = subjects.merge(
        baseline_features,
        on="target_islander",
        how="left",
        validate="one_to_one",
    )
    baseline["baseline_entry_time"] = 1
    baseline["baseline_available"] = baseline["baseline_target_episode"].notna()

    # 11. Check that the dataset matches the original season assumptions.
    assert not intervals.duplicated(["target_islander", "target_episode"]).any()
    assert intervals["event"].sum() == subjects["event_observed"].sum() == 26
    assert intervals["censored_at_stop"].sum() == 8
    assert (intervals["stop"] > intervals["start"]).all()

    exits_per_islander = intervals.groupby("target_islander")["event"].sum()
    assert (exits_per_islander <= 1).all()

    last_observed_times = intervals.groupby("target_islander")["stop"].max()
    expected_durations = subjects.set_index("target_islander")["duration_episodes"]
    assert last_observed_times.sort_index().equals(expected_durations.sort_index())

    has_timestamp = intervals["asof_latest_timestamp"].notna()
    latest_timestamps = intervals.loc[has_timestamp, "asof_latest_timestamp"]
    prediction_cutoffs = intervals.loc[has_timestamp, "cutoff_proxy"]
    assert (latest_timestamps < prediction_cutoffs).all()

    # 12. Describe each column's role and missing values.
    outcome_columns = {
        "exit_episode", "exit_day", "final_placement", "duration_episodes",
        "event_observed", "censored", "event", "outcome_next_episode",
        "censored_at_stop",
    }
    dictionary_rows = []

    for column in intervals.columns:
        if column in outcome_columns:
            role = "outcome_only_never_predictor"
        elif column.startswith("asof_"):
            role = "candidate_cutoff_filtered"
        elif column.startswith("explore_"):
            role = "exploration_only_timing_or_future_context_unverified"
        else:
            role = "identifier_time_or_baseline_metadata"

        dictionary_row = {
            "column": column,
            "role": role,
            "missing_rows": int(intervals[column].isna().sum()),
            "dtype": str(intervals[column].dtype),
        }
        dictionary_rows.append(dictionary_row)

    feature_dictionary = pd.DataFrame(dictionary_rows)
    tables = {
        "survival_subjects": subjects,
        "survival_intervals": intervals,
        "survival_baseline_covariates": baseline,
        "survival_feature_dictionary": feature_dictionary,
        "survival_feature_timing_audit": timing_audit,
    }

    # 13. Save all tables. Roll back database changes if an export fails.
    output_directory = project_root / "tables/survival_dashboard"
    output_directory.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN")
        try:
            for table_name, dataset in tables.items():
                connection.register("survival_export", dataset)
                connection.execute(
                    f"CREATE OR REPLACE TABLE dashboard.{table_name} "
                    "AS SELECT * FROM survival_export"
                )
                saved_row_count = connection.execute(
                    f"SELECT count(*) FROM dashboard.{table_name}"
                ).fetchone()[0]
                assert saved_row_count == len(dataset)
                connection.unregister("survival_export")

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    for table_name, dataset in tables.items():
        output_path = output_directory / f"{table_name}.csv"
        dataset.to_csv(output_path, index=False)

    print(
        f"{len(subjects)} subjects; "
        f"{len(intervals)} intervals; "
        f"{intervals['event'].sum()} final exits; "
        f"{intervals['censored_at_stop'].sum()} finale censorings."
    )
    print(
        f"{intervals['prediction_eligible'].sum()} post-entry rows available "
        "for next-episode exploration. No model fitted."
    )
    return tables
