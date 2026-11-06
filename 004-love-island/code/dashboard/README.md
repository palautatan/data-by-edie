# Local network dashboard

Open `viz/dashboard/index.html` in a web browser. It is self-contained and works offline.

Rebuild from saved DuckDB results using section 18 at the end of `notebooks/08-dashboard.ipynb`, or:

```sh
python code/build_network_dashboard.py
```

The builder reads saved results only; it does not fit models or change DuckDB.

- All overall islanders, positions and communities come from notebook 07's saved tables.
- Episode defaults match 07: entrance through final exit, inclusive. The departed toggle restores all saved discussion nodes for that episode.
- Edge density changes relationships, never the node population. Focus mode is an explicit exception that displays the selected islander and their neighbours.
- Logistic probabilities join selected episode to `feature_episode`, so episode k displays a forecast for k+1. Missing values remain unavailable.
- Cox survival is a full-season baseline-landmark descriptive estimate, not a live forecast. Its curve stops at observed follow-up. Kaplan–Meier is the whole-cast descriptive curve.
- Overall mode does not average episode probabilities. Curves and historical trends remain available.

Run the interaction/data checks with `node tests/test_network_dashboard.cjs` after building. These test the JavaScript logic with a minimal DOM/Plotly stub, not browser rendering.
