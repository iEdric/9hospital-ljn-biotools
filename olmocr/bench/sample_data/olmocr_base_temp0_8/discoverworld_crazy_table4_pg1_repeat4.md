Table 4: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| # | Topic | Task | Task Completion | Knowledge | Procedure | Task Completion | Knowledge | Procedure | Task Completion | Knowledge | Procedure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Proteomics | Clustering | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 | 0.90 | 0.40 | 1.00 |
| 2 | Chemistry | Exploring Combinations and Hill Climbing | 0.88 | 0.40 | 0.88 | 0.40 | 0.88 | 0.40 | 0.91 | 0.40 | 0.60 |
| 4 | Archaeology | Correlations | 0.87 | 1.00 | 0.87 | 1.00 | 0.87 | 1.00 | 0.87 | 1.00 | 0.87 |
| 7 | Reactor Lab | Regression | 0.27 | 0.60 | 0.27 | 0.60 | 0.27 | 0.60 | 0.60 | 0.20 | 0.50 |
| 9 | Space Sick | Single instrument | 0.72 | 0.40 | 0.72 | 0.40 | 0.72 | 0.40 | 0.64 | 0.40 | 0.40 |
| 10 | Plant Nutrients | Simplified rules | 0.46 | 0.20 | 0.46 | 0.20 | 0.46 | 0.20 | 0.55 | 0.20 | 0.05 |
| 13 | Normal | Multiple instruments | 0.73 | 0.40 | 0.73 | 0.40 | 0.73 | 0.40 | 0.66 | 0.20 | 0.50 |
| 15 | Challenge | Novel instruments | 0.21 | 0.40 | 0.21 | 0.40 | 0.21 | 0.40 | 0.56 | 0.40 | 0.40 |
| 17 | Chemistry | Clustering (2D) | 0.31 | 0.40 | 0.31 | 0.40 | 0.31 | 0.40 | 0.52 | 0.40 | 0.40 |
| 19 | Space Sick | Open-ended discovery | 0.88 | 0.40 | 0.88 | 0.40 | 0.88 | 0.40 | 0.88 | 0.40 | 0.88 |
| 21 | Normal | Mix of 3 substances | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 |
| 23 | Challenge | Mix of 4 substances | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 |
| 26 | Plant Nutrients | Quadratic regression | 0.39 | 0.40 | 0.39 | 0.40 | 0.39 | 0.40 | 0.39 | 0.40 | 0.39 |

Table 5: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| # | Unit Test Topic | Task Completion | Knowledge | Procedure | Task Completion | Knowledge | Procedure |
|---|---|---|---|---|---|---|---|
| 25 | Multi-turn dialog with an agent | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 26 | Measure an object with an instrument | 0.88 | 0.60 | 0.88 | 0.60 | 0.88 | 0.60 |
| 27 | Pick-and-place object | 0.55 | 0.40 | 0.55 | 0.40 | 0.55 | 0.40 |
| 28 | Read DiscoveryFeed posts | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 30 | Move through doors | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 |
| 31 | Using keys with doors | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| 32 | Navigate to a specific room in a house | 0.80 | 0.20 | 0.80 | 0.20 | 0.80 | 0.20 |
| 33 | Interact with a moving agent | 0.60 | 0.20 | 0.60 | 0.20 | 0.60 | 0.20 |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-40 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide