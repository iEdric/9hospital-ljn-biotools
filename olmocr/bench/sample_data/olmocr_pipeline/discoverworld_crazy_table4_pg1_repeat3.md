Table 4: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| #  | Topic          | Task                     | ReACT Completion | Knowledge | Plan+Execute Completion | Knowledge | Hypothizer Completion | Knowledge |
|----|----------------|--------------------------|------------------|-----------|------------------------|-----------|----------------------|-----------|
| 1  | Proteomics     | Simplified Clustering    | 0.87             | 0.20      | 0.80                   | 0.00      | 0.00                 | 0.90      |
| 2  | Chemistry      | Exploring Combinations   | 0.88             | 0.40      | 0.88                   | 0.40      | 0.60                 | 0.93      |
| 3  | Chemistry      | Hill Climbing            | 0.88             | 0.40      | 0.88                   | 0.40      | 0.60                 | 0.93      |
| 4  | Archaeology    | Mix of 3 substances      | 0.82             | 0.00      | 0.87                   | 0.40      | 0.00                 | 0.93      |
| 5  | Archaeology    | Mix of 4 substances      | 0.88             | 0.00      | 0.88                   | 0.40      | 0.00                 | 0.93      |
| 6  | Archaeology    | Simple instrument        | 0.27             | 0.60      | 0.33                   | 0.20      | 0.00                 | 0.60      |
| 7  | Archaeology    | Instrument Use           | 0.72             | 0.40      | 0.74                   | 0.00      | 0.00                 | 0.64      |

Table 5: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| #  | Unit Test Topic                              | ReACT Completion | Knowledge | Plan+Execute Completion | Knowledge | Hypothizer Completion | Knowledge |
|----|---------------------------------------------|------------------|-----------|------------------------|-----------|----------------------|-----------|
| 25 | Multi-turn dialog with an agent              | 1.00             | 1.00      | 1.00                   | 1.00      | 1.00                 | 1.00      |
| 26 | Measure an object with an instrument        | 0.87             | 0.60      | 0.73                   | 0.40      | 1.00                 | 1.00      |
| 27 | Pick-and-place object                       | 0.90             | 0.80      | 0.80                   | 0.60      | 1.00                 | 1.00      |
| 28 | Search an environment for an object         | 0.80             | 0.80      | 0.80                   | 0.60      | 1.00                 | 1.00      |
| 29 | Read Discovery Feed posts                  | 1.00             | 1.00      | 0.90                   | 0.80      | 1.00                 | 1.00      |
| 30 | Move through doors                          | 0.58             | 0.20      | 0.25                   | 0.00      | 0.30                 | 0.00      |
| 31 | Using keys with doors                       | 0.69             | 0.20      | 0.54                   | 0.00      | 0.69                 | 0.00      |
| 32 | Navigate to a specific room in a house      | 0.20             | 0.20      | 0.20                   | 0.00      | 0.20                 | 0.20      |
| 33 | Interact with a moving agent                | 0.60             | 0.20      | 0.53                   | 0.00      | 0.53                 | 0.20      |
| 34 | Rosetta-stone style linguistic discovery of alien language | 0.59 | 0.38 | 0.25 | 0.56 | 0.18 | 0.11 | 0.56 | 0.28 | 0.34 |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-40 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide