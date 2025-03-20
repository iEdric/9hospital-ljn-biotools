Table 4: Baseline model performance on each of the three scoring metrics (*task completion, task process, explanatory knowledge discovery*) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| #  | Topic          | Task                                  | ReACT | Plan+Execute | Hypothesizer |
|----|----------------|---------------------------------------|-------|--------------|--------------|
| 1  | Proteomics     | Clustering (2D)                       | 0.87  | 0.20         | 0.00         |
| 2  |                | Simplified Clustering                 | 0.88  | 0.40         | 0.00         |
| 3  |                | Clustering (3D)                       | 0.88  | 0.40         | 0.60         |
| 4  | Chemistry      | Single substances                     | 0.87  | 1.00         | 0.00         |
| 5  |                | Mix of 3 substances                   | 0.82  | 0.00         | 0.00         |
| 6  | Archaeology    | Correlations                          | 0.87  | 0.00         | 0.00         |
| 7  |                | Simple instrument                     | 0.87  | 0.60         | 0.00         |
| 8  | Reactor Lab    | Linear regression                     | 0.42  | 0.00         | 0.00         |
| 9  |                | Quadratic regression                  | 0.43  | 0.00         | 0.20         |
| 10 | Plant Nutrients| Uncovering systems of rules           | 0.51  | 0.00         | 0.00         |
| 11 | Space Sick     | Presence rules                        | 0.91  | 0.60         | 0.00         |
| 12 |                | Legal rules                           | 0.00  | 0.00         | 0.00         |
| 13 | Rocket Science | Open-ended discovery                  | 0.78  | 0.60         | 0.00         |
| 14 | Translation    | Rosetta-stone style linguistic discovery| 0.30 | 0.40         | 0.00         |
| 15 |                | Noun and verb                         | 0.49  | 0.00         | 0.00         |
| 16 |                | Noun, adj., and verb                  | 0.49  | 0.00         | 0.00         |
|    | **Average (Easy)** |                                   | 0.59  | 0.23         | 0.05         |
|    | **Average (Normal)** |                                 | 0.09  | 0.06         | 0.14         |
|    | **Average (Challenge)** |                              | 0.63  | 0.18         | 0.10         |

Table 5: Baseline model performance on each of the three scoring metrics (*task completion, task process, explanatory knowledge discovery*) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| #  | Unit Test Topic | ReACT | Plan+Execute | Hypothesizer |
|----|-----------------|-------|--------------|--------------|
| 25 | Multi-turn dialog with an agent | 1.00  | 1.00         | 1.00         |
| 26 | Measure an object with an instrument | 0.87  | 0.60         | 0.73         |
| 27 | Pick-and-place object | 0.90  | 0.80         | 0.80         |
| 28 | Pick-and-place object | 1.00  | 1.00         | 1.00         |
| 29 | Read DiscoveryFeed posts | 1.00  | 1.00         | 1.00         |
| 30 | Move through doors | 0.55  | 0.20         | 0.25         |
| 31 | Using keys with doors | 0.60  | 0.20         | 0.25         |
| 32 | Navigate to a specific room in a house | 0.20  | 0.00         | 0.00         |
| 33 | Search an environment for an object | 0.80  | 0.20         | 0.00         |
| 34 | Interact with a moving agent | 0.80  | 0.20         | 0.53         |
|    | **Average (Unit Tests)** | 0.76  | 0.60         | 0.66         |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-4 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide...