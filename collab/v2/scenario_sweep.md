# Multi-scenario alert sweep

- Association: `outputs_v2_assoc_th071/associations/associations_v2.csv`
- Global IDs: `outputs_v2_assoc_th071/global_ids/global_ids_v2.csv`

## Results

|   ticket_rate |   n_high_mean |   n_high_std |   high_precision_mean |   high_precision_std |   high_recall_mean |   high_recall_std |   mean_score_mean |   watch_precision_mean |   watch_recall_mean |
|--------------:|--------------:|-------------:|----------------------:|---------------------:|-------------------:|------------------:|------------------:|-----------------------:|--------------------:|
|          0.30 |        408.00 |        16.70 |                  1.00 |                 0.00 |               1.00 |              0.00 |              0.82 |                   1.00 |                1.00 |
|          0.60 |        229.00 |        10.82 |                  1.00 |                 0.00 |               1.00 |              0.00 |              0.67 |                   1.00 |                1.00 |
|          0.80 |        116.67 |         7.57 |                  1.00 |                 0.00 |               1.00 |              0.00 |              0.58 |                   1.00 |                1.00 |

Conclusions: lower ticket_rate increases alert burden and usually reduces precision of high alerts.
The current 0.60 rate is the middle case.
