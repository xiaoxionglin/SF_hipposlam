# DG Anti-Collapse Batch Results

Batch: `intrmotiv_dg_anti_collapse_20260824`.
All 120 one-policy jobs completed at about 80M environment frames. Terminal
metrics use the last 10M frames. The TensorBoard evaluation artifacts live at:

```text
/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/dg_anti_collapse_20260825
```

## Conclusion

The global pre-threshold punishment does **not** solve DG activity collapse.
It has no consistent paired benefit over coefficient zero and commonly makes
the silent-unit fraction and exploration worse. The row-repulsion arm also
does not establish a solution: its terminal DG activity remains sparse and it
does not show a controlled stability advantage.

The batch does not support a claim that the methods solve receptive-field
drift. Checkpoint DG-row drift indicates that stronger global punishment can
increase total parameter rotation, while most conditions already change little
over their final approximately 5M frames. Parameter stability is not a
place-field stability measurement.

## Collapse Evidence

The best terminal activity in the main factorial was the flat,
`threshold=1.8`, `mean`, coefficient-zero control:

| Condition | DG density | Silent-unit fraction | Coverage AUC |
| --- | ---: | ---: | ---: |
| Flat, T=1.8, mean, global coeff 0 | 0.00106 | 0.827 | 62.7 |
| Flat, T=1.8, mean, global coeff 0.01 | 0.00117 | 0.855 | 63.4 |
| Flat, T=1.8, mean, global coeff 0.03 | 0.00144 | 0.884 | 48.1 |

Thus lowering the threshold helps relative to high-threshold cells, but still
leaves roughly 13 of 16 DG units silent in the terminal learner windows.

Across the replicated global-penalty factorial, only 3 of 24
architecture x threshold x feedback x coefficient comparisons had a lower
mean silent-unit fraction than their same-seed coefficient-zero controls. This
is not a credible correction of collapse. The clearest failure is `punish`:
the penalty usually reduces density and coverage further.

High coverage is not evidence of healthy DG activity. For example, fixed/global
HRL with T=2.0, `punish`, and coefficient zero reached coverage AUC 74.9 but
had DG density 0.000130 and silent fraction 0.932. The global-HRL
row-repulsion `punish` condition reached coverage AUC 75.8 with silent fraction
0.954. These are exploration outcomes with substantially collapsed DG events.

## Drift Evidence

The ResNet layer-2 trunk is ImageNet-pretrained and fixed. Drift below is the
corresponding cosine similarity of DG-projection rows from the first retained
checkpoint (about 3-4M frames) to the final checkpoint (about 80M frames).
Late-to-final compares the checkpoint nearest 75M with the final checkpoint.

| Replicated condition | Initial-to-final cosine | Mean angle | Late-to-final cosine |
| --- | ---: | ---: | ---: |
| Global HRL, T=1.8, mean, coeff 0 | 0.962 | 10.9 deg | 0.99994 |
| Global HRL, T=1.8, mean, coeff 0.01 | 0.878 | 25.4 deg | 0.99893 |
| Flat, T=1.8, mean, coeff 0 | 0.928 | 16.9 deg | 0.99485 |
| Flat, T=1.8, mean, coeff 0.03 | 0.884 | 24.9 deg | 0.99808 |
| Global HRL, T=2.2, punish, coeff 0.03 | 0.701 | 43.9 deg | 0.99653 |

The penalty therefore does not stabilize DG rows; in these representative
matched cases, stronger punishment increases total rotation. Most rows are
nearly stationary near the end of training regardless of arm, so late drift
was not the dominant problem to begin with.

## Limits and Next Decision

The online activity metrics establish collapse. Weight cosine does not establish
whether DG spatial fields are stable or useful. A field-level conclusion needs
fixed-trajectory DMLab place-field evaluation at several checkpoints for a
small set of matched coefficient-zero and penalty conditions.

Do not continue the global-pre-threshold punishment sweep as the primary
anti-collapse mechanism. Retain low-threshold `mean` as the activity baseline,
and redesign the unexplored-location objective so it creates or preserves DG
events instead of directly suppressing every DG logit.
