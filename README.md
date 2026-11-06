# Medicine stockout prediction (prototype)

For a store-drug that is in stock this week, will it stock out within the time it takes to restock (supplier lead time + 1 week)?

**The data is synthetic.** The four CSVs in `data/` describe 16 stores, 19 drugs and 156 weeks, generated with a known supply process (orders arrive exactly on schedule). The scores below show how well the method recovers that process, not how real pharmacies behave.

## Approach

- **Target:** 1 if the store-drug has at least one stockout in weeks t+1 to t+H, where H = lead time + 1. Only rows that are in stock at week t are scored, so a model cannot do well by predicting "still out of stock". Rows whose window runs past the end of the data are dropped.
- **Features:** only what is known at the end of week t: stock cover against lead time, demand lags and rolling means, orders in transit, stockout history, seasonality, and the severity of the diseases a drug treats.
- **Split:** by time (60% train, 20% validation, 20% test) with a 4-week gap so label windows do not overlap.
- **Threshold:** chosen on validation (F-beta, beta = 2, so recall counts double) and applied once to test.
- **Comparison:** four one-line rules (stock cover, orders in transit, recent stockouts) against logistic regression, random forest and gradient boosting. Accuracy is not reported because stockouts are rare.

## Results

Test period: 25 weeks, 7,309 scored rows, 237 positives (base rate 3.2%). A random score gives a PR-AUC of 0.03.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| Rule: cover below lead time | 0.255 | 0.888 |
| Rule: pipeline cover below lead time | 0.219 | 0.810 |
| Rule: recent stockouts | 0.178 | 0.780 |
| Rule: low projected cover | 0.133 | 0.802 |
| Logistic regression | 0.518 | 0.934 |
| Random forest | 0.532 | 0.943 |
| Gradient boosting | 0.554 | 0.937 |

Gradient boosting has a 95% CI of 0.42 to 0.66 (resampling store-drug series). The three models fall within each other's uncertainty, so I do not rank them. All three are about twice as good as the best rule.

**By lead time.** The window grows with lead time, so the base rate does too.

| Lead time | Positives | Base rate | Model PR-AUC | Best rule PR-AUC |
|---|---|---|---|---|
| 1 week | 21 | 0.6% | 0.05 | 0.04 |
| 2 weeks | 91 | 3.4% | 0.50 | 0.16 |
| 3 weeks | 125 | 15.2% | 0.67 | 0.34 |

The model beats the rule at lead times of 2 and 3 weeks. At 1 week there are too few events, and the model is weak in absolute terms.

**Alert load.** At the validation threshold the model flagged 913 rows in 25 weeks, about 37 a week across 16 stores. It caught 198 of 237 events (recall 0.84, precision 0.22), roughly 3.6 false alarms per catch. Recall was 0.73 on validation, so expect somewhere between the two.

![Precision-recall, test period](docs/pr_curve_test.png)

## Limitations

- Synthetic data: results depend on how it was generated.
- One 25-week test window, under half a seasonal cycle. Seasonal generalisation is untested.
- Four drugs hold 76% of the test events. Per-drug and per-category recall for the rest rests on 1 to 8 events each.
- The features use true demand. Real dispensing records only show units dispensed, so demand is censored when stock runs out.
- Not validated on real stock records.

## Run it

Python 3.10 or later. Tested with pandas 3.0, scikit-learn 1.9, numpy 2.5, matplotlib 3.11 and seaborn 0.13.

```bash
pip install -r requirements.txt
jupyter notebook Medicine-stockout-prediction.ipynb
```

The notebook reads the CSVs from `data/`. Set `MEDSTOCK_DATA` to use another folder.

## Files

```
Medicine-stockout-prediction.ipynb   full pipeline, results and charts
data/                                dispensing, context, drugs, stores (synthetic)
docs/                                figures used in this README
requirements.txt
```
