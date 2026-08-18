# Routernet OpenML Benchmark Summary

> `RouterNet_vs_Uniform` is the routing ablation: > 0 means the router adds value over the exact experts it routes; ~0 means routing is unnecessary; < 0 means routing is actively hurting.

| dataset             |   BestExpert |   Uniform |   RouterNet |   RouterNet_vs_Uniform |   RandomForest |   XGBoost |
|:--------------------|-------------:|----------:|------------:|-----------------------:|---------------:|----------:|
| balance-scale       |       1      |    0.9206 |      0.9524 |                 0.0317 |         0.8889 |    0.9365 |
| breast-w            |       0.9857 |    0.9571 |      0.9571 |                 0      |         0.9714 |    0.9714 |
| cmc                 |       0.5946 |    0.5473 |      0.5676 |                 0.0203 |         0.5743 |    0.5743 |
| kr-vs-kp            |       0.9938 |    0.9906 |      0.9969 |                 0.0062 |         0.9875 |    0.9938 |
| letter              |       0.9725 |    0.9745 |      0.974  |                -0.0005 |         0.8685 |    0.955  |
| mfeat-factors       |       0.975  |    0.98   |      0.98   |                 0      |         0.96   |    0.96   |
| mfeat-fourier       |       0.85   |    0.83   |      0.825  |                -0.005  |         0.825  |    0.85   |
| mfeat-karhunen      |       0.98   |    0.985  |      0.98   |                -0.005  |         0.97   |    0.955  |
| mfeat-morphological |       0.7    |    0.665  |      0.665  |                 0      |         0.67   |    0.67   |
| mfeat-zernike       |       0.81   |    0.78   |      0.795  |                 0.015  |         0.74   |    0.795  |
