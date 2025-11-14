# TODO: Fix Issues in Accuracy Check Scripts

- [x] Update debug_accuracy.py: Add zero_division to precision_score, class_weight to RandomForest, add CV for MCC and Kappa
- [x] Update comprehensive_accuracy_check.py: Fix scorer names (cohen_kappa_score -> cohen_kappa), add CV for MCC
- [x] Run updated debug_accuracy.py to verify fixes
- [x] Analyze new results and compare
- [x] Suppress UndefinedMetricWarning in debug_accuracy.py
- [x] Suppress UndefinedMetricWarning in comprehensive_accuracy_check.py
- [x] Add zero_division to precision_score in comprehensive_accuracy_check.py
- [x] Use make_scorer for CV metrics in comprehensive_accuracy_check.py
