# Classification eval — 19 hand-labeled reviews

Model: openai/gpt-oss-120b
Overall accuracy: 84.21%
Unclassified (model returned no answer): 1

```
                 precision    recall  f1-score   support

        billing       0.00      0.00      0.00         1
          crash       1.00      0.80      0.89         5
feature_request       1.00      0.75      0.86         4
         praise       1.00      1.00      1.00         5
   unclassified       0.00      0.00      0.00         0
             ux       0.67      1.00      0.80         4

       accuracy                           0.84        19
      macro avg       0.61      0.59      0.59        19
   weighted avg       0.88      0.84      0.85        19

```

```
Confusion matrix (rows=true, cols=predicted):
labels: billing, crash, feature_request, praise, unclassified, ux
         billing: [0, 0, 0, 0, 0, 1]
           crash: [0, 4, 0, 0, 1, 0]
 feature_request: [0, 0, 3, 0, 0, 1]
          praise: [0, 0, 0, 5, 0, 0]
    unclassified: [0, 0, 0, 0, 0, 0]
              ux: [0, 0, 0, 0, 0, 4]
```

## Misclassifications
- true=`crash` predicted=`unclassified`: "cant íntall on my iphone"
- true=`feature_request` predicted=`ux`: "App is nice but when is ee any status of my friends it doesn't have sound I have to increase sound e"
- true=`billing` predicted=`ux`: "ads"