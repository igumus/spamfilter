spamfilter
==========

Naive Bayes classifier for identifying spam

First, create the features file:
spamfilter.py -getfeatures

Then, create a boolean or ntf model:
Usage: %s -makemodel [boolean or ntf]

Then, test the boolean or ntf model:
Usage: %s -testmodel [boolean or ntf] [cost_ratio]

Then, filter a stripped email file with the boolean or ntf model and a cost ratio:
Usage: %s -filteremail [boolean or ntf] [filename] [cost_ratio]
