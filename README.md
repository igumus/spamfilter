spamfilter
==========

Naive Bayes classifier for identifying spam

First, create the features file:
spamfilter.py -getfeatures

Then, create a boolean or ntf model:
spamfilter.py -makemodel [boolean or ntf]

Then, test the boolean or ntf model:
spamfilter.py -testmodel [boolean or ntf] [cost_ratio]

Then, filter a stripped email file with the boolean or ntf model and a cost ratio:
spamfilter.py -filteremail [boolean or ntf] [filename] [cost_ratio]
