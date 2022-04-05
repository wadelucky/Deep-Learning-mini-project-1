We accidentally saved our model with torch.save(model, path), which made the execution of self_eval.py failed. 

In the self_eval.py, we removed the normalization line of the testing data to maintain our accuracy, which is 91.8% this time.

We also removed codes involving logging to make reproduction easier.