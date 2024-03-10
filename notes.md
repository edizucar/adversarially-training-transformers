# Notes

## 10th March
 - What should be the output of the linear probe?
  - It depends on the type of feature we care about.
  - If we care about 'previous token' we can represent this as either an embedding vector or a one hot encoding of the tokens.
  - These both pose issues. How can we compute a loss on these that usefully represents the idea of 'information is not contained within this layer of the residual stream'
  - Instead use binary feature e.g. 'is capital letter'
  - Still not 100% what the loss will be for this?
  - But its easier cheaper to train (less datapoints needed)

- What does a training loop look like?
 - If we train a LP from scratch for each outer loop, its too expensive
 - Only need a few LP iterations for each transformer iterations. Transformer doesn't move much in that time anyway so its fine.