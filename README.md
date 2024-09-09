Flexible LSTM code for the processing of numerical data (FOREX).
The code is set up so that changing variables or adding complexities is easier to accomplish.

Several functions used to prepare series data as a dataframe while avoiding duplication and unecessary conversions.
  1. classify - classifies future price price based on current price into 1's and 0's.
  2. PrePro - preprocessing dataframe into series style window for all chosen variables.
  3. Shuffle_Split - ensures equal number of examples for each classification (useful for unbalanced datasets).

The model itself is set up minimalistical as to be adapted for larger projects and to allow for the simple swap of lstm with any 
other deep learning model.
TensorBoard is used to track the models based on val_loss function.
