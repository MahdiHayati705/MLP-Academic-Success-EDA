# MLP-Academic-Success-EDA
Explores UCI student dropout and success data for an MLP model, tests how each hyperparameter changes results, picks the best model, and checks the confusion matrix, neuron heatmaps, and what happens when one of two similar neurons is removed.

A clear, beginner-friendly project using the UCI **Predict Students' Dropout and Academic Success** dataset (ID: `697`, `4,424` rows, `37` columns)[1] to explore the data, test **MLP (Multi-Layer Perceptron)** hyperparameters, pick the best model, inspect how the network behaves, and test how changing one input feature affects a prediction.

## Main Goal

The purpose of this project is to:
1. Explore and prepare the student dataset for an MLP model.
2. Check the effect of changing each **MLP hyperparameter**.
3. Choose the **best MLP model**.
4. Inspect the best model using a **confusion matrix**, a **neuron heatmap**, and an experiment that **removes one of two similar neurons** in a layer.
5. Take **one sample**, keep all its inputs fixed except one, and **sweep that single feature from min to max** to see how it changes the prediction.

## Hyperparameters Tested

This project tests the effect of changing each of these settings:
- `HIDDEN_LAYERS`: Number of hidden layers in the network.
- `HIDDEN_UNITS`: Number of neurons in hidden layers.
- `ACTIVATION`: Activation function used between layers.
- `OPTIMIZER`: Algorithm used to update weights during training.
- `LEARNING_RATE`: Step size used when updating weights.
- `BATCH_SIZE`: Number of samples processed before each weight update.
- `DROPOUT_RATE`: Fraction of neurons randomly dropped during training to reduce overfitting.
- `BATCH_NORMALIZATION`: Whether batch normalization is applied to stabilize training.
- `EARLY_STOPPING`: Whether training stops early when validation performance stops improving.
- `PATIENCE`: How many epochs to wait with no improvement before stopping early.
- `WEIGHT_INITIALIZATION`: How starting weights are set before training.
- `LOSS_FUNCTION`: Objective used to measure prediction error.
- `EPOCHS`: Maximum number of full passes over the training data.

## What This Project Covers

1. **Check Tools & Load Data:** Checks Python and deep learning library versions, loads UCI dataset `697` (`4,424` rows and `37` columns), cleans column names, and groups columns into numbers, ordered categories, and unordered categories.
2. **Explore & Clean Data:** Looks at basic number stats (min, max, average, middle, spread), draws charts for columns like `Age at enrollment`, `Admission grade`, and `Previous qualification (grade)`, and checks for missing values or duplicate rows in the features.
3. **Test Hyperparameters:** Changes each hyperparameter above (`HIDDEN_LAYERS`, `HIDDEN_UNITS`, `ACTIVATION`, `OPTIMIZER`, `LEARNING_RATE`, `BATCH_SIZE`, `DROPOUT_RATE`, `BATCH_NORMALIZATION`, `EARLY_STOPPING`, `PATIENCE`, `WEIGHT_INITIALIZATION`, `LOSS_FUNCTION`, and `EPOCHS`) to see how it affects results.
4. **Pick the Best Model:** Compares the tested configurations and selects the best-performing MLP model.
5. **Confusion Matrix:** Shows where the best model gets predictions right and where it mixes up student outcomes.
6. **Neuron Heatmap:** Visualizes neuron weights or activations to make patterns inside a layer easier to see.
7. **Similar Neuron Removal Test:** Finds two neurons in a layer that behave similarly, removes one of them, and checks what changes.
8. **Single-Feature Min-to-Max Sweep:** Takes one sample, keeps every input fixed except one chosen feature, and sweeps that feature from its minimum to maximum value to see how the model's prediction changes.

## What You Need

- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `tensorflow`
- `ucimlrepo` (`pip install ucimlrepo`)

## How to Run

Open the notebook in Jupyter Notebook or Google Colab and run the cells from top to bottom.
