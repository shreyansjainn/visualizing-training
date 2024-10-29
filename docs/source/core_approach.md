# Core Approach

The core idea in our approach for Training Dynamics research is the existence of **Latent States** i.e. Hidden states which the model navigates through during the training process and studying these latent states can help us draw insights about the training dynamics a model is following for choosing different generalization strategies.

We believe that parameters (weights and biases of the model) a Neural Network is encountering during the training process encode information about these latent states, and calculating metrics on top of these parameters can facilitate that process for us. We collect a handpicked selection of 20 [metrics](./metrics.md) from these parameters using which we can do further Latent State analysis. User has the flexibility to specify hook points based on their requirements and these metrics will be calculated for those specific hookpoints and stored, which can be proccessed into preparing our data for further analysis.

For predicting latent states using these data points, we are leveraging [Hidden Markov Models (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model) (read more [here](./resources.md#hidden-markov-models)) which helps us narrow down the training dynamics to our choice of number of latent states for studying different generalization strategies.

To assist with the analysis, a visualization module is provided to plot network graphs to study state transitions along with curve plotting for different metric combinations.
