## Core Approach

The core idea in our approach for Training Dynamics research is the existence of **Latent States** i.e. Hidden states which the model navigates through during the training process and studying these latent states can help us draw insights about the training dynamics a model is following for choosing different generalization strategies.

We believe that parameters a Neural Network is encountering during the training process encode information about these latent states, and calculating metrics on top of these parameters can facilitate that process for us. We collect a handpicked selection of 20 [metrics]() using these parameters using which we can do further Latent State analysis. User has the flexibility to specify hook points based on their requirements and these metrics will be calculated for those specific hookpoints and stored in json files which can be proccessed into preparing our data
for further analysis.

For predicting latent states from these data points, we are using
