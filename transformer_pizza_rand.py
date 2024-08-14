import argparse
import os

from visualize_training.train import ModelManager
from visualize_training.utils import training_run_json_to_csv, save_model
from visualize_training.hmm import HMM

from pizza.model import Transformer
from pizza.data import get_dataloaders, get_dataloaders_with_full_dataset


def main():
    parser = argparse.ArgumentParser(description="Collect training metrics for a Transformer model on the Modular dataset")
    parser.add_argument("--n_seeds", type=int, required=True, help="Number of seeds to run the training for")
    parser.add_argument("--run_output_dir", type=str, default="output", help="Directory to save the training metrics")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--use_ln", type=bool, default=False, help="Whether to use layer normalization in the model")
    parser.add_argument("--clip_grad", type=bool, default=False, help="Whether to clip the gradients")
    parser.add_argument("--init_scaling", type=float, default=1.0, help="Scaling factor for the initialization of the model weights")
    parser.add_argument("--cpu", action="store_false", help="Whether to run the training on CPU")
    parser.add_argument("--use_accelerator", action="store_false", help="Whether to use the accelerator for training")
    parser.add_argument("--eval_every", type=int, default=5, help="Frequency of evaluation during training")
    parser.add_argument("--clock_pizza_metrics", action="store_true", default=True, help="Whether to use the clock-pizza metrics")
    parser.add_argument("--attention_rate", type=float, default=1.0, help="Fraction of attention heads to use")
    parser.add_argument("--n_epochs", type=int, default=2000, help="Number of epochs to train the model for")
    parser.add_argument("--hook_layers", nargs="+", type=str, default=['blocks.0.mlp','blocks.0.attn'],help="Layers to attach the hooks to")
    parser.add_argument("--random_features", type=bool, default=True, help="Whether to use random features in the model")
    parser.add_argument("--projection_dim", type=int, default=10, help="Dimension of the random features projection")

    args = parser.parse_args()

    n_seeds = args.n_seeds
    lr = args.lr
    clock_pizza_metrics = args.clock_pizza_metrics
    run_output_dir = args.run_output_dir
    use_ln = args.use_ln
    clip_grad = args.clip_grad
    init_scaling = args.init_scaling
    cpu = args.cpu
    eval_every = args.eval_every
    use_accelerator = args.use_accelerator
    attention_rate = args.attention_rate
    num_epochs = args.n_epochs
    hook_layers = args.hook_layers
    random_features = args.random_features
    projection_dim = args.projection_dim

    # Define the configuration for the training run and the model
    C=59
    n_layers=1
    frac_coeff=attention_rate
    diff_vocab=0
    eqn_sign=0
    d_model=128
    n_ctx=2

    optimizer = "adamw"
    is_transformer = True
    dataset_name = "modular"

    config=dict(
            name='modadd_'+str(C),
            funcs='lambda x: (x[0]+x[1])%'+str(C),
            C=C,
            n_heads=4,
            d_model=d_model,
            n_layers=n_layers,
            n_ctx=n_ctx,
            attention_dir='casual',
            act_fn='ReLU',
            num_epochs=num_epochs,
            batch_size=C*C,
            lr=lr,
            weight_decay=2.,
            frac=0.8,
            attn_coeff=frac_coeff,
            diff_vocab=diff_vocab,
            eqn_sign=eqn_sign,
            optimizer=optimizer,
            cpu=cpu,
            eval_every=eval_every,
            run_output_dir=run_output_dir,
            is_transformer=is_transformer,
            init_scaling=init_scaling,
            use_accelerator=use_accelerator,
            clip_grad=clip_grad,
            use_ln=use_ln,
            clock_pizza_metrics=clock_pizza_metrics,
            random_features=random_features,
            projection_dim=projection_dim,
            dataset_name=dataset_name
        )
    
    # Create the output directory if it does not exist
    os.makedirs(run_output_dir, exist_ok=True)


    # Train the model for each seed
    for seed in range(n_seeds):
        config["seed"] = seed

        model = Transformer(
            num_layers=config.get('n_layers',1),
            num_heads=config['n_heads'],
            d_model=config['d_model'],
            d_head=config.get('d_head',config['d_model']//config['n_heads']),
            attn_coeff=config['attn_coeff'],
            d_vocab=config['C'],
            act_type=config.get('act_fn','relu'),
            n_ctx=config['n_ctx'],
            use_ln=config['use_ln']
        )
        train_loader, test_loader = get_dataloaders(config)
        full_loader = get_dataloaders_with_full_dataset(config)
        model_manager = ModelManager(model,train_loader,test_loader,full_loader, config)
        model_manager.attach_hooks(hook_layers)
        model_manager.train_and_save_metrics()
    print("Finished training and saving metrics for all seeds")


    # Convert the training metrics to a CSV file
    training_run_json_to_csv(config['run_output_dir'], is_transformer=True, has_loss=True, lr=config['lr'], optimizer=config['optimizer'],
                                init_scaling=config['init_scaling'], input_dir=config['run_output_dir'], n_seeds=n_seeds, clock_pizza_metrics=config['clock_pizza_metrics'],random_features=config['random_features'])
    
    # training hmm
    max_components = 8
    cov_type = "diag"
    n_iter = 10
    cols = ['l1', 'l2', 'trace', 'spectral', 'code_sparsity',
       'computational_sparsity', 'mean_lambda', 'variance_lambda', 'mean_w',
       'median_w', 'var_w', 'mean_b', 'median_b', 'var_b','grad_sym',
       'dist_irr', 'train_loss', 'eval_loss', 'train_accuracy',
       'eval_accuracy',"feature_0","feature_1","feature_2","feature_3",
       "feature_4","feature_5","feature_6","feature_7","feature_8","feature_9"]

    hmm_model = HMM(max_components, cov_type, n_seeds, n_iter)
    hmm_output = hmm_model.get_avg_log_likelihood(f"{run_output_dir}/", cols)

    model_path = os.path.join(run_output_dir, "hmm_model")
    save_model(model_path,hmm_output)
    

if __name__ == "__main__":
    main()