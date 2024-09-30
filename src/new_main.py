import argparse
# from typing import List, Optional
import os

from new_log import init
from new_gd import prepare as prepare_gd
from new_gd import train as train_gd
from new_eval import prepare as prepare_eval
from new_eval import do_eval
from new_main_utils import load_checkpoint
from new_data_main import DATASETS as SUPPORTED_DATASETS

def main(args):
    if not "RESULTS" in os.environ:
        os.environ["RESULTS"] = os.path.join(args.result_dir, "results")
    if not "DATASETS" in os.environ:
        os.environ["DATASETS"] = os.path.join(args.dataset_dir, "data")

    init(args, "wandb_key.txt", "wandb_entity.txt", "wandb_project.txt")

    if args.type == "train":
        if args.method == "gd": # params = inspect.signature(method).parameters
            prepared_dict = prepare_gd(args)
            load_checkpoint(prepared_dict["network"], args.load_path)
            train_gd(args, **prepared_dict)

    elif args.type == "eval":
        prepared_dict = prepare_eval()
        load_checkpoint(prepared_dict["network"], args.load_path)
        do_eval(**prepared_dict)
    

if __name__ == "__main__":
    ###
    parser = argparse.ArgumentParser(description="Train using gradient descent / gradient flow")
    # TYPE
    parser.add_argument("--type", choices=["train", "eval"], # "aggregate"
                        help="what task to do")
    # TRAIN - GENERAL - MANDATORY
    parser.add_argument("--method", choices=["gd"], 
                        help="Which learning algorithm to choose") # "flow"
    parser.add_argument("--train_dataset", type=str, choices=SUPPORTED_DATASETS, 
                        help="which dataset used to train")
    parser.add_argument("--test_datasets", nargs='+', # type=List[str], 
                        help="which dataset to evaluate against")
    parser.add_argument("--arch_id", type=str, 
                        help="Which network architectures to train")
    parser.add_argument("--input_shape", nargs='+', # type=List[int], default=[1,28,28],
                        help="Size of a single data point")
    parser.add_argument("--output_shape", type=int, 
                        help="Number of classes", default=10)
    parser.add_argument("--softmax", action=argparse.BooleanOptionalAction, default=False,
                        help="Output of the network followed by softmax layer")    
    parser.add_argument("--loss", type=str, choices=["ce", "mse"], 
                        help="which loss function to use") # "focal"?
    # NETWORK - AUXILIARY
    
    # TRAIN - GENERAL - AUXILIARY
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False,
                        help="Traing on GPU - otherwise on CPU")
    parser.add_argument("--seed", type=int, 
                        help="The random seed used when initializing the network weights", default=0)
    parser.add_argument("--train_batch_size", type=int, default=1000,
                        help="The maximum number of examples on the GPU at once")
    parser.add_argument("--test_batch_size", type=int, default=1000,
                        help="The maximum number of examples on the GPU at once (for all test)")
    parser.add_argument("--train_min_acc", type=float, default=1.0,
                        help="Terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--train_max_loss", type=float, default=0.0,
                        help="Terminate training if the train loss ever crosses this value")
    parser.add_argument("--load_path", type=str, default=None,
                        help="How to initialize the model")
    #parser.add_argument("--trakked_data", action=argparse.BooleanOptionalAction, help="Whether to apply mask on data")
    #parser.add_argument("--trakked_models", type=str, default=None, help="Path dir to models on which select mask to data")
    # TRAIN - (S)GD - OPTIMIZER
    parser.add_argument("--max_iters", type=int, required="method"=="gd",
                        help="the maximum number of steps to train for")
    parser.add_argument("--opti_type", type=str, choices=["sam", "sgd"], # "polyak", "nesterov" 
                        help="Which optimization algorithm to use", default="sam")
    parser.add_argument("--lr", type=float, required="method"=="gd",
                        help="The learning rate")
    parser.add_argument("--aux_type", type=str, choices=["sam", "sgd"], default=None,
                        help="Which optimization algorithm to use for special part of the net")
    parser.add_argument("--aux_lr", type=float, default=None,
                        help="The learning rate of weight/features component of the optimizer")
    parser.add_argument("--data_type", type=str, choices=["sam", "sgd"], default=None,
                        help="Which optimization algorithm to use for special part of the batch",)
    parser.add_argument("--data_lr", type=float, default=None,
                        help="The learning rate of data component of the optimizer")
    parser.add_argument("--beta_momentum", type=float, required="opt"=="polyak" or "opt"=="nesterov",
                        help="Momentum parameter (used if opt = polyak or nesterov)", default=0.0)
    parser.add_argument("--delta_dampening", type=float, required="opt"=="polyak" or "opt"=="nesterov",
                        help="Dumpening parameter (not used with nesterov)", default=0.0)
    parser.add_argument("--rho_sam", type=float, required="method"=="sam",
                        help="Rho parameter for sharpness-Aware Minimization", default=0.05)
    parser.add_argument("--adaptive_sam", action=argparse.BooleanOptionalAction, required="method"=="sam",
                        help="Stochastic Weight Averaging", default=False)
    parser.add_argument("--swa", action=argparse.BooleanOptionalAction, default=False,
                        help="Stochastic Weight Averaging")
    parser.add_argument("--swa_lr", type=float, default=None,
                        help="LR parameter for stochastic weight averaging")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False,
                        help="Exponential Moving Average")
    parser.add_argument("--gamma_ema", type=float, default=None,
                        help="Ema gamma parameter for stochastic weight averaging")
    parser.add_argument("--omega_wd", type=float, 
                        help="regularization penalty (standard torch)", default=0.0)
    parser.add_argument("--omega_wd_0", type=float, 
                        help="regularization penalty for weight norm L0", default=0.0)
    parser.add_argument("--omega_wd_1", type=float, 
                        help="regularization penalty for weight norm L1", default=0.0)
    parser.add_argument("--omega_wd_2", type=float, 
                        help="regularization penalty for weight norm L2", default=0.0)
    parser.add_argument("--tau_0", type=float, 
                        help="regularization penalty for Fisher trace", default=0.0)
    parser.add_argument("--tau_1", type=float, 
                        help="regularization penalty for top eigenvalue of Hessian", default=0.0)
    parser.add_argument("--separate_data", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate batch and outline the special data")
    parser.add_argument("--separate_data_threshold", type=float, 
                        help="Separate batch and outline the special data coefficient", default=0.0)
    parser.add_argument("--separate_data_gradient_large", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate batch and outline the special data based on large gradient")
    parser.add_argument("--separate_data_gradient_small", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate batch and outline the special data based on small gradient")
    parser.add_argument("--separate_weights", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate weights and outline the special neurons")
    parser.add_argument("--separate_weights_threshold", type=float, 
                        help="Separate weights and outline the special neurons coefficient", default=0.0)
    parser.add_argument("--separate_weights_gradient_large", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate weights and outline the special neurons based on large gradient")
    parser.add_argument("--separate_weights_gradient_small", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate weights and outline the special neurons based on small gradient")
    parser.add_argument("--separate_features", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate weights and outline the special neurons on feature layer")
    parser.add_argument("--separate_features_threshold", type=float, 
                        help="Separate weights and outline the special neurons on feature layer coefficient", default=0.0)
    parser.add_argument("--separate_features_gradient_large", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate weights and outline the special neurons on feature layer based on large gradient")
    parser.add_argument("--separate_features_gradient_small", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate weights and outline the special neurons on feature layer based on small gradient")
    #parser.add_argument("--complex_freq", type=int, default=-1,
    #                    help="The frequency at which do fancy iteration (-1 never)")
    # TRAIN - (S)GD - ADDITIONAL

    # TRAIN - SHARPNESS
    parser.add_argument("--sharpness_batch_size", type=int, default=64,
                        help="Batch size to compute sh")
    parser.add_argument("--sharpness_neigs", type=int, default=2,
                        help="Number of top eigs (unless needed more for analysis)")
    parser.add_argument("--sharpness_repeats", type=int, default=None,
                        help="Number of batches to compute sh (None for full data)")
    parser.add_argument("--sharpness_quick_batch_size", type=int, default=64,
                        help="Batch size to compute sh quickly")
    parser.add_argument("--sharpness_num_iterations", type=int, default=10,
                        help="num_iterations to compute sh quickly with power iteration method")
    parser.add_argument("--sharpness_only_top", action=argparse.BooleanOptionalAction, default=False,
                        help="When true then use method Hessian yield only top eigenvalue")
    # TRAIN - RANDOM SHOCK
    parser.add_argument("--random_perturbation", action=argparse.BooleanOptionalAction, default=False,
                        help="Do shrink and perturbation operation")
    parser.add_argument("--random_perturbation_alpha", type=float, 
                        help="Shrink and perturbation coeff shrink", default=0.0)
    parser.add_argument("--random_perturbation_std", type=float, 
                        help="Shrink and perturbation coeff strength", default=0.0)
    parser.add_argument("--random_perturbation_threshold", type=float, 
                        help="Shrink and perturbation coeff selection", default=0.0)
    parser.add_argument("--random_reinitialization", action=argparse.BooleanOptionalAction, default=False,
                        help="Random reinitialization of the network")
    parser.add_argument("--random_reinitialization_threshold", type=float, 
                        help="Random reinitialization of the network coefficient", default=0.0)
    # TRAIN - SUBNETS

    # ANALYSIS
    parser.add_argument("--nproj", type=int, default=2,
                        help="Project gradient onto ?-dimensional space")
    parser.add_argument("--log_top_eigs", action=argparse.BooleanOptionalAction, default=False,
                        help="Plot biggest eigenvalues (2*len_labels)")
    parser.add_argument("--analysis_gradcam", action=argparse.BooleanOptionalAction, default=False,
                        help="Do gradcam analysis")
    parser.add_argument("--gradcam_batch_size", type=int, default=64,
                        help="Batch size to compute gradcam")
    parser.add_argument("--gradcam_counter", type=int, default=5,
                        help="Batch size to compute gradcam")
    parser.add_argument("--gradcam_rescale", action=argparse.BooleanOptionalAction, default=False,
                        help="Do gradcam rescaling at the end of img to [0,1] from [-1,1]")
    parser.add_argument("--analysis_logits", action=argparse.BooleanOptionalAction, default=False,
                        help="Do logits analysis")
    parser.add_argument("--analysis_selectivity", action=argparse.BooleanOptionalAction, default=False,
                        help="Do gradcam analysis")
    parser.add_argument("--selectivity_batch_size", type=int, default=64,
                        help="Batch size to compute selectivity")
    parser.add_argument("--selectivity_epsilon", type=float, default=0.005,
                        help="Epsilon for selectivity error")
    parser.add_argument("--analysis_perclass", action=argparse.BooleanOptionalAction, default=False,
                        help="Do logits analysis")
    parser.add_argument("--perclass_batch_size", type=int, default=64,
                        help="Batch size to compute perclass analysis")
    parser.add_argument("--analysis_sparsity", action=argparse.BooleanOptionalAction, default=False,
                        help="Do logits analysis") 
    parser.add_argument("--sparsity_epsilon", type=float, default=0.005,
                        help="Epsilon for sparsity")
    parser.add_argument("--analysis_eigenvalues", action=argparse.BooleanOptionalAction, default=False,
                        help="Do logits analysis")
    # TRAIN - FREQS
    parser.add_argument("--complex_freq", type=int, default=-1,
                        help="The frequency at which do fancy iteration (-1 never)")
    parser.add_argument("--generic_freq", type=int, default=-1,
                        help="The frequency at which calculate loss,acc for whole data (-1 never)")
    parser.add_argument("--gradient_freq", type=int, default=-1,
                        help="The frequency at which do gradient projection (-1 never)")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="The frequency at which do projection (-1 never)")
    parser.add_argument("--hessian_freq", type=int, default=-1,
                        help="The frequency at which calculate Hessian (-1 never)")
    parser.add_argument("--fisher_freq", type=int, default=-1,
                        help="The frequency at which calculate Fisher (-1 never)")
    parser.add_argument("--perturb_freq", type=int, default=-1,
                        help="The frequency at which do perturbation (-1 never)")
    parser.add_argument("--reinit_freq", type=int, default=-1,
                        help="The frequency at which do weight reinitialization (-1 never)")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="The frequency at which save model (-1 never)")
    parser.add_argument("--info_freq", type=int, default=-1,
                        help="The frequency at which info about training (-1 never)")
    parser.add_argument("--analysis_freq", type=int, default=-1,
                        help="The frequency at which do extra analysis (-1 never)")
    parser.add_argument("--skip_analysis_start", action=argparse.BooleanOptionalAction, default=False,
                        help="Do any analysis in iter freq-1 (otherwise first analysis for iter 0)")
    # TRAIN - AUXILIARY
    parser.add_argument("--dataset_dir", type=str, default=os.path.abspath(os.getcwd()))
    parser.add_argument("--result_dir", type=str, default=os.path.abspath(os.getcwd()))
    parser.add_argument("--path_extra", type=str, default="")
    parser.add_argument("--no_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_tag", nargs='+', default=None) # type=List[str], default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--save_end", action=argparse.BooleanOptionalAction, default=False,
                        help="save model weights and other results at end of training")
    ### 
    args = parser.parse_args()
    main(args)
