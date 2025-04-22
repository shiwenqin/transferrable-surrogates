import argparse


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Run MCTS on a grammar.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="The config file to use.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use.")
    # search details
    parser.add_argument("--search_space", type=str, default="einspace", help="The grammar to use.")
    parser.add_argument("--search_strategy", type=str, default="mcts", help="The search strategy to use.")
    parser.add_argument("--steps", type=int, default=1000, help="The number of search steps.")
    parser.add_argument("--backtrack", action="store_true", help="Backtrack when out of options.")
    parser.add_argument("--mode", type=str, default="iterative", help="The mode to use.")
    parser.add_argument("--time_limit", type=int, default=300, help="The time limit to use.")
    parser.add_argument("--max_id_limit", type=int, default=10000, help="The maximum ID limit to use.")
    parser.add_argument("--depth_limit", type=int, default=20, help="The depth limit to use.")
    parser.add_argument("--mem_limit", type=int, default=4096, help="The memory limit in MB to use.")
    parser.add_argument("--individual_mem_limit", type=int, default=1024, help="The memory limit in MB to use.")
    parser.add_argument("--batch_pass_limit", type=int, default=0.1, help="The batch pass limit in seconds to use.")
    # search strategy specific details
    # mcts
    parser.add_argument("--acquisition_fn", type=str, default="uct", help="The acquisition function to use.")
    parser.add_argument("--exploration_weight", type=float, default=1.0, help="The exploration weight to use.")
    parser.add_argument("--incubent_type", type=str, default="parent", help="The incubent type to use in Expected Improvement")
    parser.add_argument("--reward_mode", type=str, default="sum", help="The reward mode to use.")
    parser.add_argument("--add_full_paths", action="store_true", help="Add the full paths to the search tree in MCTS.")
    # evolution
    parser.add_argument("--regularised", action="store_true", help="Use regularised evolution.")
    parser.add_argument("--population_size", type=int, default=100, help="The population size to use.")
    parser.add_argument("--architecture_seed", type=str, default=None, help="Baseline architectures to seed the search with, separate with '+', e.g. resnet18+vit+mlpmixer.")
    parser.add_argument("--mutation_strategy", type=str, default="random", help="The mutation strategy to use.")
    parser.add_argument("--mutation_rate", type=float, default=1.0, help="The mutation rate to use.")
    parser.add_argument("--crossover_strategy", type=str, default="random", help="The crossover strategy to use.")
    parser.add_argument("--crossover_rate", type=float, default=0.5, help="The crossover rate to use.")
    parser.add_argument("--selection_strategy", type=str, default="tournament", help="The selection strategy to use.")
    parser.add_argument("--tournament_size", type=int, default=10, help="The tournament size to use.")
    parser.add_argument("--elitism", action="store_true", help="Use regularised evolution.")
    parser.add_argument("--n_tries", type=int, default=None, help="The number of tries to use in evolution before randomly generating an individual.")
    # evaluation details
    parser.add_argument("--dataset", type=str, default="mnist", help="The dataset to use.")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use.")
    parser.add_argument("--channels", type=int, default=1, help="The number of channels in the data.")
    # parser.add_argument("--image_size", type=int, default=28, help="The size of the image.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use.")
    parser.add_argument("--load_in_gpu", action="store_true", help="Load the data in GPU.")
    # logging and plotting
    parser.add_argument("--verbose_search", action="store_true", help="Print verbose output during search.")
    parser.add_argument("--verbose_eval", action="store_true", help="Print verbose output during evaluation.")
    parser.add_argument("--visualise", action="store_true", help="Visualise the derivation tree.")
    parser.add_argument("--visualise_scale", type=float, default=0.8, help="The scale of the visualisation.")
    parser.add_argument("--vis_interval", type=int, default=10, help="The interval to log the results.")
    # saving results and figures
    parser.add_argument("--figures_path", type=str, default="figures", help="The path to save the figures.")
    parser.add_argument("--results_path", type=str, default="results", help="The path to save the results.")
    # continue search
    parser.add_argument("--continue_search", action="store_true", help="Continue the search.")
    # Surrogate model
    parser.add_argument("--surrogate", type=str, default="rf", help="Which surrogate used for search.")
    parser.add_argument("--use_zcp", action="store_true", help="Use zero-cost proxy based encoding.")
    parser.add_argument("--use_features", action="store_true", help="Use GRAF based encoding.")
    parser.add_argument("--ground_truth_steps", type=int, default=None, help="How often to eval ground truth instead of the surrogate.")
    parser.add_argument("--refit_steps", type=int, default=1, help="How often to refit the surrogate.")
    parser.add_argument("--surrogate_n_sampled", type=int, default=20, help="How many samples to use for the surrogate.")
    parser.add_argument("--surrogate_n_chosen", type=int, default=1, help="How many best samples to choose and get ground truth for.")
    parser.add_argument("--fit_on_cached", type=bool, default=False, help="If True, fit the surrogate on the whole archive of encountered architectures.")
    parser.add_argument("--first_gen_path", type=str, default=None, help="Path to saved first generation of individuals.")
    parser.add_argument("--gen_next_freq", type=int, default=None, help="M - After M generations, evaluate and save N individuals.")
    parser.add_argument("--gen_next_n", type=int, default=20, help="N - After M generations, evaluate and save N individuals.")
    parser.add_argument("--surrogate_start_iter", type=int, default=0, help="Use the surrogate only after N iterations.")
    parser.add_argument("--prefix", type=str, default=None, help="Path prefix for experiments.")
    parser.add_argument("--rejection_quantile", type=float, default=0.5, help="The quantile to use for rejection.")

    # Surrogate model - LM
    parser.add_argument("--model_ckp", type=str, default=None, help="Path to the LM surrogate checkpoint")
    parser.add_argument("--model_device", type=str, default="cuda:0", help="The device to use for the LM surrogate")
    parser.add_argument("--no_refit", type=bool, default=False, help="Refit the LM surrogate")

    if args is not None:
        args = parser.parse_args(args=args)
    else:
        args = parser.parse_args()
    return args
