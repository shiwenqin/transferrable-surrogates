import os
from os.path import join

from surrogates.evaluators import SurrogateEvaluator, SelectiveEvaluator
from surrogates.predictors.base import SimpleSurrogate
from surrogates.predictors import predictor_cls
from surrogates.predictors.bert_pred import BertPredictor
from surrogates.encodings import get_encoder

from copy import deepcopy

from .random_search import RandomSearch
from .evolution import Evolution
from .surrogate_evolution import SurrogateEvolution, ChooseFromSampledEvolution, RejectionFilterEvolution
from .mcts import MCTS
import utils

__all__ = [
    "RandomSearch",
    "Evolution",
    "MCTS",
]


def create_search_strategy(args, grammar, evaluation_fn, limiter, input_params, data_loader, compile_fn=None, batch=None):
    # create the search strategy
    search_strategy = {
        "random_search": RandomSearch,
        "evolution": Evolution,
        "surrogate_evolution": SurrogateEvolution,
        "surrogate_sample_evolution": ChooseFromSampledEvolution,
        "surrogate_rejection_evolution": RejectionFilterEvolution,
        "mcts": MCTS,
    }[args.search_strategy]

    # specific parameters for each search strategy
    search_specific_params = {
        "random_search": {
            "figures_path": join(
                args.figures_path,
                utils.get_exp_path(args),
            ),
            "results_path": join(
                args.results_path,
                utils.get_exp_path(args),
            ),
        },
        "mcts": {
            "figures_path": join(
                args.figures_path,
                utils.get_exp_path(args),
            ),
            "results_path": join(
                args.results_path,
                utils.get_exp_path(args),
            ),
            "acquisition_fn": args.acquisition_fn,
            "exploration_weight": args.exploration_weight,
            "incubent_type": args.incubent_type,
            "reward_mode": args.reward_mode,
            "add_full_paths": args.add_full_paths,
        },
        "evolution": {
            "figures_path": join(
                args.figures_path,
                utils.get_exp_path(args),
            ),
            "results_path": join(
                args.results_path,
                utils.get_exp_path(args),
            ),
            "regularised": args.regularised,
            "population_size": args.population_size,
            "architecture_seed": args.architecture_seed,
            "mutation_strategy": args.mutation_strategy,
            "mutation_rate": args.mutation_rate,
            "crossover_strategy": args.crossover_strategy,
            "crossover_rate": args.crossover_rate,
            "selection_strategy": args.selection_strategy,
            "tournament_size": args.tournament_size,
            "elitism": args.elitism,
            "first_gen_path": args.first_gen_path,
            "gen_next_freq": args.gen_next_freq,
            "gen_next_n": args.gen_next_n,
            "batch": batch,
            "compile_fn": compile_fn,
            "n_tries": args.n_tries
        }
    }

    search_specific_params["surrogate_evolution"] = deepcopy(search_specific_params["evolution"])
    search_specific_params["surrogate_evolution"]["ground_truth_steps"] = args.ground_truth_steps
    search_specific_params["surrogate_evolution"]["refit_steps"] = args.refit_steps
    search_specific_params["surrogate_evolution"]["surrogate_start_iter"] = args.surrogate_start_iter

    search_specific_params["surrogate_sample_evolution"] = deepcopy(search_specific_params["surrogate_evolution"])
    search_specific_params["surrogate_sample_evolution"]["surrogate_n_sampled"] = args.surrogate_n_sampled
    search_specific_params["surrogate_sample_evolution"]["surrogate_n_chosen"] = args.surrogate_n_chosen

    search_specific_params["surrogate_rejection_evolution"] = deepcopy(search_specific_params["surrogate_sample_evolution"])

    # from sklearn.ensemble import RandomForestRegressor
    if args.surrogate == "bert":
        model = BertPredictor(random_state=args.seed, device=args.model_device, model_ckp=args.model_ckp, no_refit=args.no_refit)
    else:
        model = predictor_cls[args.surrogate](args.seed)
    encoder = get_encoder(data_loader, args)
    surrogate = SimpleSurrogate(model, encoder, fit_on_cached=args.fit_on_cached)

    if 'rejection' not in args.search_strategy:
        evaluator = SurrogateEvaluator(evaluation_fn, surrogate)
    else:
        evaluator = SelectiveEvaluator(evaluation_fn, surrogate, quantile=args.rejection_quantile)

    os.makedirs(search_specific_params[args.search_strategy]['figures_path'], exist_ok=True)

    search_specific_params["surrogate_evolution"]["evaluator"] = evaluator
    search_specific_params["surrogate_sample_evolution"]["evaluator"] = evaluator
    search_specific_params["surrogate_rejection_evolution"]["evaluator"] = evaluator

    # create the search
    search = search_strategy(
        # common parameters
        evaluation_fn=evaluation_fn,
        pcfg=grammar,
        limiter=limiter,
        input_params=input_params,
        seed=args.seed,
        mode=args.mode,
        backtrack=args.backtrack,
        verbose=args.verbose_search,
        visualise=args.visualise,
        visualise_scale=args.visualise_scale,
        vis_interval=args.vis_interval,
        continue_search=args.continue_search,
        # specific parameters
        **search_specific_params[args.search_strategy],
    )

    return search
