from pickle import load
from glob import glob
from os.path import join, exists
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", context="paper")

from tqdm import tqdm
import sys
sys.path.append('..')

from arguments import parse_arguments
from visualise import visualise_derivation_tree
from utils import load_config, get_exp_path


def compile_fn(node, args):
    backbone = node.build(node, set_memory_checkpoint=True)
    return Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        vars(args)
    ).to(args.device)


class Plotter:
    def __init__(self, results):
        if results is None:
            self.results = self.load_results(join(args.results_path, get_exp_path(args)))
        else:
            self.results = results

    def load_results(self, path):
        full_path = join(path, "search_results.pkl")
        if not exists(full_path):
            raise FileNotFoundError(f"No checkpoint found at {full_path}")
        results = load(open(full_path, "rb"))
        print(f"Successfully loaded {results['iteration'] + 1} result iterations")
        return results

    def plot_results(self, key, save_path):
        plt.figure(figsize=(6, 3))
        colors = sns.color_palette("tab10")
        # node_type = {"sequential": 0, "branching(2)": 1, "branching(4)": 2, "branching(8)": 3, "routing": 4, "computation": 5}
        node_type = {"sequential": 0, "sequential(4)": 1, "sequential(8)": 2, "branching(2)": 3, "branching(4)": 4, "branching(8)": 5, "routing": 6, "computation": 7}
        data = []
        for i, result in enumerate(self.results[key]):
            arch, reward = result[0], result[1]
            if len(result) not in [2, 4, 5]:
                raise ValueError("Unexpected result format")

            color = colors[node_type[arch[0].operation.name]]
            label = arch[0].operation.name
            data.append((i, reward, color, label))
        for i, reward, color, label in tqdm(data, desc=f"Plotting {key}"):
            plt.scatter(i, reward, color=color, label=label, alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1, 1))
        plt.xlabel("Iteration")
        plt.ylabel(key.capitalize())
        # plt.title(self.grammar_name)
        sns.despine()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(save_path, f"plot_{key}.pdf"))
        print(f"Saved plot to {join(save_path, f'plot_{key}.pdf')}")
        plt.close()

    def plot_num_params(self, save_path):
        plt.figure(figsize=(6, 3))
        data = []
        for i, result in enumerate(self.results["rewards"]):
            arch, reward = result[0], result[1]
            num_params = arch[0].num_params()
            data.append((i, num_params, reward))
        for i, num_params, reward in tqdm(data, desc="Plotting num_params"):
            plt.scatter(i, num_params, c=reward, cmap="viridis", alpha=0.5)
        plt.yscale('log')
        plt.colorbar()
        plt.xlabel("Iteration")
        plt.ylabel("Number of parameters")
        # plt.title(self.grammar_name)
        sns.despine()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(save_path, "plot_num_params.pdf"))
        print(f"Saved plot to {join(save_path, 'plot_num_params.pdf')}")
        plt.close()

    def plot_num_nodes(self, save_path):
        plt.figure(figsize=(6, 3))
        data = []
        for i, result in enumerate(self.results["rewards"]):
            arch, reward = result[0], result[1]
            num_nodes = len(arch)
            data.append((i, num_nodes, reward))
        for i, num_nodes, reward in tqdm(data, desc="Plotting num_nodes"):
            plt.scatter(i, num_nodes, c=reward, cmap="viridis", alpha=0.5)
        plt.yscale('log')
        plt.colorbar()
        plt.xlabel("Iteration")
        plt.ylabel("Number of nodes")
        # plt.title(self.grammar_name)
        sns.despine()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(save_path, "plot_num_nodes.pdf"))
        print(f"Saved plot to {join(save_path, 'plot_num_nodes.pdf')}")
        plt.close()

    def find_best_architecture(self):
        idx, best = max(enumerate(self.results["rewards"]), key=lambda x: x[1][1])
        best_arch, best_reward = best[0], best[1]
        return idx, best_arch, best_reward


if __name__ == "__main__":
    # parse the arguments
    args = parse_arguments()
    args = load_config(args)

    # create Plotter instance
    plotter = Plotter(results=None)

    # find best architecture
    idx, best_arch, best_reward = plotter.find_best_architecture()

    # visualise it
    visualise_derivation_tree(
        best_arch[0], iteration=f"best_{idx}", score=best_reward, show=False,
        save_path=join(args.figures_path, get_exp_path(args))
    )
    # plot results
    plotter.plot_results("rewards", join(args.figures_path, get_exp_path(args)))
    # plot number of parameters
    plotter.plot_num_params(join(args.figures_path, get_exp_path(args)))
    # plot number of nodes
    plotter.plot_num_nodes(join(args.figures_path, get_exp_path(args)))
