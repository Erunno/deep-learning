#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    distributions = {}
    
    data_path = args.data_path if args.recodex else '/workdir/' + args.data_path
    model_path = args.model_path if args.recodex else '/workdir/' + args.model_path

    with open(data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            if line in distributions:
                distributions[line] += 1
            else:
                distributions[line] = 1
                
    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    keys = list(distributions.keys())

    vals = [distributions[key] for key in keys]
    sum_vals = sum(vals) * 1.0

    data_distribution = np.array(vals, dtype=np.float32) / sum_vals

    # TODO: Load model distribution, each line `string \t probability`.
    model_distribution = {}
    with open(model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            model_distribution[line.split('\t')[0]] = float(line.split('\t')[1])

    for key in keys:
        if key not in model_distribution:
            model_distribution[key] = 0

    model_dist_vals = [model_distribution[key] for key in keys]
    model_distribution = np.array(model_dist_vals, dtype=np.float32)

    # TODO: Create a NumPy array containing the model distribution.

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_distribution * np.log(data_distribution))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # the resulting crossentropy should be `np.inf`.
    crossentropy = -np.sum(data_distribution * np.log(model_distribution))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = np.sum(data_distribution * np.log(data_distribution / model_distribution))

    # Return the computed values for ReCodEx to validate.
    return float(entropy), float(crossentropy), float(kl_divergence)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
