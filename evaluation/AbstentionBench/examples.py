"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import inspect
import random

from recipe import datasets


def issubclass_strict(subclass, _class):
    return subclass != _class and issubclass(subclass, _class)


def main(seed):
    random.seed(seed)

    members = inspect.getmembers(datasets, inspect.isclass)

    for class_name, _class in members:
        if not issubclass_strict(_class, datasets.AbstentionDataset):
            continue

        if class_name.startswith("_"):
            continue

        dataset = _class()

        (example,) = dataset.sample_questions(1, should_abstain=True)

        print(f"Class: {class_name}, Example: {example}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41615561)
    args = parser.parse_args()

    main(args.seed)
