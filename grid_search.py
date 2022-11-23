import argparse
import os.path
import itertools as it

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", choices=["wn18rr", "fb15k237"])
    parser.add_argument("output")
    parser.add_argument("--dimensions", type=int, default=[50])
    parser.add_argument("--alpha", type=int, default=[0.001])
    parser.add_argument("--margin", nargs="*", type=float, default=[0.5])
    parser.add_argument("--learning-rate", nargs="*", type=float, default=[0.1])
    parser.add_argument("--batch-size", nargs="*", type=int, default=[64])
    parser.add_argument("--epochs", nargs="*", type=int, default=[500])

    args = parser.parse_args()

    filtered_args = {key: value for key, value in vars(args).items()
                     if value is not None and key not in ["output", "dataset"]}

    keys, values = zip(*sorted(filtered_args.items()))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for value in it.product(*values):
        config = dict(zip(keys, value))

        config_key = '-'.join(["{}-{}".format(key, value) for key, value in config.items()])

        output_dir = os.path.join(args.output, config_key)

        argument_string = " ".join(
            ["--{} {}".format(key.replace("_", "-"), value) for key, value in config.items()]
        )

        with open(os.path.join(args.output, "{}.sh".format(config_key)), "w") as file:
            file.write(
                "(time python train.py --save-dir {} {} {}) &> {}".format(
                    output_dir,
                    argument_string,
                    args.dataset,
                    os.path.abspath(os.path.join(args.output, "{}.out".format(config_key))).replace(" ", "\ ")
                )
            )


if __name__ == "__main__":
    main()
