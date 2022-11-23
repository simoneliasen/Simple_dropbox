import argparse
import params
import trainer_tester

def main(dataset, dimensions=50, alpha=0.001, batch_size=64, epochs=500, learning_rate=0.01, margin=1., save_dir=None):
    p = params.Params()
    p.set_values(lr=learning_rate, alpha=alpha, bsize=batch_size, max_itr=epochs, emb_size=dimensions, neg_ratio=1, gamma=margin)
    tt = trainer_tester.TrainerTester(
        model_name="SimplE_avg",
        dataset=dataset,
        params=p,
        save_dir=save_dir
    )
    tt.train_earlystop_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wn18rr", "fb15k237"])
    parser.add_argument("--dimensions", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--save-dir", type=str)

    print(parser.parse_args())

    main(**vars(parser.parse_args()))
