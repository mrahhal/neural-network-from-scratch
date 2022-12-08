mod datasets;

fn main() {
    let train_data = datasets::mnist::parse(
        "datasets/mnist/train-images.idx3-ubyte",
        "datasets/mnist/train-labels.idx1-ubyte",
    );
    let test_data = datasets::mnist::parse(
        "datasets/mnist/t10k-images.idx3-ubyte",
        "datasets/mnist/t10k-labels.idx1-ubyte",
    );
}
