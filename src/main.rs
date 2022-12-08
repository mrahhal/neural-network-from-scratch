mod datasets;

fn main() {
    let train_data = datasets::mnist::parse(
        "dataset/train-images.idx3-ubyte",
        "dataset/train-labels.idx1-ubyte",
    );
    let test_data = datasets::mnist::parse(
        "dataset/t10k-images.idx3-ubyte",
        "dataset/t10k-labels.idx1-ubyte",
    );
}
