mod datasets;

use datasets::mnist::Data;

fn main() {
    let Data { train, test } = datasets::mnist::load_and_parse_data();

    println!("Number of train data: {}", train.len());
    println!("Number of test data: {}", test.len());
}
