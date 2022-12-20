mod datasets;
mod neural_networks;
mod utils;

use std::time::Instant;

use crate::datasets::mnist::{Data, Image};
use crate::neural_networks::basic::{self as nn, CCELoss, Loss};
use crate::utils::{argmax, shuffle_vec};

fn main() {
    let Data {
        mut train,
        mut test,
    } = datasets::mnist::load_and_parse_data();

    let mut network = nn::Network::<{ 28 * 28 }, 10>::new();

    // let sample = &train[0];

    // let input = sample.to_vec();
    // let output = network.run(&input);

    // let distribution_sum: f64 = output.iter().sum();
    // // Make sure things look good.
    // debug_assert!(distribution_sum > 0.99f64 && distribution_sum < 1.01f64);

    // println!(
    //     "{}, sum {}",
    //     Vector::from_vec(output.clone()),
    //     distribution_sum
    // );

    // let expected = sample.to_label_distribution();
    // let loss = CCELoss::sample_loss(&expected, &output);

    // println!(
    //     "expected: {}, predicted: {}, loss: {}",
    //     Vector::from_vec(expected.clone()),
    //     Vector::from_vec(output.clone()),
    //     loss,
    // );

    shuffle_vec(&mut train);
    // let validate: Vec<Image> = train.drain(50_000..).collect();
    let train: Vec<Image> = train.drain(..5_000).collect();
    let test: Vec<Image> = test.drain(..1_000).collect();

    println!("BEFORE TRAINING");

    // Before training
    // ---

    let mut prev_values: Option<(f64, f64)> = None;
    prev_values = Some(test_on(&network, &test, &prev_values));

    // Training
    // ---

    let target_accuracy = 0.95;

    println!("===");
    println!(
        "Training and validating accuracy after each epoch. Target accuracy is {}%.",
        target_accuracy * 100f64,
    );

    let instant_total = Instant::now();

    let mut epochs = 1;
    let dataset: Vec<_> = train
        .iter()
        .map(|image| (image.to_vec(), image.to_label_distribution()))
        .collect();
    loop {
        println!("-------------------");
        print!("Epoch #{}: training on {} samples...", epochs, dataset.len(),);
        let instant = Instant::now();

        for chunk in train.chunks(100) {
            let dataset = chunk
                .iter()
                .map(|image| (image.to_vec(), image.to_label_distribution()))
                .collect();
            network.train(&dataset);
        }

        println!(" took {:.1} secs", instant.elapsed().as_secs_f64());

        let (accuracy, mean_loss) = test_on(&network, &test, &prev_values);
        prev_values = Some((accuracy, mean_loss));

        if accuracy > target_accuracy {
            println!(
                "Reached target accuracy. Took {:.1} secs. Exiting.",
                instant_total.elapsed().as_secs_f64(),
            );
            break;
        }

        epochs += 1;
    }
}

fn test_on<const I: usize, const O: usize>(
    network: &nn::Network<I, O>,
    samples: &[Image],
    prev_values: &Option<(f64, f64)>,
) -> (f64, f64) {
    let total = samples.len() as f64;
    let mut success = 0;
    let mut total_loss = 0f64;
    for sample in samples {
        let input = sample.to_vec();
        let output = network.run(&input);
        let predicted = argmax(&output);
        let expected = sample.to_label_distribution();
        success += if predicted == sample.label.into() {
            1
        } else {
            0
        };
        let loss = CCELoss::sample_loss(&expected, &output);
        total_loss += loss;
    }
    let accuracy: f64 = success as f64 / total;
    let mean_loss: f64 = total_loss / total;

    let prev_accuracy = prev_values.map_or(None, |t| Some(t.0));
    let prev_mean_loss = prev_values.map_or(None, |t| Some(t.1));
    let accuracy_diff: f64 = prev_accuracy.map_or(0.0, |prev_accuracy| {
        (accuracy * 100f64) - (prev_accuracy * 100f64)
    });
    let loss_diff: f64 = prev_mean_loss.map_or(0.0, |prev_mean_loss| mean_loss - prev_mean_loss);
    print!("Accuracy: {:.2}%", accuracy * 100f64);
    if accuracy_diff == 0.0 {
        println!();
    } else {
        println!(
            " ({}{:.2})",
            if accuracy_diff > 0.0 { "+" } else { "" },
            accuracy_diff
        );
    }
    print!("Mean loss: {:.5}", mean_loss);
    if loss_diff == 0.0 {
        println!();
    } else {
        println!(
            " ({}{:.5})",
            if loss_diff > 0.0 { "+" } else { "" },
            loss_diff
        );
    }
    println!("LR: {}", network.get_learning_rate());

    (accuracy, mean_loss)
}
