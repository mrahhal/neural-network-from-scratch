use std::fs::read;

use byteorder::{BigEndian, ReadBytesExt};
use nalgebra::{ArrayStorage, Matrix, U28};

pub type ImageData = Matrix<u8, U28, U28, ArrayStorage<u8, 28, 28>>;

pub struct Data {
    pub train: Vec<Image>,
    pub test: Vec<Image>,
}

#[derive(Clone, Debug)]
pub struct Image {
    pub label: u8,
    pub data: ImageData,
}

impl Image {
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(28 * 28);
        self.fill_vec(&mut v);
        v
    }

    /// Fill image data into the vec.
    pub fn fill_vec(&self, v: &mut Vec<f64>) {
        v.clear();
        for x in self.data.iter() {
            v.push((*x).into());
        }
    }

    pub fn to_label_distribution(&self) -> Vec<f64> {
        (0..10)
            .map(|i| if i == self.label { 1f64 } else { 0f64 })
            .collect()
    }
}

/// Loads and parses data from the mnist dataset files.
pub fn load_and_parse_data() -> Data {
    let train = parse_from_files(
        "datasets/mnist/train-images.idx3-ubyte",
        "datasets/mnist/train-labels.idx1-ubyte",
    );
    let test = parse_from_files(
        "datasets/mnist/t10k-images.idx3-ubyte",
        "datasets/mnist/t10k-labels.idx1-ubyte",
    );

    Data { train, test }
}

/// Parses mnist dataset files into images with their labels.
fn parse_from_files(images_path: &str, labels_path: &str) -> Vec<Image> {
    let images_bytes = read(images_path).unwrap();
    let labels_bytes = read(labels_path).unwrap();

    let number_of_items = (&images_bytes[4..8]).read_u32::<BigEndian>().unwrap() as usize;
    let number_of_rows = (&images_bytes[8..12]).read_u32::<BigEndian>().unwrap() as usize;
    let number_of_cols = (&images_bytes[12..16]).read_u32::<BigEndian>().unwrap() as usize;

    // Strip away well known header.
    let images_bytes = &images_bytes[16..];
    let labels_bytes = &labels_bytes[8..];

    let mut images = vec![];

    let mut images_bytes_offset = 0;
    for i in 0..number_of_items {
        let label = labels_bytes[i];
        let mut image = Image {
            label,
            data: Default::default(),
        };

        for row in 0..number_of_rows {
            for col in 0..number_of_cols {
                let pixel = images_bytes[images_bytes_offset];
                image.data[(row, col)] = pixel.into();
                images_bytes_offset += 1;
            }
        }

        images.push(image);
    }

    images
}
