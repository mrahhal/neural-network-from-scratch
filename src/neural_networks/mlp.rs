//! MLP (multi-layer perceptron) neural network implementation.
//!
//! Legend:
//! ```plain
//! a: Activation value of a neuron.
//! z: Value of a neuron before activation (summation).
//! L: Loss of the network.
//! g_x: Gradient of x (∂L/∂x).
//! ```

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    f64::EPSILON,
    hash::Hash,
};

use nalgebra::{Dynamic, MatrixSlice, U1};

use crate::utils::{rnd_normal, softmax, Arith, Matrix, Vector};

/// An MLP deep neural network.
///
/// I: The number of neurons in the input layer.
/// O: The number of neurons in the output layer.
pub struct Network<const I: usize, const O: usize> {
    optimizer: Box<dyn Optimizer>,

    /// All layers after the input layer (i.e hidden layers + output layer).
    layers: Vec<Layer>,
}

fn create_layer(
    index: usize,
    prev_neurons_count: usize,
    neurons_count: usize,
    activator_creator: impl Fn() -> Box<dyn Activator>,
) -> Layer {
    let mut neurons = Vec::with_capacity(neurons_count);
    for _ in 0..neurons_count {
        let neuron = Neuron::new(
            generate_random_neuron_params(prev_neurons_count),
            activator_creator(),
        );

        neurons.push(neuron);
    }

    Layer::new(index, neurons)
}

impl<const I: usize, const O: usize> Network<I, O> {
    pub fn new() -> Self {
        let hidden_layers = [200, 80];
        let hidden_layers_count = hidden_layers.len();

        // + 1 output layer
        let mut layers = Vec::with_capacity(hidden_layers_count + 1);
        let mut prev_neurons_count = I;
        for i in 0..hidden_layers_count {
            let neurons_count = hidden_layers[i];
            layers.push(create_layer(i, prev_neurons_count, neurons_count, || {
                Box::new(ReLUActivator)
            }));
            prev_neurons_count = neurons_count;
        }

        layers.push(create_layer(
            hidden_layers_count,
            prev_neurons_count,
            O,
            || Box::new(NoopActivator),
        ));

        Self {
            optimizer: Box::new(AdamOptimizer::default()),
            layers,
        }
    }

    /// Run through the network and produce an output.
    pub fn run(&self, input: &[f64]) -> Vec<f64> {
        if I != input.len() {
            panic!("Mismatch of inputs length.")
        }

        let mut prev_activs = Vector::from_vec(input.to_owned());
        for layer in &self.layers {
            prev_activs = layer.forward(prev_activs);
        }

        let output_activs_non_normalized = prev_activs;
        let softmaxed = softmax(&output_activs_non_normalized);

        softmaxed.data.into()
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.optimizer.get_learning_rate()
    }

    pub fn train(&mut self, samples: &Vec<(Vec<f64>, Vec<f64>)>) {
        for (input, expected) in samples {
            self.backward(input, expected);
        }

        for layer in &mut self.layers {
            // Produce gradient mean.
            layer.gradients.as_mut().unwrap().mean(samples.len());
        }

        // Run the optimizer to do a gradient descent step.
        self.optimizer.enter();
        for layer in &mut self.layers.iter_mut() {
            self.optimizer.update(layer);
        }
        self.optimizer.exit();
    }

    fn backward(&mut self, input: &[f64], expected: &[f64]) {
        // output is softmaxed
        let output = self.run(&input);

        // Don't need it because it's implicit in d_l_z below.
        // let loss = CCELoss::sample_loss(&expected, &output);

        // Calculate deriv of CCE + Softmax and produce a Matrix to start feeding the first layer.
        let mut d_l_z = Matrix::zeros(output.len(), 1);
        for i in 0..output.len() {
            d_l_z[i] = output[i] - expected[i];
        }

        // Start applying those to layers in back order.
        let mut prev_values = d_l_z;
        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];
            prev_values = layer.backward(&prev_values);
        }

        for layer in &mut self.layers {
            let prev_layer_neuron_count = layer.neurons[0].params.weights.len();
            let curr_layer_neuron_count = layer.neurons.len();

            let mut weights = Matrix::zeros(prev_layer_neuron_count, curr_layer_neuron_count);
            let mut biases = Vec::with_capacity(curr_layer_neuron_count);

            for (i, n) in layer.neurons.iter().enumerate() {
                weights.set_column(i, &n.gradient.as_ref().unwrap().weights);
                biases.push(n.gradient.as_ref().unwrap().bias);
            }

            let layer_gradients = LayerGradients {
                weights,
                biases: Vector::from_vec(biases),
            };
            if layer.gradients.is_none() {
                layer.gradients.replace(layer_gradients);
            } else {
                layer.gradients.as_mut().unwrap().add(&layer_gradients);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_weights(&self) {
        for layer in &self.layers {
            for n in &layer.neurons {
                println!("{}", n.params.weights);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_biases(&self) {
        for layer in &self.layers {
            for n in &layer.neurons {
                println!("{}", n.params.bias);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_gradients(&self) {
        for layer in &self.layers {
            for n in &layer.neurons {
                println!("{}", n.gradient.as_ref().unwrap().weights);
                println!("{}", n.gradient.as_ref().unwrap().bias);
            }
        }
    }
}

struct LayerGradients {
    weights: Matrix,
    biases: Vector,
}

impl LayerGradients {
    fn add(&mut self, other: &LayerGradients) {
        self.weights += &other.weights;
        self.biases += &other.biases;
    }

    fn mean(&mut self, samples_count: usize) {
        self.weights /= samples_count as f64;
        self.biases /= samples_count as f64;
    }
}

/// Represents a layer in a network.
struct Layer {
    index: usize,

    /// The neurons forming the layer.
    neurons: Vec<Neuron>,

    // /// The activator used in the layer.
    // activator: Box<dyn Activator>,
    /// The current run's activations of the preceding layer.
    /// Used when backpropagating.
    prev_activs: RefCell<Option<Vector>>,

    gradients: Option<LayerGradients>,
}

// Only valid to compare layers of the same network.
impl PartialEq for Layer {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for Layer {}

impl Hash for Layer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl Layer {
    fn new(index: usize, neurons: Vec<Neuron>) -> Self {
        Layer {
            index,
            neurons,
            prev_activs: Default::default(),
            gradients: None,
        }
    }

    fn forward(&self, prev_activs: Vector) -> Vector {
        self.prev_activs.replace(Some(prev_activs));
        let prev_activs = self.prev_activs.borrow();
        let prev_activs = prev_activs.as_ref().unwrap();

        let mut activs = Vec::with_capacity(self.neurons.len());
        for n in &self.neurons {
            let activ = n.forward(prev_activs);
            activs.push(activ);
        }
        Vector::from_vec(activs)
    }

    // next_values includes a dot for W if this is a hidden layer.
    fn backward(&mut self, next_values: &Matrix) -> Matrix {
        let prev_activs = self.prev_activs.borrow();
        let prev_activs = prev_activs.as_ref().unwrap();

        let prev_layer_neuron_count = self.neurons[0].params.weights.len();
        let curr_layer_neuron_count = self.neurons.len();

        // Values for a single neuron will be organized in a column.
        let mut values_matrix = Matrix::zeros(prev_layer_neuron_count, curr_layer_neuron_count);
        for i in 0..self.neurons.len() {
            let neuron = &mut self.neurons[i];
            let next_values_for_neuron = next_values.row(i);
            let values = neuron.backward(prev_activs, next_values_for_neuron);
            values_matrix.set_column(i, &values);
        }

        values_matrix
    }
}

/// Represents a neuron in a layer.
struct Neuron {
    params: NeuronParams,
    /// Current activation of the neuron.
    z: Cell<f64>,
    activ: Cell<f64>,
    gradient: Option<NeuronGradient>,

    activator: Box<dyn Activator>,
}

impl Neuron {
    fn new(params: NeuronParams, activator: Box<dyn Activator>) -> Self {
        Self {
            params,
            z: Default::default(),
            activ: Default::default(),
            gradient: None,
            activator,
        }
    }

    /// Does a forward step.
    ///
    /// z = Σw.a + b
    fn forward(&self, prev_activs: &Vector) -> f64 {
        debug_assert!(self.params.weights.len() == prev_activs.len());

        let z = self.params.weights.dot(prev_activs) + self.params.bias;
        self.z.set(z);

        let activ = self.activator.activate(z);
        self.activ.set(activ);

        activ
    }

    /// Does a backward step, returning a vector of `g_z.W`.
    fn backward<'a, 'b>(
        &'a mut self,
        prev_activs: &Vector,
        next_values: MatrixSlice<'b, f64, U1, Dynamic, U1, Dynamic>,
    ) -> Vector {
        let z = self.z.get();

        // ∂a/∂z
        let d_a_z = self.activator.derivative(z);

        // ∂L/∂z
        // next_values are `g_z.W` from the next layer, or in the case of the output layer, it'll be the softmax+loss deriv.
        let g_z = d_a_z * next_values.sum();

        // ∂L/∂w
        let g_w = prev_activs * g_z;

        // ∂L/∂b
        let g_b = g_z;

        self.gradient.replace(NeuronGradient {
            weights: g_w,
            bias: g_b,
        });

        g_z * &self.params.weights
    }
}

struct NeuronParams {
    weights: Vector,
    bias: f64,
}

struct NeuronGradient {
    weights: Vector,
    bias: f64,
}

trait Activator {
    fn activate(&self, z: f64) -> f64;

    fn derivative(&self, z: f64) -> f64;
}

/// Rectified linear unit ([ReLU]) activation.
///
/// [ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
struct ReLUActivator;

impl Activator for ReLUActivator {
    /// a(z) =
    ///     0 if z <= 0
    ///     z if z > 0
    fn activate(&self, z: f64) -> f64 {
        z.max(0.0)
    }

    /// a' =
    ///     0 if z <= 0
    ///     1 if z > 0
    fn derivative(&self, z: f64) -> f64 {
        if z > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// [Sigmoid] activation.
/// Not supported yet, because an optimization that combines the deriv of CCE and ReLU is hard
/// coded right now for optimization.
///
/// [Sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
struct SigmoidActivator;

impl Activator for SigmoidActivator {
    /// a(z) = 1 / 1 + e^-z
    fn activate(&self, z: f64) -> f64 {
        1.0 / (1.0 + -z.exp())
    }

    /// a' = a . (1 - a)
    fn derivative(&self, z: f64) -> f64 {
        let a = self.activate(z);
        a * (1.0 - a)
    }
}

/// A noop activator, used in the output layer sometimes when the activation/derivation is done
/// in a separate process (i.e combining activation + loss as an optimization).
struct NoopActivator;

impl Activator for NoopActivator {
    fn activate(&self, z: f64) -> f64 {
        z
    }

    fn derivative(&self, _: f64) -> f64 {
        1.0
    }
}

pub trait Loss {
    /// Computes the loss for a single sample.
    fn sample_loss(expected: &[f64], predicted: &[f64]) -> f64;

    /// Computes the mean loss for all samples in the batch.
    #[inline]
    fn mean_loss(sample_losses: &[f64]) -> f64 {
        Self::total_loss(sample_losses) / sample_losses.len() as f64
    }

    /// Computes the total loss for all samples in the batch.
    #[inline]
    fn total_loss(sample_losses: &[f64]) -> f64 {
        sample_losses.iter().sum()
    }
}

/// Categorical cross entropy implementation of a loss function.
///
/// ye: expected
/// yp: predicted
/// L = -Σye.ln(yp)
///
/// Could have been simplified to the following if instead of expected vec, we get the index of
/// the class that should be 1, because we know other classes will be 0.
/// L = -ln(yp)
pub struct CCELoss;

impl Loss for CCELoss {
    fn sample_loss(expected: &[f64], predicted: &[f64]) -> f64 {
        if expected.len() != predicted.len() {
            panic!("Mismatch of length.")
        }

        let predicted: Vec<_> = predicted
            .iter()
            .map(|v| v.clamp(EPSILON, 1.0 - EPSILON))
            .collect();

        let mut sum = 0.0;
        for i in 0..expected.len() {
            sum += expected[i] * predicted[i].ln();
        }

        -sum
    }
}

/// Represents an optimizer that updates params (weights and biases) of layers accroding to
/// a certain algorithm working on the calculated gradients.
trait Optimizer {
    /// Called once for each chunk of training samples before calling updates on layers.
    fn enter(&mut self);

    /// Called for each layer to update the params.
    fn update(&mut self, layer: &mut Layer);

    /// Called once for each chunk of training samples after calling updates on layers.
    fn exit(&mut self);

    /// Don't override! Applies the calculated descent on the layer's params.
    fn update_for_descent(&mut self, layer: &mut Layer, d_weights: Matrix, d_biases: Vector) {
        for (ni, n) in &mut layer.neurons.iter_mut().enumerate() {
            n.params.weights += d_weights.column(ni);
            n.params.bias += d_biases[ni];
        }
    }

    fn get_learning_rate(&self) -> f64;
}

/// An optimizer that applies a constant unchanging learning rate to update params.
struct ConstantRateOptimizer {
    learning_rate: f64,
}

impl Default for ConstantRateOptimizer {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl ConstantRateOptimizer {
    fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for ConstantRateOptimizer {
    fn enter(&mut self) {}

    fn update(&mut self, layer: &mut Layer) {
        let g = layer.gradients.as_mut().unwrap();

        let d_weights = &g.weights * -self.learning_rate;
        let d_biases = &g.biases * -self.learning_rate;

        self.update_for_descent(layer, d_weights, d_biases);
    }

    fn exit(&mut self) {}

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

struct AdamOptimizer {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    beta1: f64,
    beta2: f64,
    iterations: u64,

    layer_data: HashMap<usize, AdamOptimizerLayerData>,
}

struct AdamOptimizerLayerData {
    weights_momentums: Matrix,
    weights_cache: Matrix,
    biases_momentums: Vector,
    biases_cache: Vector,
}

impl AdamOptimizer {
    fn new(learning_rate: f64, decay: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            current_learning_rate: learning_rate,
            decay,
            iterations: 0,
            beta1,
            beta2,

            layer_data: Default::default(),
        }
    }
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self::new(0.01, 0.0, 0.9, 0.999)
    }
}

impl Optimizer for AdamOptimizer {
    fn enter(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate =
                self.learning_rate * (1.0 / 1.0 + self.decay * self.iterations as f64);
        }
    }

    fn update(&mut self, layer: &mut Layer) {
        let g = layer.gradients.as_mut().unwrap();

        if !self.layer_data.contains_key(&layer.index) {
            let g_w = &g.weights;
            let g_b = &g.biases;
            self.layer_data.insert(
                layer.index,
                AdamOptimizerLayerData {
                    weights_momentums: Matrix::zeros(g_w.nrows(), g_w.ncols()),
                    weights_cache: Matrix::zeros(g_w.nrows(), g_w.ncols()),
                    biases_momentums: Vector::zeros(g_b.nrows()),
                    biases_cache: Vector::zeros(g_b.nrows()),
                },
            );
        }

        let data = self.layer_data.get_mut(&layer.index);
        let data = data.unwrap();

        data.weights_momentums =
            self.beta1 * &data.weights_momentums + (1.0 - self.beta1) * &g.weights;
        data.biases_momentums =
            self.beta1 * &data.biases_momentums + (1.0 - self.beta1) * &g.biases;

        let weights_momentums_corrected =
            &data.weights_momentums / (1.0 - self.beta1.powf(self.iterations as f64 + 1.0));
        let biases_momentums_corrected =
            &data.biases_momentums / (1.0 - self.beta1.powf(self.iterations as f64 + 1.0));

        data.weights_cache =
            self.beta2 * &data.weights_cache + (1.0 - self.beta2) * g.weights.pow2(2);
        data.biases_cache = self.beta2 * &data.biases_cache + (1.0 - self.beta2) * g.biases.pow2(2);

        let weights_cache_corrected =
            &data.weights_cache / (1.0 - self.beta2.powf(self.iterations as f64 + 1.0));
        let biases_cache_corrected =
            &data.biases_cache / (1.0 - self.beta2.powf(self.iterations as f64 + 1.0));

        let u_weights = -self.current_learning_rate
            * &weights_momentums_corrected
                .div2(&weights_cache_corrected.sqrt().add_scalar(EPSILON));
        let u_biases = -self.current_learning_rate
            * &biases_momentums_corrected.div2(&biases_cache_corrected.sqrt().add_scalar(EPSILON));

        self.update_for_descent(layer, u_weights, u_biases);
    }

    fn exit(&mut self) {
        self.iterations += 1;
    }

    fn get_learning_rate(&self) -> f64 {
        self.current_learning_rate
    }
}

fn generate_random_neuron_params(count: usize) -> NeuronParams {
    NeuronParams {
        weights: generate_random_weights(count),
        bias: 0f64,
    }
}

fn generate_random_weights(count: usize) -> Vector {
    let mut weights = Vec::with_capacity(count);

    for _ in 0..count {
        let r = rnd_normal();
        weights.push(r * 0.01);
    }

    Vector::from_vec(weights)
}
