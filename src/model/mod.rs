use motor_calc_core::parameters::Parameters;
use num_traits::Float;

pub mod assembly;
pub mod calculation;
pub mod motor;
pub mod mounted_sensor;
pub mod sensor;

pub trait Adjustable<F: Float>: Sized {
    fn adjusted(&self, gradient: Parameters<F>) -> Self;

    fn adjust(&mut self, gradient: Parameters<F>) {
        *self = self.adjusted(gradient);
    }
}
