use core::{
    marker::PhantomData,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
};

use num_traits::Float;

use crate::types::sensor_type::SensorType;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voltage<F: Float, T: SensorType> {
    volts: F,
    sensor_type: PhantomData<T>,
}

impl<F: Float, T: SensorType> Voltage<F, T> {
    pub fn from_volts_f(volts: F) -> Self {
        Self {
            volts,
            sensor_type: PhantomData,
        }
    }

    pub fn from_millivolts_u32(millivolts: u32) -> Option<Self> {
        F::from(millivolts).map(|millivolts_f| {
            Self::from_volts_f(
                millivolts_f / F::from(1000).expect("1000 can always be represented with a float"),
            )
        })
    }

    pub fn volts(self) -> F {
        self.volts
    }
}

impl<F: Float, T: SensorType> Deref for Voltage<F, T> {
    type Target = F;

    fn deref(&self) -> &Self::Target {
        &self.volts
    }
}

impl<F: Float, T: SensorType> Add for Voltage<F, T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_volts_f(self.volts() + rhs.volts())
    }
}

impl<F: Float, T: SensorType> Sub for Voltage<F, T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Voltage::from_volts_f(self.volts() - rhs.volts())
    }
}

impl<F: Float, T: SensorType> Mul<F> for Voltage<F, T> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Voltage::from_volts_f(self.volts() * rhs)
    }
}

impl<F: Float, T: SensorType> Div<F> for Voltage<F, T> {
    type Output = Self;

    fn div(self, rhs: F) -> Self::Output {
        Voltage::from_volts_f(self.volts() / rhs)
    }
}

impl<F: Float, T: SensorType> Neg for Voltage<F, T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Voltage::from_volts_f(-self.volts())
    }
}
