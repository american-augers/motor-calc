use motor_calc_core::parameters::Parameters;
use num_traits::Float;

use crate::{
    model::{calculation::CalculationContext, sensor::SensorModel},
    types::{
        sensor_type::{Alpha, Beta, SensorType},
        voltage::Voltage,
    },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientSensorModel<'g, F: Float, T: SensorType> {
    sensor: &'g SensorModel<F, T>,
}

impl<'g, F: Float, T: SensorType> GradientSensorModel<'g, F, T> {
    pub fn new(sensor: &'g SensorModel<F, T>) -> Self {
        Self { sensor }
    }

    pub fn sensor(self) -> &'g SensorModel<F, T> {
        self.sensor
    }
}

impl<'g, F: Float> GradientSensorModel<'g, F, Alpha> {
    pub fn grad_voltage(self) -> Parameters<F> {
        Parameters::zero()
    }

    pub fn grad_estimated_voltage_scale(self) -> Parameters<F> {
        Parameters::just_alpha_voltage_scale()
    }

    pub fn grad_estimated_voltage_offset(self) -> Parameters<F> {
        Parameters::just_alpha_voltage_offset()
    }

    pub fn grad_estimated_offset_voltage(self) -> Parameters<F> {
        motor_calc_core::gradient::grad_estimated_offset_alpha_voltage(
            self.grad_voltage(),
            self.grad_estimated_voltage_offset(),
        )
    }

    pub fn grad_estimated_sensor_angle(self, alpha_voltage: Voltage<F, Alpha>) -> Parameters<F> {
        motor_calc_core::gradient::grad_estimated_alpha_sensor_angle(
            self.grad_estimated_offset_voltage(),
            self.sensor().parameters().voltage_scale(),
            alpha_voltage.volts(),
            self.sensor().parameters().voltage_offset().volts(),
            self.grad_estimated_voltage_scale(),
        )
    }

    pub fn grad_tan_estimated_sensor_angle(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_tan_estimated_alpha_sensor_angle(
            context
                .estimated_alpha_sensor_angle(self.sensor().inverse())
                .tan(),
            self.grad_estimated_sensor_angle(context.alpha_voltage()),
        )
    }
}

impl<'g, F: Float> GradientSensorModel<'g, F, Beta> {
    pub fn grad_voltage(self) -> Parameters<F> {
        Parameters::zero()
    }

    pub fn grad_estimated_voltage_scale(self) -> Parameters<F> {
        Parameters::just_beta_voltage_scale()
    }

    pub fn grad_estimated_voltage_offset(self) -> Parameters<F> {
        Parameters::just_beta_voltage_offset()
    }

    pub fn grad_estimated_offset_voltage(self) -> Parameters<F> {
        motor_calc_core::gradient::grad_estimated_offset_beta_voltage(
            self.grad_voltage(),
            self.grad_estimated_voltage_offset(),
        )
    }

    pub fn grad_estimated_sensor_angle(self, beta_voltage: Voltage<F, Beta>) -> Parameters<F> {
        motor_calc_core::gradient::grad_estimated_alpha_sensor_angle(
            self.grad_estimated_offset_voltage(),
            self.sensor().parameters().voltage_scale(),
            beta_voltage.volts(),
            self.sensor().parameters().voltage_offset().volts(),
            self.grad_estimated_voltage_scale(),
        )
    }

    pub fn grad_tan_estimated_sensor_angle(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_tan_estimated_beta_sensor_angle(
            context
                .estimated_beta_sensor_angle(self.sensor().inverse())
                .tan(),
            self.grad_estimated_sensor_angle(context.beta_voltage()),
        )
    }
}
