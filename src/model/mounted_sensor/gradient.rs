use motor_calc_core::parameters::Parameters;
use num_traits::Float;

use crate::{
    model::mounted_sensor::MountedSensorModel,
    types::{
        sensor_type::{Beta, SensorType},
        shaft_angle::ShaftAngle,
    },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientMountedSensorModel<'g, 'm, 's, F: Float, T: SensorType> {
    mounted_sensor: &'g MountedSensorModel<'m, 's, F, T>,
}

impl<'g, 'm, 's, F: Float, T: SensorType> GradientMountedSensorModel<'g, 'm, 's, F, T> {
    pub fn new(mounted_sensor: &'g MountedSensorModel<'m, 's, F, T>) -> Self {
        Self { mounted_sensor }
    }

    pub fn mounted_sensor(self) -> &'g MountedSensorModel<'m, 's, F, T> {
        self.mounted_sensor
    }
}

impl<'g, 'm, 's, F: Float> GradientMountedSensorModel<'g, 'm, 's, F, Beta> {
    pub fn grad_shaft_angle_offset(self) -> Parameters<F> {
        Parameters::just_beta_shaft_angle_offset()
    }

    pub fn grad_sin_estimated_shaft_angle_offset(self) -> Parameters<F> {
        motor_calc_core::gradient::grad_sin_beta_shaft_angle_offset(
            self.mounted_sensor()
                .parameters()
                .shaft_angle_offset()
                .cos(),
            self.grad_shaft_angle_offset(),
        )
    }

    pub fn grad_cos_estimated_shaft_angle_offset(self) -> Parameters<F> {
        motor_calc_core::gradient::grad_cos_beta_shaft_angle_offset(
            self.mounted_sensor()
                .parameters()
                .shaft_angle_offset()
                .sin(),
            self.grad_shaft_angle_offset(),
        )
    }

    pub fn grad_sin_cos_estimated_shaft_angle_offset(self) -> (Parameters<F>, Parameters<F>) {
        let (sin_beta_shaft_angle_offset, cos_beta_shaft_angle_offset) = self
            .mounted_sensor()
            .parameters()
            .shaft_angle_offset()
            .sin_cos();

        (
            motor_calc_core::gradient::grad_sin_beta_shaft_angle_offset(
                cos_beta_shaft_angle_offset,
                self.grad_shaft_angle_offset(),
            ),
            motor_calc_core::gradient::grad_cos_beta_shaft_angle_offset(
                sin_beta_shaft_angle_offset,
                self.grad_shaft_angle_offset(),
            ),
        )
    }

    pub fn grad_estimated_relative_shaft_angle(
        self,
        grad_estimated_shaft_angle: Parameters<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_estimated_relative_shaft_angle(
            grad_estimated_shaft_angle,
            self.grad_shaft_angle_offset(),
        )
    }

    pub fn grad_sin_estimated_relative_shaft_angle(
        self,
        estimated_shaft_angle: ShaftAngle<F>,
        grad_estimated_shaft_angle: Parameters<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_sin_estimated_relative_shaft_angle(
            self.mounted_sensor()
                .forward()
                .relative_shaft_angle(estimated_shaft_angle)
                .cos(),
            self.grad_estimated_relative_shaft_angle(grad_estimated_shaft_angle),
        )
    }

    pub fn grad_cos_estimated_relative_shaft_angle(
        self,
        estimated_shaft_angle: ShaftAngle<F>,
        grad_estimated_shaft_angle: Parameters<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_cos_estimated_relative_shaft_angle(
            self.mounted_sensor()
                .forward()
                .relative_shaft_angle(estimated_shaft_angle)
                .sin(),
            self.grad_estimated_relative_shaft_angle(grad_estimated_shaft_angle),
        )
    }

    pub fn grad_sin_cos_estimated_relative_shaft_angle(
        self,
        estimated_shaft_angle: ShaftAngle<F>,
        grad_estimated_shaft_angle: Parameters<F>,
    ) -> (Parameters<F>, Parameters<F>) {
        let (sin_estimated_relative_shaft_angle, cos_estimated_relative_shaft_angle) = self
            .mounted_sensor()
            .forward()
            .relative_shaft_angle(estimated_shaft_angle)
            .sin_cos();

        (
            motor_calc_core::gradient::grad_sin_estimated_relative_shaft_angle(
                cos_estimated_relative_shaft_angle,
                self.grad_estimated_relative_shaft_angle(grad_estimated_shaft_angle),
            ),
            motor_calc_core::gradient::grad_cos_estimated_relative_shaft_angle(
                sin_estimated_relative_shaft_angle,
                self.grad_estimated_relative_shaft_angle(grad_estimated_shaft_angle),
            ),
        )
    }
}
