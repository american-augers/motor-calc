use num_traits::Float;

use crate::{
    model::mounted_sensor::MountedSensorModel,
    types::{sensor_type::SensorType, shaft_angle::ShaftAngle},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseMountedSensorModel<'i, 'm, 's, F: Float, T: SensorType> {
    mounted_sensor: &'i MountedSensorModel<'m, 's, F, T>,
}

impl<'i, 'm, 's, F: Float, T: SensorType> InverseMountedSensorModel<'i, 'm, 's, F, T> {
    pub fn new(mounted_sensor: &'i MountedSensorModel<'m, 's, F, T>) -> Self {
        Self { mounted_sensor }
    }

    pub fn mounted_sensor(self) -> &'i MountedSensorModel<'m, 's, F, T> {
        self.mounted_sensor
    }

    pub fn estimated_relative_shaft_angle(
        self,
        estimated_shaft_angle: ShaftAngle<F>,
    ) -> ShaftAngle<F> {
        ShaftAngle::from_radians_f(motor_calc_core::forward::relative_shaft_angle(
            estimated_shaft_angle.radians(),
            self.mounted_sensor()
                .parameters()
                .shaft_angle_offset()
                .radians(),
        ))
    }
}
