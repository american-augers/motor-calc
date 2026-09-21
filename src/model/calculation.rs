use num_traits::Float;

use crate::{
    model::{assembly::inverse::InverseAssemblyModel, sensor::inverse::InverseSensorModel},
    types::{
        sensor_angle::SensorAngle,
        sensor_type::{Alpha, Beta},
        shaft_angle::ShaftAngle,
        voltage::Voltage,
    },
};

pub struct CalculationContext<F: Float> {
    alpha_voltage: Voltage<F, Alpha>,
    beta_voltage: Voltage<F, Beta>,
    maybe_estimated_alpha_sensor_angle: Option<SensorAngle<F, Alpha>>,
    maybe_estimated_beta_sensor_angle: Option<SensorAngle<F, Beta>>,
    maybe_estimated_shaft_angle: Option<ShaftAngle<F>>,
    maybe_alpha_estimated_displacement: Option<Option<F>>,
    maybe_beta_estimated_displacement: Option<Option<F>>,
    maybe_estimated_displacement: Option<F>,
    maybe_estimated_beta_relative_shaft_angle: Option<ShaftAngle<F>>,
    maybe_estimated_beta_sensor_shaft_angle_offset: Option<ShaftAngle<F>>,
}

impl<F: Float> CalculationContext<F> {
    pub fn new(alpha_voltage: Voltage<F, Alpha>, beta_voltage: Voltage<F, Beta>) -> Self {
        Self {
            alpha_voltage,
            beta_voltage,
            maybe_estimated_alpha_sensor_angle: None,
            maybe_estimated_beta_sensor_angle: None,
            maybe_estimated_shaft_angle: None,
            maybe_alpha_estimated_displacement: None,
            maybe_beta_estimated_displacement: None,
            maybe_estimated_displacement: None,
            maybe_estimated_beta_relative_shaft_angle: None,
            maybe_estimated_beta_sensor_shaft_angle_offset: None,
        }
    }

    pub fn alpha_voltage(&self) -> Voltage<F, Alpha> {
        self.alpha_voltage
    }

    pub fn beta_voltage(&self) -> Voltage<F, Beta> {
        self.beta_voltage
    }

    pub fn estimated_alpha_sensor_angle(
        &mut self,
        inverse_sensor_model: InverseSensorModel<'_, F, Alpha>,
    ) -> &mut SensorAngle<F, Alpha> {
        if self.maybe_estimated_alpha_sensor_angle.is_none() {
            let estimated_alpha_sensor_angle =
                inverse_sensor_model.estimated_sensor_angle(self.alpha_voltage);

            self.maybe_estimated_alpha_sensor_angle = Some(estimated_alpha_sensor_angle);
        }

        self.maybe_estimated_alpha_sensor_angle.as_mut().unwrap()
    }

    pub fn estimated_beta_sensor_angle(
        &mut self,
        inverse_sensor_model: InverseSensorModel<'_, F, Beta>,
    ) -> &mut SensorAngle<F, Beta> {
        if self.maybe_estimated_beta_sensor_angle.is_none() {
            let estimated_beta_sensor_angle =
                inverse_sensor_model.estimated_sensor_angle(self.beta_voltage);

            self.maybe_estimated_beta_sensor_angle = Some(estimated_beta_sensor_angle);
        }

        self.maybe_estimated_beta_sensor_angle.as_mut().unwrap()
    }

    pub fn estimated_shaft_angle(
        &mut self,
        inverse_assembly_model: InverseAssemblyModel<'_, F>,
    ) -> &mut ShaftAngle<F> {
        if self.maybe_estimated_shaft_angle.is_none() {
            let estimated_shaft_angle = inverse_assembly_model.estimated_shaft_angle(self);

            self.maybe_estimated_shaft_angle = Some(estimated_shaft_angle);
        }

        self.maybe_estimated_shaft_angle.as_mut().unwrap()
    }

    pub fn alpha_estimated_displacement(
        &mut self,
        inverse_assembly_model: InverseAssemblyModel<'_, F>,
    ) -> Option<F> {
        if self.maybe_alpha_estimated_displacement.is_none() {
            let alpha_estimated_displacement =
                inverse_assembly_model.alpha_estimated_displacement(self);

            self.maybe_alpha_estimated_displacement = Some(alpha_estimated_displacement);
        }

        self.maybe_alpha_estimated_displacement.unwrap()
    }

    pub fn beta_estimated_displacement(
        &mut self,
        inverse_assembly_model: InverseAssemblyModel<'_, F>,
    ) -> Option<F> {
        if self.maybe_beta_estimated_displacement.is_none() {
            let beta_estimated_displacement =
                inverse_assembly_model.beta_estimated_displacement(self);

            self.maybe_beta_estimated_displacement = Some(beta_estimated_displacement);
        }

        self.maybe_beta_estimated_displacement.unwrap()
    }

    pub fn estimated_displacement(
        &mut self,
        inverse_assembly_model: InverseAssemblyModel<'_, F>,
    ) -> F {
        if self.maybe_estimated_displacement.is_none() {
            let estimated_displacement = inverse_assembly_model.estimated_displacement(self);

            self.maybe_estimated_displacement = Some(estimated_displacement);
        }

        self.maybe_estimated_displacement.unwrap()
    }

    pub fn estimated_beta_relative_shaft_angle(
        &mut self,
        inverse_assembly_model: InverseAssemblyModel<'_, F>,
    ) -> &mut ShaftAngle<F> {
        if self.maybe_estimated_beta_relative_shaft_angle.is_none() {
            let estimated_shaft_angle = self.estimated_shaft_angle(inverse_assembly_model);

            let estimated_beta_relative_shaft_angle = inverse_assembly_model
                .assembly()
                .mounted_beta_sensor()
                .inverse()
                .estimated_relative_shaft_angle(*estimated_shaft_angle);

            self.maybe_estimated_beta_relative_shaft_angle =
                Some(estimated_beta_relative_shaft_angle);
        }

        self.maybe_estimated_beta_relative_shaft_angle
            .as_mut()
            .unwrap()
    }

    pub fn estimated_beta_sensor_shaft_angle_offset(&mut self, estimated_beta_sensor_shaft_angle_offset: ShaftAngle<F>) -> &mut ShaftAngle<F> {
        if self.maybe_estimated_beta_sensor_shaft_angle_offset.is_none() {
            self.maybe_estimated_beta_sensor_shaft_angle_offset = Some(estimated_beta_sensor_shaft_angle_offset);
        }

        self.maybe_estimated_beta_sensor_shaft_angle_offset.as_mut().unwrap()
    }
}
