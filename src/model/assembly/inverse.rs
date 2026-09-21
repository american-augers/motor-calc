use num_traits::Float;

use crate::{
    model::{assembly::AssemblyModel, calculation::CalculationContext},
    types::shaft_angle::ShaftAngle,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseAssemblyModel<'i, F: Float> {
    assembly: &'i AssemblyModel<F>,
}

impl<'i, F: Float> InverseAssemblyModel<'i, F> {
    pub fn new(assembly: &'i AssemblyModel<F>) -> Self {
        Self { assembly }
    }

    pub fn assembly(self) -> &'i AssemblyModel<F> {
        self.assembly
    }

    pub fn squared_error_estimated_displacement(self, error_estimated_displacement: F) -> F {
        error_estimated_displacement * error_estimated_displacement
    }

    pub fn error_estimated_displacement(self, estimated_displacement: F, displacement: F) -> F {
        estimated_displacement - displacement
    }

    pub fn estimated_displacement(self, context: &mut CalculationContext<F>) -> F {
        let maybe_alpha_estimated_displacement = context.alpha_estimated_displacement(self);

        let maybe_beta_estimated_displacement = context.beta_estimated_displacement(self);

        match (
            maybe_alpha_estimated_displacement,
            maybe_beta_estimated_displacement,
        ) {
            (Some(alpha_estimated_displacement), Some(beta_estimated_displacement)) => {
                (alpha_estimated_displacement + beta_estimated_displacement)
                    * F::from(0.5).expect("0.5 can always be represented with a float")
            }
            (Some(alpha_estimated_displacement), None) => alpha_estimated_displacement,
            (None, Some(beta_estimated_displacement)) => beta_estimated_displacement,
            (None, None) => unreachable!(
                "unless both sensors are occupying the exact same space, one of their estimated displacements will always be Some"
            ),
        }
    }

    pub fn estimated_shaft_angle(self, context: &mut CalculationContext<F>) -> ShaftAngle<F> {
        ShaftAngle::from_radians_f(
            self.estimated_shaft_angle_numerator(context)
                .atan2(self.estimated_shaft_angle_denominator(context)),
        )
    }

    pub fn estimated_shaft_angle_inner(self, context: &mut CalculationContext<F>) -> F {
        self.estimated_shaft_angle_numerator(context)
            / self.estimated_shaft_angle_denominator(context)
    }

    pub fn estimated_shaft_angle_numerator(self, context: &mut CalculationContext<F>) -> F {
        let (
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
        ) = context
                .estimated_beta_sensor_shaft_angle_offset(
                    self.assembly()
                        .parameters()
                        .mounted_beta_sensor_parameters()
                        .shaft_angle_offset(),
                )
            .sin_cos();

        let estimated_alpha_sensor_angle =
            context.estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse());

        let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();

        let estimated_beta_sensor_angle =
            context.estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse());

        let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();

        motor_calc_core::inverse::estimated_shaft_angle_numerator(
            tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
        )
    }

    pub fn estimated_shaft_angle_denominator(self, context: &mut CalculationContext<F>) -> F {
        let (
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
        ) = context
                .estimated_beta_sensor_shaft_angle_offset(
                    self.assembly()
                        .parameters()
                        .mounted_beta_sensor_parameters()
                        .shaft_angle_offset(),
                )
            .sin_cos();

        let estimated_alpha_sensor_angle =
            context.estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse());

        let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();

        let estimated_beta_sensor_angle =
            context.estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse());

        let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();

        motor_calc_core::inverse::estimated_shaft_angle_denominator(
            tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
        )
    }

    pub fn alpha_estimated_displacement(self, context: &mut CalculationContext<F>) -> Option<F> {
        let estimated_displacement = motor_calc_core::inverse::estimated_alpha_displacement(
            self.alpha_estimated_displacement_numerator(context),
            self.alpha_estimated_displacement_denominator(context),
        );

        estimated_displacement
            .is_finite()
            .then_some(estimated_displacement)
    }

    pub fn alpha_estimated_displacement_numerator(self, context: &mut CalculationContext<F>) -> F {
        motor_calc_core::inverse::estimated_alpha_displacement_numerator(
            context
                .estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse())
                .tan(),
            self.assembly().motor_ref().parameters().distance_ratio(),
        )
    }

    /// Sometimes the denominator is zero. This is bad, because dividing by zero is bad.
    /// Sadly, this is unavoidable, because there are some scenarios where a single sensor
    /// just can't estimate the displacement given its angle and the angle of the shaft.
    ///
    /// More precisely, when the sine of relative shaft angle
    /// (defined as the difference between shaft angle and the sensor's shaft angle offset)
    /// is equal to zero. The only time this really happens is when the shaft is either facing directly towards or away from the sensor.
    pub fn alpha_estimated_displacement_denominator(
        self,
        context: &mut CalculationContext<F>,
    ) -> F {
        let (sin_estimated_shaft_angle, cos_estimated_shaft_angle) =
            context.estimated_shaft_angle(self).sin_cos();

        motor_calc_core::inverse::estimated_alpha_displacement_denominator(
            context
                .estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse())
                .tan(),
            sin_estimated_shaft_angle,
            cos_estimated_shaft_angle,
        )
    }

    pub fn beta_estimated_displacement(self, context: &mut CalculationContext<F>) -> Option<F> {
        let estimated_displacement = motor_calc_core::inverse::estimated_beta_displacement(
            self.beta_estimated_displacement_numerator(context),
            self.beta_estimated_displacement_denominator(context),
        );

        estimated_displacement
            .is_finite()
            .then_some(estimated_displacement)
    }

    pub fn beta_estimated_displacement_numerator(self, context: &mut CalculationContext<F>) -> F {
        motor_calc_core::inverse::estimated_beta_displacement_numerator(
            context
                .estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse())
                .tan(),
            self.assembly().motor_ref().parameters().distance_ratio(),
        )
    }

    /// Sometimes the denominator is zero. This is bad, because dividing by zero is bad.
    /// Sadly, this is unavoidable, because there are some scenarios where a single sensor
    /// just can't estimate the displacement given its angle and the angle of the shaft.
    ///
    /// More precisely, when the sine of relative shaft angle
    /// (defined as the difference between shaft angle and the sensor's shaft angle offset)
    /// is equal to zero. The only time this really happens is when the shaft is either facing directly towards or away from the sensor.
    pub fn beta_estimated_displacement_denominator(self, context: &mut CalculationContext<F>) -> F {
        let (sin_estimated_relative_shaft_angle, cos_estimated_relative_shaft_angle) =
            context.estimated_beta_relative_shaft_angle(self).sin_cos();

        motor_calc_core::inverse::estimated_beta_displacement_denominator(
            context
                .estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse())
                .tan(),
            sin_estimated_relative_shaft_angle,
            cos_estimated_relative_shaft_angle,
        )
    }
}
