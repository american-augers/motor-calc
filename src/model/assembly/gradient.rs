use motor_calc_core::parameters::Parameters;
use num_traits::Float;

use crate::model::{assembly::AssemblyModel, calculation::CalculationContext};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientAssemblyModel<'g, F: Float> {
    assembly: &'g AssemblyModel<F>,
}

impl<'g, F: Float> GradientAssemblyModel<'g, F> {
    pub fn new(assembly: &'g AssemblyModel<F>) -> Self {
        Self { assembly }
    }

    pub fn assembly(self) -> &'g AssemblyModel<F> {
        self.assembly
    }

    pub fn grad_squared_error_estimated_displacement(
        self,
        context: &mut CalculationContext<F>,
        displacement: F,
    ) -> Parameters<F> {
        let estimated_displacement = context.estimated_displacement(self.assembly().inverse());

        self.grad_error_estimated_displacement(context)
            * F::from(2).expect("2 can always be represented with a float")
            * self
                .assembly()
                .inverse()
                .error_estimated_displacement(estimated_displacement, displacement)
    }

    pub fn grad_error_estimated_displacement(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        self.grad_estimated_displacement(context)
    }

    pub fn grad_estimated_displacement(self, context: &mut CalculationContext<F>) -> Parameters<F> {
        self.grad_estimated_alpha_displacement(context)
            + self.grad_estimated_beta_displacement(context)
    }

    pub fn grad_estimated_alpha_displacement(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let grad_estimated_alpha_displacement_numerator =
            self.grad_estimated_alpha_displacement_numerator(context);

        let grad_estimated_alpha_displacement_denominator =
            self.grad_estimated_alpha_displacement_denominator(context);

        let estimated_alpha_displacement_denominator = self
            .assembly()
            .inverse()
            .alpha_estimated_displacement_denominator(context);

        let estimated_alpha_displacement_numerator = self
            .assembly()
            .inverse()
            .alpha_estimated_displacement_numerator(context);

        motor_calc_core::gradient::grad_estimated_alpha_displacement(
            grad_estimated_alpha_displacement_numerator,
            estimated_alpha_displacement_denominator,
            estimated_alpha_displacement_numerator,
            grad_estimated_alpha_displacement_denominator,
        )
    }

    pub fn grad_estimated_alpha_displacement_numerator(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let estimated_alpha_sensor_angle =
            context.estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse());

        motor_calc_core::gradient::grad_estimated_alpha_displacement_numerator(
            estimated_alpha_sensor_angle.tan(),
            Parameters::just_distance_ratio(),
            self.assembly().motor().parameters().distance_ratio(),
            self.assembly()
                .alpha_sensor_ref()
                .gradient()
                .grad_tan_estimated_sensor_angle(context),
        )
    }

    pub fn grad_estimated_alpha_displacement_denominator(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let estimated_alpha_sensor_angle =
            context.estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse());

        let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();

        motor_calc_core::gradient::grad_estimated_alpha_displacement_denominator(
            context
                .estimated_shaft_angle(self.assembly().inverse())
                .cos(),
            tan_estimated_alpha_sensor_angle,
            self.assembly()
                .alpha_sensor_ref()
                .gradient()
                .grad_tan_estimated_sensor_angle(context),
            self.grad_cos_estimated_shaft_angle(context),
            self.grad_sin_estimated_shaft_angle(context),
        )
    }

    pub fn grad_estimated_beta_displacement(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let grad_estimated_beta_displacement_numerator =
            self.grad_estimated_beta_displacement_numerator(context);

        let grad_estimated_beta_displacement_denominator =
            self.grad_estimated_beta_displacement_denominator(context);

        let estimated_beta_displacement_denominator = self
            .assembly()
            .inverse()
            .beta_estimated_displacement_denominator(context);

        let estimated_beta_displacement_numerator = self
            .assembly()
            .inverse()
            .beta_estimated_displacement_numerator(context);

        motor_calc_core::gradient::grad_estimated_beta_displacement(
            grad_estimated_beta_displacement_numerator,
            estimated_beta_displacement_denominator,
            estimated_beta_displacement_numerator,
            grad_estimated_beta_displacement_denominator,
        )
    }

    fn grad_estimated_beta_displacement_numerator(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let estimated_beta_sensor_angle =
            context.estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse());

        motor_calc_core::gradient::grad_estimated_beta_displacement_numerator(
            estimated_beta_sensor_angle.tan(),
            Parameters::just_distance_ratio(),
            self.assembly().motor().parameters().distance_ratio(),
            self.assembly()
                .beta_sensor_ref()
                .gradient()
                .grad_tan_estimated_sensor_angle(context),
        )
    }

    fn grad_estimated_beta_displacement_denominator(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let grad_estimated_shaft_angle = self.grad_estimated_shaft_angle(context);

        let estimated_beta_sensor_angle =
            context.estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse());

        let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();

        let estimated_beta_relative_shaft_angle =
            context.estimated_beta_relative_shaft_angle(self.assembly().inverse());

        let cos_estimated_beta_relative_shaft_angle = estimated_beta_relative_shaft_angle.cos();

        let estimated_shaft_angle = *context.estimated_shaft_angle(self.assembly().inverse());

        let (
            grad_sin_estimated_beta_relative_shaft_angle,
            grad_cos_estimated_beta_relative_shaft_angle,
        ) = self
            .assembly()
            .mounted_beta_sensor()
            .gradient()
            .grad_sin_cos_estimated_relative_shaft_angle(
                estimated_shaft_angle,
                grad_estimated_shaft_angle,
            );

        motor_calc_core::gradient::grad_estimated_beta_displacement_denominator(
            cos_estimated_beta_relative_shaft_angle,
            tan_estimated_beta_sensor_angle,
            self.assembly()
                .beta_sensor_ref()
                .gradient()
                .grad_tan_estimated_sensor_angle(context),
            grad_cos_estimated_beta_relative_shaft_angle,
            grad_sin_estimated_beta_relative_shaft_angle,
        )
    }

    pub fn grad_cos_estimated_shaft_angle(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_cos_estimated_shaft_angle(
            context
                .estimated_shaft_angle(self.assembly().inverse())
                .sin(),
            self.grad_estimated_shaft_angle(context),
        )
    }

    pub fn grad_sin_estimated_shaft_angle(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        motor_calc_core::gradient::grad_sin_estimated_shaft_angle(
            context
                .estimated_shaft_angle(self.assembly().inverse())
                .cos(),
            self.grad_estimated_shaft_angle(context),
        )
    }

    pub fn grad_estimated_shaft_angle(self, context: &mut CalculationContext<F>) -> Parameters<F> {
        let grad_estimated_shaft_angle_inner = self.grad_estimated_shaft_angle_inner(context);

        motor_calc_core::gradient::grad_estimated_shaft_angle(
            grad_estimated_shaft_angle_inner,
            self.assembly()
                .inverse()
                .estimated_shaft_angle_inner(context),
        )
    }

    pub fn grad_estimated_shaft_angle_inner(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let grad_estimated_shaft_angle_numerator =
            self.grad_estimated_shaft_angle_numerator(context);
        let grad_estimated_shaft_angle_denominator =
            self.grad_estimated_shaft_angle_denominator(context);

        motor_calc_core::gradient::grad_estimated_shaft_angle_inner(
            grad_estimated_shaft_angle_numerator,
            self.assembly()
                .inverse()
                .estimated_shaft_angle_denominator(context),
            self.assembly()
                .inverse()
                .estimated_shaft_angle_numerator(context),
            grad_estimated_shaft_angle_denominator,
        )
    }

    pub fn grad_estimated_shaft_angle_numerator(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let grad_tan_estimated_alpha_sensor_angle = self
            .assembly()
            .alpha_sensor_ref()
            .gradient()
            .grad_tan_estimated_sensor_angle(context);

        let grad_tan_estimated_beta_sensor_angle = self
            .assembly()
            .beta_sensor_ref()
            .gradient()
            .grad_tan_estimated_sensor_angle(context);

        let estimated_alpha_sensor_angle =
            context.estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse());

        let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();

        let estimated_beta_sensor_angle =
            context.estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse());

        let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();

        let (sin_estimated_beta_shaft_angle_offset, cos_estimated_beta_shaft_angle_offset) =
            context
                .estimated_beta_sensor_shaft_angle_offset(
                    self.assembly()
                        .parameters()
                        .mounted_beta_sensor_parameters()
                        .shaft_angle_offset(),
                )
                .sin_cos();

        let (
            grad_sin_estimated_beta_shaft_angle_offset,
            grad_cos_estimated_beta_shaft_angle_offset,
        ) = self
            .assembly()
            .mounted_beta_sensor()
            .gradient()
            .grad_sin_cos_estimated_shaft_angle_offset();

        motor_calc_core::gradient::grad_estimated_shaft_angle_numerator(
            grad_tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            tan_estimated_alpha_sensor_angle,
            grad_tan_estimated_beta_sensor_angle,
            sin_estimated_beta_shaft_angle_offset,
            cos_estimated_beta_shaft_angle_offset,
            grad_cos_estimated_beta_shaft_angle_offset,
            grad_sin_estimated_beta_shaft_angle_offset,
        )
    }

    pub fn grad_estimated_shaft_angle_denominator(
        self,
        context: &mut CalculationContext<F>,
    ) -> Parameters<F> {
        let grad_tan_estimated_alpha_sensor_angle = self
            .assembly()
            .alpha_sensor_ref()
            .gradient()
            .grad_tan_estimated_sensor_angle(context);

        let grad_tan_estimated_beta_sensor_angle = self
            .assembly()
            .beta_sensor_ref()
            .gradient()
            .grad_tan_estimated_sensor_angle(context);

        let estimated_alpha_sensor_angle =
            context.estimated_alpha_sensor_angle(self.assembly().alpha_sensor_ref().inverse());

        let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();

        let estimated_beta_sensor_angle =
            context.estimated_beta_sensor_angle(self.assembly().beta_sensor_ref().inverse());

        let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();

        let (sin_estimated_beta_shaft_angle_offset, cos_estimated_beta_shaft_angle_offset) =
            context
                .estimated_beta_sensor_shaft_angle_offset(
                    self.assembly()
                        .parameters()
                        .mounted_beta_sensor_parameters()
                        .shaft_angle_offset(),
                )
                .sin_cos();

        let (
            grad_sin_estimated_beta_shaft_angle_offset,
            grad_cos_estimated_beta_shaft_angle_offset,
        ) = self
            .assembly()
            .mounted_beta_sensor()
            .gradient()
            .grad_sin_cos_estimated_shaft_angle_offset();

        motor_calc_core::gradient::grad_estimated_shaft_angle_denominator(
            grad_tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            sin_estimated_beta_shaft_angle_offset,
            cos_estimated_beta_shaft_angle_offset,
            tan_estimated_alpha_sensor_angle,
            grad_tan_estimated_beta_sensor_angle,
            grad_sin_estimated_beta_shaft_angle_offset,
            grad_cos_estimated_beta_shaft_angle_offset,
        )
    }
}
