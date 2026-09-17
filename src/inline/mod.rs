use motor_calc_core::parameters::Parameters;
use num_traits::Float;

use crate::types::{
    sensor_type::{Alpha, Beta},
    voltage::Voltage,
};

pub fn grad_squared_error_estimated_displacement_inline<F: Float>(
    parameters: Parameters<F>,
    alpha_voltage: Voltage<F, Alpha>,
    beta_voltage: Voltage<F, Beta>,
    displacement: F,
) -> Parameters<F> {
    // PARAMETERS

    let estimated_distance_ratio = parameters.distance_ratio();
    let estimated_beta_shaft_angle_offset = parameters.beta_shaft_angle_offset();
    let estimated_alpha_voltage_scale = parameters.alpha_voltage_scale();
    let estimated_alpha_voltage_offset = parameters.alpha_voltage_offset();
    let estimated_beta_voltage_scale = parameters.beta_voltage_scale();
    let estimated_beta_voltage_offset = parameters.beta_voltage_offset();

    // ESTIMATIONS

    let estimated_alpha_sensor_angle = motor_calc_core::inverse::estimated_sensor_angle(
        motor_calc_core::inverse::estimated_offset_voltage(
            alpha_voltage.volts(),
            estimated_alpha_voltage_offset,
        ),
        estimated_alpha_voltage_scale,
    );

    let estimated_beta_sensor_angle = motor_calc_core::inverse::estimated_sensor_angle(
        motor_calc_core::inverse::estimated_offset_voltage(
            beta_voltage.volts(),
            estimated_beta_voltage_offset,
        ),
        estimated_beta_voltage_scale,
    );

    let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();
    let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();

    let (
        sin_estimated_beta_sensor_shaft_angle_offset,
        cos_estimated_beta_sensor_shaft_angle_offset,
    ) = estimated_beta_shaft_angle_offset.sin_cos();

    let estimated_shaft_angle_numerator = motor_calc_core::inverse::estimated_shaft_angle_numerator(
        tan_estimated_alpha_sensor_angle,
        tan_estimated_beta_sensor_angle,
        sin_estimated_beta_sensor_shaft_angle_offset,
        cos_estimated_beta_sensor_shaft_angle_offset,
    );

    let estimated_shaft_angle_denominator =
        motor_calc_core::inverse::estimated_shaft_angle_denominator(
            tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
        );

    let estimated_shaft_angle = motor_calc_core::inverse::estimated_shaft_angle(
        estimated_shaft_angle_numerator,
        estimated_shaft_angle_denominator,
    );

    let estimated_relative_shaft_angle = motor_calc_core::forward::relative_shaft_angle(
        estimated_shaft_angle,
        estimated_beta_shaft_angle_offset,
    );

    let tan_estimated_alpha_sensor_angle = estimated_alpha_sensor_angle.tan();
    let tan_estimated_beta_sensor_angle = estimated_beta_sensor_angle.tan();
    let (sin_estimated_shaft_angle, cos_estimated_shaft_angle) = estimated_shaft_angle.sin_cos();
    let (sin_estimated_relative_shaft_angle, cos_estimated_relative_shaft_angle) =
        estimated_relative_shaft_angle.sin_cos();

    let estimated_alpha_displacement_numerator =
        motor_calc_core::inverse::estimated_alpha_displacement_numerator(
            tan_estimated_alpha_sensor_angle,
            estimated_distance_ratio,
        );

    let estimated_alpha_displacement_denominator =
        motor_calc_core::inverse::estimated_alpha_displacement_denominator(
            tan_estimated_alpha_sensor_angle,
            sin_estimated_shaft_angle,
            cos_estimated_shaft_angle,
        );

    let estimated_beta_displacement_numerator =
        motor_calc_core::inverse::estimated_beta_displacement_numerator(
            tan_estimated_beta_sensor_angle,
            estimated_distance_ratio,
        );

    let estimated_beta_displacement_denominator =
        motor_calc_core::inverse::estimated_beta_displacement_denominator(
            tan_estimated_beta_sensor_angle,
            sin_estimated_relative_shaft_angle,
            cos_estimated_relative_shaft_angle,
        );

    let alpha_estimated_displacement = motor_calc_core::inverse::estimated_alpha_displacement(
        estimated_alpha_displacement_numerator,
        estimated_alpha_displacement_denominator,
    );

    let maybe_alpha_estimated_displacement = alpha_estimated_displacement
        .is_finite()
        .then_some(alpha_estimated_displacement);

    let beta_estimated_displacement = motor_calc_core::inverse::estimated_beta_displacement(
        estimated_beta_displacement_numerator,
        estimated_beta_displacement_denominator,
    );

    let maybe_beta_estimated_displacement = beta_estimated_displacement
        .is_finite()
        .then_some(beta_estimated_displacement);

    let estimated_displacement = match (
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
    };

    let error_estimated_displacement = estimated_displacement - displacement;

    // GRADIENTS

    let grad_distance_ratio = Parameters::just_distance_ratio();
    let grad_beta_shaft_angle_offset = Parameters::just_beta_shaft_angle_offset();
    let grad_estimated_alpha_voltage_scale = Parameters::just_alpha_voltage_scale();
    let grad_estimated_beta_voltage_scale = Parameters::just_beta_voltage_scale();

    let grad_estimated_offset_alpha_voltage =
        motor_calc_core::gradient::grad_estimated_offset_alpha_voltage(
            Parameters::zero(),
            Parameters::just_alpha_voltage_offset(),
        );

    let grad_estimated_offset_beta_voltage =
        motor_calc_core::gradient::grad_estimated_offset_beta_voltage(
            Parameters::zero(),
            Parameters::just_beta_voltage_offset(),
        );

    let grad_estimated_alpha_sensor_angle =
        motor_calc_core::gradient::grad_estimated_alpha_sensor_angle(
            grad_estimated_offset_alpha_voltage,
            estimated_alpha_voltage_scale,
            alpha_voltage.volts(),
            estimated_alpha_voltage_offset,
            grad_estimated_alpha_voltage_scale,
        );

    let grad_estimated_beta_sensor_angle =
        motor_calc_core::gradient::grad_estimated_beta_sensor_angle(
            grad_estimated_offset_beta_voltage,
            estimated_beta_voltage_scale,
            beta_voltage.volts(),
            estimated_beta_voltage_offset,
            grad_estimated_beta_voltage_scale,
        );

    let grad_tan_estimated_alpha_sensor_angle =
        motor_calc_core::gradient::grad_tan_estimated_alpha_sensor_angle(
            tan_estimated_alpha_sensor_angle,
            grad_estimated_alpha_sensor_angle,
        );

    let grad_tan_estimated_beta_sensor_angle =
        motor_calc_core::gradient::grad_tan_estimated_beta_sensor_angle(
            tan_estimated_beta_sensor_angle,
            grad_estimated_beta_sensor_angle,
        );

    let grad_sin_estimated_beta_shaft_angle_offset =
        motor_calc_core::gradient::grad_sin_beta_shaft_angle_offset(
            cos_estimated_beta_sensor_shaft_angle_offset,
            grad_beta_shaft_angle_offset,
        );

    let grad_cos_estimated_beta_shaft_angle_offset =
        motor_calc_core::gradient::grad_cos_beta_shaft_angle_offset(
            sin_estimated_beta_sensor_shaft_angle_offset,
            grad_beta_shaft_angle_offset,
        );

    let grad_estimated_shaft_angle_numerator =
        motor_calc_core::gradient::grad_estimated_shaft_angle_numerator(
            grad_tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            tan_estimated_alpha_sensor_angle,
            grad_tan_estimated_beta_sensor_angle,
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
            grad_cos_estimated_beta_shaft_angle_offset,
            grad_sin_estimated_beta_shaft_angle_offset,
        );

    let grad_estimated_shaft_angle_denominator =
        motor_calc_core::gradient::grad_estimated_shaft_angle_denominator(
            grad_tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
            tan_estimated_alpha_sensor_angle,
            grad_tan_estimated_beta_sensor_angle,
            grad_sin_estimated_beta_shaft_angle_offset,
            grad_cos_estimated_beta_shaft_angle_offset,
        );

    let grad_estimated_shaft_angle_inner =
        motor_calc_core::gradient::grad_estimated_shaft_angle_inner(
            grad_estimated_shaft_angle_numerator,
            estimated_shaft_angle_denominator,
            estimated_shaft_angle_numerator,
            grad_estimated_shaft_angle_denominator,
        );

    let estimated_shaft_angle_numerator = motor_calc_core::inverse::estimated_shaft_angle_numerator(
        tan_estimated_alpha_sensor_angle,
        tan_estimated_beta_sensor_angle,
        sin_estimated_beta_sensor_shaft_angle_offset,
        cos_estimated_beta_sensor_shaft_angle_offset,
    );

    let estimated_shaft_angle_denominator =
        motor_calc_core::inverse::estimated_shaft_angle_denominator(
            tan_estimated_alpha_sensor_angle,
            tan_estimated_beta_sensor_angle,
            sin_estimated_beta_sensor_shaft_angle_offset,
            cos_estimated_beta_sensor_shaft_angle_offset,
        );

    let estimated_shaft_angle_inner =
        estimated_shaft_angle_numerator / estimated_shaft_angle_denominator;

    let grad_estimated_alpha_sensor_angle =
        motor_calc_core::gradient::grad_estimated_alpha_sensor_angle(
            grad_estimated_offset_alpha_voltage,
            estimated_alpha_voltage_scale,
            alpha_voltage.volts(),
            estimated_alpha_voltage_offset,
            grad_estimated_alpha_voltage_scale,
        );

    let grad_estimated_beta_sensor_angle =
        motor_calc_core::gradient::grad_estimated_beta_sensor_angle(
            grad_estimated_offset_beta_voltage,
            estimated_beta_voltage_scale,
            beta_voltage.volts(),
            estimated_beta_voltage_offset,
            grad_estimated_beta_voltage_scale,
        );

    let grad_estimated_shaft_angle = motor_calc_core::gradient::grad_estimated_shaft_angle(
        grad_estimated_shaft_angle_inner,
        estimated_shaft_angle_inner,
    );

    let grad_estimated_relative_shaft_angle =
        motor_calc_core::gradient::grad_estimated_relative_shaft_angle(
            grad_estimated_shaft_angle,
            grad_beta_shaft_angle_offset,
        );

    let grad_tan_estimated_alpha_sensor_angle =
        motor_calc_core::gradient::grad_tan_estimated_alpha_sensor_angle(
            tan_estimated_alpha_sensor_angle,
            grad_estimated_alpha_sensor_angle,
        );

    let grad_tan_estimated_beta_sensor_angle =
        motor_calc_core::gradient::grad_tan_estimated_beta_sensor_angle(
            tan_estimated_beta_sensor_angle,
            grad_estimated_beta_sensor_angle,
        );

    let grad_sin_estimated_shaft_angle = motor_calc_core::gradient::grad_sin_estimated_shaft_angle(
        cos_estimated_shaft_angle,
        grad_estimated_shaft_angle,
    );

    let grad_cos_estimated_shaft_angle = motor_calc_core::gradient::grad_cos_estimated_shaft_angle(
        sin_estimated_shaft_angle,
        grad_estimated_shaft_angle,
    );

    let grad_sin_estimated_relative_shaft_angle =
        motor_calc_core::gradient::grad_sin_estimated_relative_shaft_angle(
            cos_estimated_relative_shaft_angle,
            grad_estimated_relative_shaft_angle,
        );

    let grad_cos_estimated_relative_shaft_angle =
        motor_calc_core::gradient::grad_cos_estimated_relative_shaft_angle(
            sin_estimated_relative_shaft_angle,
            grad_estimated_relative_shaft_angle,
        );

    let grad_estimated_alpha_displacement_numerator =
        motor_calc_core::gradient::grad_estimated_alpha_displacement_numerator(
            tan_estimated_alpha_sensor_angle,
            grad_distance_ratio,
            estimated_distance_ratio,
            grad_tan_estimated_alpha_sensor_angle,
        );

    let grad_estimated_alpha_displacement_denominator =
        motor_calc_core::gradient::grad_estimated_alpha_displacement_denominator(
            cos_estimated_shaft_angle,
            tan_estimated_alpha_sensor_angle,
            grad_tan_estimated_alpha_sensor_angle,
            grad_cos_estimated_shaft_angle,
            grad_sin_estimated_shaft_angle,
        );

    let grad_estimated_beta_displacement_numerator =
        motor_calc_core::gradient::grad_estimated_beta_displacement_numerator(
            tan_estimated_beta_sensor_angle,
            grad_distance_ratio,
            estimated_distance_ratio,
            grad_tan_estimated_beta_sensor_angle,
        );

    let grad_estimated_beta_displacement_denominator =
        motor_calc_core::gradient::grad_estimated_beta_displacement_denominator(
            cos_estimated_relative_shaft_angle,
            tan_estimated_beta_sensor_angle,
            grad_tan_estimated_beta_sensor_angle,
            grad_cos_estimated_relative_shaft_angle,
            grad_sin_estimated_relative_shaft_angle,
        );

    let grad_estimated_alpha_displacement =
        motor_calc_core::gradient::grad_estimated_alpha_displacement(
            grad_estimated_alpha_displacement_numerator,
            estimated_alpha_displacement_denominator,
            estimated_alpha_displacement_numerator,
            grad_estimated_alpha_displacement_denominator,
        );

    let grad_estimated_beta_displacement =
        motor_calc_core::gradient::grad_estimated_beta_displacement(
            grad_estimated_beta_displacement_numerator,
            estimated_beta_displacement_denominator,
            estimated_beta_displacement_numerator,
            grad_estimated_beta_displacement_denominator,
        );

    let grad_estimated_displacement =
        grad_estimated_alpha_displacement + grad_estimated_beta_displacement;

    let grad_error_estimated_displacement = grad_estimated_displacement;

    grad_error_estimated_displacement
        * F::from(2).expect("2 can always be represented with a float")
        * error_estimated_displacement
}
