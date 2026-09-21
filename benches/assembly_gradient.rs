use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use motor_calc::{
    model::{assembly::AssemblyModel, calculation::CalculationContext},
    types::voltage::Voltage,
};

fn bench_grad_squared_error_estimated_displacement(c: &mut Criterion) {
    let params = (2.5, 2.5);

    let assembly = AssemblyModel::<f64>::default();
    let gradient = assembly.gradient();

    c.bench_with_input(
        BenchmarkId::new(
            "grad_squared_error_estimated_displacement",
            format!("{}_{}", params.0, params.1),
        ),
        &params,
        |b, &(raw_alpha_voltage, raw_beta_voltage)| {
            let mut context = CalculationContext::new(
                Voltage::from_volts_f(raw_alpha_voltage),
                Voltage::from_volts_f(raw_beta_voltage),
            );

            b.iter(|| gradient.grad_squared_error_estimated_displacement(&mut context, 1.0));
        },
    );
}

criterion_group!(benches, bench_grad_squared_error_estimated_displacement,);
criterion_main!(benches);
