use zernike;
use rand::prelude::*;
use plotters::prelude::*;

fn main() {
    let n_xy = 601;
    let n_radial_order = 5;
    let zern = zernike::mgs_mode_set(n_radial_order, n_xy);
    //let c: Vec<f64> = (0..n_radial_order*(n_radial_order-1)/2).map(|x| random()).collect();
    let n = n_xy*n_xy;
    let surface: Vec<_> = zern.chunks(n).fold(vec![0f64;n], |mut a,x| {
        let c: f64 = random();
        println!("c: {}",c);
        a.iter_mut().zip(x).for_each(|(a,x)| {
            *a += c*x;
        });
        a
    });
    let filename = "examples/surface.png";
    let plot =
        BitMapBackend::new(&filename, (512,512)).into_drawing_area();
    plot.fill(&WHITE).unwrap();
    let chart = ChartBuilder::on(&plot)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64)
        .unwrap();
    let plotting_area = chart.plotting_area();
    let z_max = surface.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_min = surface.iter().cloned().fold(f64::INFINITY, f64::min);
    //println!("Min-Max: [{:.3};{:.3}]",z_min,z_max);
    let uz: Vec<f64> = surface.iter().map(|p| (p - z_min) / (z_max - z_min)).collect();
    let d = 2f64 / (n_xy - 1) as f64;
    let h = ((n_xy - 1) / 2) as f64;
    let xy: Vec<_> = (0..n_xy * n_xy)
        .map(|k| {
            let i = (k / n_xy) as f64 - h;
            let j = (k % n_xy) as f64 - h;
            let x = i * d;
            let y = j * d;
            (x, y)
        })
        .collect();
    uz.iter().zip(xy.iter()).for_each(|(z, xy)| {
        plotting_area
            .draw_pixel(*xy, &HSLColor(0.5 * z, 0.5, 0.4))
            .unwrap();
    });

    let c_e = zernike::projection(&surface, n_radial_order, n_xy);
    println!("<c>: {:?}",c_e)
}
