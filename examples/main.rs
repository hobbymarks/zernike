use plotters::prelude::*;
use zernike;

fn main() {
    let n_xy = 501;
    let n_radial_order = 11;
    let (j, n, m) = zernike::jnm(n_radial_order);
    println!("{:2} {:2} {:2}", "n", "m", "j");
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
    (0..j.len()).for_each(|k| println!("{:2} {:2} {:2}", n[k], m[k], j[k]));

    {
        let zern = zernike::mode_set(n_radial_order, n_xy);
        zern.chunks(n_xy * n_xy).enumerate().for_each(|(k, z)| {
            let filename = format!("examples/zernike_{}.png", j[k]);
            let plot =
                BitMapBackend::new(&filename, (n_xy as u32, n_xy as u32)).into_drawing_area();
            plot.fill(&WHITE).unwrap();
            let chart = ChartBuilder::on(&plot)
                .build_cartesian_2d(-1f64..1f64, -1f64..1f64)
                .unwrap();
            let plotting_area = chart.plotting_area();
            let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
            //println!("Min-Max: [{:.3};{:.3}]",z_min,z_max);
            let uz: Vec<f64> = z.iter().map(|p| (p - z_min) / (z_max - z_min)).collect();
            uz.iter().zip(xy.iter()).for_each(|(z, xy)| {
                plotting_area
                    .draw_pixel(*xy, &HSLColor(0.5 * z, 0.5, 0.4))
                    .unwrap();
            })
        });
    }
    {
        let zern = zernike::mgs_mode_set(n_radial_order, n_xy);
        zern.chunks(n_xy * n_xy).enumerate().for_each(|(k, z)| {
            let filename = format!("examples/msg_zernike_{}.png", j[k]);
            let plot =
                BitMapBackend::new(&filename, (n_xy as u32, n_xy as u32)).into_drawing_area();
            plot.fill(&WHITE).unwrap();
            let chart = ChartBuilder::on(&plot)
                .build_cartesian_2d(-1f64..1f64, -1f64..1f64)
                .unwrap();
            let plotting_area = chart.plotting_area();
            let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
            //println!("Min-Max: [{:.3};{:.3}]",z_min,z_max);
            let uz: Vec<f64> = z.iter().map(|p| (p - z_min) / (z_max - z_min)).collect();
            uz.iter().zip(xy.iter()).for_each(|(z, xy)| {
                plotting_area
                    .draw_pixel(*xy, &HSLColor(0.5 * z, 0.5, 0.4))
                    .unwrap();
            })
        });
    }
}
