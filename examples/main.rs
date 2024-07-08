use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    element::Rectangle,
    style::{Color, HSLColor, BLACK, WHITE},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::Instant;

fn correlation(n_radial_order: u32, n_xy: usize, zern: &[f64], filename: &str) {
    println!("Computing correlation matrix ...");
    let now = Instant::now();
    let nz = (n_radial_order * (n_radial_order + 1) / 2) as usize;
    let v: Vec<Vec<f64>> = zern.chunks(n_xy * n_xy).map(|x| x.to_vec()).collect();
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y.iter()).fold(0f64, |a, (x, y)| a + x * y);
    let corr: Vec<Vec<u8>> = (0..nz)
        .into_par_iter()
        .map(|i| {
            (0..nz)
                .into_par_iter()
                .map(|j| {
                    if i == j {
                        0
                    } else {
                        let r = dot(&v[i], &v[j])
                            / (dot(&v[i], &v[i]).sqrt() * dot(&v[j], &v[j]).sqrt());
                        if r.abs() > 1e-6 {
                            1
                        } else {
                            0
                        }
                    }
                })
                .collect()
        })
        .collect();
    println!(" ... in {:.3}s", now.elapsed().as_secs_f64());

    let trace_corr = (0..nz).fold(0f64, |a, i| a + dot(&v[i], &v[i]).sqrt());
    println!("Trace[C]: {}", trace_corr);

    let plot = BitMapBackend::new(&filename, (1024, 1024)).into_drawing_area();
    plot.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&plot)
        .build_cartesian_2d(0..nz as i32, 0..nz as i32)
        .unwrap();
    chart
        .draw_series(
            corr.iter()
                .zip(0..)
                .flat_map(|(l, y)| l.iter().zip(0..).map(move |(v, x)| (x, y, v)))
                .map(|(x, y, v)| {
                    Rectangle::new(
                        [(x, y), (x + 1, y + 1)],
                        if *v == 0 {
                            BLACK.filled()
                        } else {
                            WHITE.filled()
                        },
                    )
                }),
        )
        .unwrap();
}

fn main() {
    let n_xy = 401;
    let n_radial_order = 6;
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
    println!("j: {:2?}", j);
    println!("n: {:2?}", n);
    println!("m: {:2?}", m);

    {
        println!("Computing Zernike mode set ...");
        let now = Instant::now();
        let zern = zernike::mode_set(n_radial_order, n_xy);
        println!(" ... in {:.3}s", now.elapsed().as_secs_f64());

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

        correlation(
            n_radial_order,
            n_xy,
            &zern,
            "examples/zernike_correlation.png",
        );
    }

    {
        println!("Computing modified Gram-Schmidt Zernike mode set ...");
        let now = Instant::now();
        let zern = zernike::mgs_mode_set(n_radial_order, n_xy);
        println!(" ... in {:.3}s", now.elapsed().as_secs_f64());
        zern.chunks(n_xy * n_xy).enumerate().for_each(|(k, z)| {
            let filename = format!("examples/mgs_zernike_{}.png", j[k]);
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

        correlation(
            n_radial_order,
            n_xy,
            &zern,
            "examples/mgs_zernike_correlation.png",
        );
    }
}
