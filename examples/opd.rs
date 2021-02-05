use ndarray::Array1;
use ndarray_npy::NpzReader;
use plotters::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut npz = NpzReader::new(
        File::open("examples/OPDData_OPD_Data_5.002000e+02.npz").expect("File not found"),
    )
    .expect("Cannot read npz file");
    println!("names: {:?}", npz.names().unwrap());
    let opd: Array1<f64> = npz.by_name("opd.npy").expect("variable not found");
    let n_xy = 512;

    println!("OPD[0,0]: {}", opd[0]);

    let filename = "examples/opd.png";
    let plot = BitMapBackend::new(&filename, (n_xy as u32, n_xy as u32)).into_drawing_area();
    plot.fill(&WHITE).unwrap();
    let chart = ChartBuilder::on(&plot)
        .build_cartesian_2d(0..n_xy, 0..n_xy)
        .unwrap();
    let plotting_area = chart.plotting_area();
    let z_max = opd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_min = opd.iter().cloned().fold(f64::INFINITY, f64::min);
    //println!("Min-Max: [{:.3};{:.3}]",z_min,z_max);
    let uz: Vec<f64> = opd.iter().map(|p| (p - z_min) / (z_max - z_min)).collect();
    let xy: Vec<_> = (0..n_xy * n_xy)
        .map(|k| {
            let i = k / n_xy;
            let j = k % n_xy;
            (i, j)
        })
        .collect();
    uz.iter().zip(xy.iter()).for_each(|(z, xy)| {
        plotting_area
            .draw_pixel(*xy, &HSLColor(0.5 * z, 0.5, 0.4))
            .unwrap();
    });

    let zern = zernike::mgs_mode_set_on_mask(11, n_xy, opd.as_slice().unwrap());
    zern.chunks(n_xy * n_xy).enumerate().for_each(|(k, z)| {
        let filename = format!("examples/mask_zernike_{}.png", k);
        let plot = BitMapBackend::new(&filename, (n_xy as u32, n_xy as u32)).into_drawing_area();
        plot.fill(&WHITE).unwrap();
        let chart = ChartBuilder::on(&plot)
            .build_cartesian_2d(0..n_xy, 0..n_xy)
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

    let opd = opd.as_slice().unwrap();
    let c = zernike::projection_on_mask(opd, 11, n_xy, opd);
    let n = n_xy * n_xy;
    let opd_e: Vec<_> = zern.chunks(n).zip(c).fold(vec![0f64; n], |mut a, (x, c)| {
        a.iter_mut().zip(x).for_each(|(a, x)| {
            *a += c * x;
        });
        a
    });
    let filename = "examples/opd_e.png";
    let plot = BitMapBackend::new(&filename, (n_xy as u32, n_xy as u32)).into_drawing_area();
    plot.fill(&WHITE).unwrap();
    let chart = ChartBuilder::on(&plot)
        .build_cartesian_2d(0..n_xy, 0..n_xy)
        .unwrap();
    let plotting_area = chart.plotting_area();
    let z_max = opd_e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_min = opd_e.iter().cloned().fold(f64::INFINITY, f64::min);
    //println!("Min-Max: [{:.3};{:.3}]",z_min,z_max);
    let uz: Vec<f64> = opd_e.iter().map(|p| (p - z_min) / (z_max - z_min)).collect();
    let xy: Vec<_> = (0..n_xy * n_xy)
        .map(|k| {
            let i = k / n_xy;
            let j = k % n_xy;
            (i, j)
        })
        .collect();
    uz.iter().zip(xy.iter()).for_each(|(z, xy)| {
        plotting_area
            .draw_pixel(*xy, &HSLColor(0.5 * z, 0.5, 0.4))
            .unwrap();
    });

    Ok(())
}
