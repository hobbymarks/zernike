//! # Zernike polynomials

//! Computes the Zernike polynomials according to the following ordering:
//!
//! | n | m | j |
//! |---|---|---|
//! | 0 | 0 | 1 |
//! | 1 | 1 | 2 |
//! | 1 | 1 | 3 |
//! | 2 | 0 | 4 |
//! | 2 | 2 | 5 |
//! | 2 | 2 | 6 |
//! | 3 | 1 | 7 |
//! | 3 | 1 | 8 |
//! | 3 | 3 | 9 |
//! | 3 | 3 |10 |
//! | 4 | 0 |11 |
//! | 4 | 2 |12 |
//! | 4 | 2 |13 |
//! | 4 | 4 |14 |
//! | 4 | 4 |15 |
//! | 5 | 1 |16 |
//! | 5 | 1 |17 |
//! | 5 | 3 |18 |
//! | 5 | 3 |19 |
//! | 5 | 5 |20 |
//! | 5 | 5 |21 |
//!
//! where n, m, and j are the radial order, the azimuthal order and the polynomial index, respectively

use rayon::prelude::*;

const PI: f64 = std::f64::consts::PI;
const E: f64 = std::f64::consts::E;

fn ln_gamma(f64: f64) -> f64 {
    // Auxiliary variable when evaluating the `gamma_ln` function
    let gamma_r: f64 = 10.900511;

    // Polynomial coefficients for approximating the `gamma_ln` function
    let gamma_dk: &[f64] = &[
        2.48574089138753565546e-5,
        1.05142378581721974210,
        -3.45687097222016235469,
        4.51227709466894823700,
        -2.98285225323576655721,
        1.05639711577126713077,
        -1.95428773191645869583e-1,
        1.70970543404441224307e-2,
        -5.71926117404305781283e-4,
        4.63399473359905636708e-6,
        -2.71994908488607703910e-9,
    ];

    let x: f64 = f64;

    if x < 0.5 {
        let s = gamma_dk
            .iter()
            .enumerate()
            .skip(1)
            .fold(gamma_dk[0], |s, t| s + *t.1 / ((t.0 as u64) as f64 - x));

        PI.ln()
            - (PI * x).sin().ln()
            - s.ln()
            - (2.0 * (E / PI).powf(0.5)).ln()
            - (0.5 - x) * ((0.5 - x + gamma_r) / E).ln()
    } else {
        let s = gamma_dk
            .iter()
            .enumerate()
            .skip(1)
            .fold(gamma_dk[0], |s, t| {
                s + *t.1 / (x + (t.0 as u64) as f64 - 1.0)
            });

        s.ln()
            + (2.0 * (E / PI).powf(0.5)).ln()
            + (x - 0.5) * ((x - 0.5 + gamma_r) / std::f64::consts::E).ln()
    }
}
fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}
/// Zernike radial function
fn radial(n: u32, m: u32, r: f64) -> f64 {
    (0..(n - m) / 2 + 1).fold(0f64, |a, k| {
        a + (-1f64).powf(k as f64) * gamma((n - k + 1) as f64) * r.powf((n - 2 * k) as f64)
            / (gamma((k + 1) as f64)
                * gamma(((n + m) / 2 - k + 1) as f64)
                * gamma(((n - m) / 2 - k + 1) as f64))
    })
}
/// Zernike azimuthal function
fn azimuthal(j: u32, m: u32, o: f64) -> f64 {
    let nkr = if m == 0 { 0f64 } else { 1f64 };
    2f64.powf(0.5 * nkr) * (m as f64 * o + nkr * ((-1f64).powf(j as f64) - 1f64) * PI * 0.25).cos()
}
/// Returns the Zernike polynomial (j,n,m) value at the polar coordinates (o,r)
pub fn zernike(j: u32, n: u32, m: u32, r: f64, o: f64) -> f64 {
    radial(n, m, r) * azimuthal(j, m, o)
}
/// Zernike mode on a regular grid n_xy X n_xy
pub fn mode(zj: u32, n: u32, m: u32, n_xy: usize) -> Vec<f64> {
    let d = 2f64 / (n_xy - 1) as f64;
    let h = ((n_xy - 1) / 2) as f64;
    (0..n_xy * n_xy)
        .into_par_iter()
        .map(|k| {
            let i = (k / n_xy) as f64 - h;
            let j = (k % n_xy) as f64 - h;
            let x = i * d;
            let y = j * d;
            let r = x.hypot(y);
            if r > 1f64 {
                0f64
            } else {
                let o = y.atan2(x);
                zernike(zj, n, m, r, o)
            }
        })
        .collect()
}
/// A complete set of `n_radial_order` Zernike modes on a regular grid n_xy X n_xy
pub fn mode_set(n_radial_order: u32, n_xy: usize) -> Vec<f64> {
    let (j, n, m) = jnm(n_radial_order);
    j.into_par_iter()
        .zip(n.into_par_iter().zip(m.into_par_iter()))
        .flat_map(|(j, (n, m))| mode(j, n, m, n_xy))
        .collect()
}
/// A complete set of `n_radial_order` orthonormalized Zernike modes on a regular grid n_xy X n_xy using the modified Gram-Schmidt algorithm
pub fn mgs_mode_set(n_radial_order: u32, n_xy: usize) -> Vec<f64> {
    let n = n_xy * n_xy;
    let nz = n_radial_order * (n_radial_order + 1) / 2;
    // v: Zernike modes
    let v: Vec<Vec<f64>> = mode_set(n_radial_order, n_xy)
        .chunks(n)
        .map(|x| x.to_vec())
        .collect();
    // u: orthonormal basis
    let mut u: Vec<Vec<f64>> = vec![0f64; n * nz as usize]
        .chunks(n)
        .map(|x| x.to_vec())
        .collect();
    // Returns the dot product: x.y
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y.iter()).fold(0f64, |a, (x, y)| a + x * y);
    // v1.v1
    let nrm = dot(&v[0], &v[0]).sqrt();
    // u1 = v1/v1.v1
    u[0].iter_mut().zip(v[0].iter()).for_each(|(u, v)| {
        *u = v / nrm;
    });
    (1..nz as usize).for_each(|i| {
        // ui = vi
        u[i].iter_mut().zip(v[i].iter()).for_each(|(u, v)| {
            *u = *v;
        });
        (0..i).for_each(|j| {
            // uj.ui/uj.uj
            let r = dot(&u[j], &u[i]) / dot(&u[j], &u[j]);
            // ui = ui - (uj.ui/uj.uj)vj
            let uj = u[j].clone();
            u[i].iter_mut().zip(uj.iter()).for_each(|(ui, uj)| {
                *ui -= r * *uj;
            });
        });
        // ui.ui
        let nrm = dot(&u[i], &u[i]).sqrt();
        // ui = ui/ui.ui
        u[i].iter_mut().for_each(|u| {
            *u /= nrm;
        });
    });
    u.iter().flat_map(|x| x.to_vec()).collect()
}
/// Returns the coefficients resulting of the projection of a surface on a complete set of `n_radial_order` orthonormalized Zernike modes defined on a regular grid n_xy X n_xy using the modified Gram-Schmidt algorithm
pub fn projection(surface: &[f64], n_radial_order: u32, n_xy: usize) -> Vec<f64> {
    let n = n_xy * n_xy;
    let nz = n_radial_order * (n_radial_order + 1) / 2;
    // v: Zernike modes
    let v: Vec<Vec<f64>> = mode_set(n_radial_order, n_xy)
        .chunks(n)
        .map(|x| x.to_vec())
        .collect();
    // u: orthonormal basis
    let mut u: Vec<Vec<f64>> = vec![0f64; n * nz as usize]
        .chunks(n)
        .map(|x| x.to_vec())
        .collect();
    // Returns the dot product: x.y
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y.iter()).fold(0f64, |a, (x, y)| a + x * y);
    // v1.v1
    let nrm = dot(&v[0], &v[0]).sqrt();
    // u1 = v1/v1.v1
    u[0].iter_mut().zip(v[0].iter()).for_each(|(u, v)| {
        *u = v / nrm;
    });
    let mut c = vec![dot(surface, &u[0])];
    c.extend((1..nz as usize).map(|i| {
        // ui = vi
        u[i].iter_mut().zip(v[i].iter()).for_each(|(u, v)| {
            *u = *v;
        });
        (0..i).for_each(|j| {
            // uj.ui/uj.uj
            let r = dot(&u[j], &u[i]) / dot(&u[j], &u[j]);
            // ui = ui - (uj.ui/uj.uj)vj
            let uj = u[j].clone();
            u[i].iter_mut().zip(uj.iter()).for_each(|(ui, uj)| {
                *ui -= r * *uj;
            });
        });
        // ui.ui
        let nrm = dot(&u[i], &u[i]).sqrt();
        dot(surface, &u[i])/nrm
    }));
    c
}
/// Returns the Zernike indices `(j,n,m)` for the first `n_radial_order`s
pub fn jnm(n_radial_order: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut j: Vec<u32> = vec![];
    let mut n: Vec<u32> = vec![];
    let mut m: Vec<u32> = vec![];
    (1..=n_radial_order).into_iter().for_each(|nro| {
        (0..nro).step_by(2).for_each(|k| {
            let odd_even = (nro - 1) % 2;
            let j_last = 1 + j.last().or(Some(&0u32)).unwrap();
            if k == 0 && odd_even == 0 {
                j.push(j_last);
                n.push(nro - 1);
                m.push(0);
            } else {
                j.push(j_last);
                j.push(1 + j_last);
                n.push(nro - 1);
                n.push(nro - 1);
                m.push(odd_even + k);
                m.push(odd_even + k);
            }
        });
    });
    (j, n, m)
}
