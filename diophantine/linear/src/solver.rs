use ndarray::{Array1, Array2, ArrayViewMut2, Axis, concatenate, s};

pub type ZZ = i128;
pub type Vector = Array1<ZZ>;
pub type Matrix = Array2<ZZ>;

fn swap_min(matrix: &mut ArrayViewMut2<ZZ>) {
    let index = matrix.row(0)
        .iter()
        .enumerate()
        .filter(|(_, &a)| a.ne(&0))
        .min_by(|(_, &a), (_, &b)| a.abs().cmp(&b.abs()))
        .map(|(index, _)| index).unwrap();

    for i in 0..matrix.shape()[0] {
        matrix.swap([i, 0], [i, index]);
    }
}

fn matrix_gcd(matrix: &mut ArrayViewMut2<ZZ>) {
    let &q = matrix.get([0, 0]).unwrap();
    for i in 1..matrix.shape()[1] {
        let &k = matrix.get([0, i]).unwrap();
        let d = (k - k % q) / q;
        let t = &matrix.column(i) - d * &matrix.column(0);
        matrix.column_mut(i).assign(&t);
    }
}

fn expand_matrix(matrix: &Matrix) -> Matrix {
    concatenate(Axis(0), &[matrix.view(), Matrix::eye(matrix.shape()[1]).view()]).unwrap()
}

fn convert_matrix(matrix: &mut Matrix) {
    for i in 0..(matrix.shape()[0] - matrix.shape()[1]) {
        let mut sub_matrix = matrix.slice_mut(s![i.., i..]);
        while !sub_matrix.row(0).slice(s![1..]).iter().all(|x| x.eq(&0)) {
            swap_min(&mut sub_matrix);
            matrix_gcd(&mut sub_matrix);
        }
    }
}

fn find_solution(matrix: &Matrix, b: &Vector) -> (Option<Vector>, Vec<Vector>) {
    let mut b_exp = -1 * concatenate(Axis(0), &[b.into(), Array1::zeros(matrix.shape()[1]).view()]).unwrap();
    let mut general: Vec<Vector> = Vec::new();
    let particular_solution;
    let m = matrix.shape()[0] - matrix.shape()[1];
    for i in 0..m {
        let &a = b_exp.get([i]).unwrap();
        let &q = matrix.get([i, i]).unwrap();
        if q == 0 && a != 0 { return (None, general); }
        if q != 0 {
            let k = (a - a % q) / q;
            b_exp.scaled_add(-k, &matrix.column(i));
        }
    }
    if b_exp.slice(s![0..m]).iter().all(|&t| t == 0) {
        let t = b_exp.slice(s![m..]).to_owned();
        if t.iter().all(|&n| n == 0) {
            let mut f = Vector::zeros(t.dim());
            f[0] = 1;
            general.push(f);
        }
        particular_solution = Some(t);
        for i in m..matrix.shape()[1] {
            general.push(matrix.slice(s![m..matrix.shape()[0], i..i+1]).column(0).to_owned());
        }
    }
    else {
        particular_solution = None;
    }
    (particular_solution, general)
}

pub fn solve_system_of_linear_diophantine_equations(mut matrix: Matrix, b: Vector) -> (Option<Vector>, Vec<Vector>) {
    matrix = expand_matrix(&matrix);
    convert_matrix(&mut matrix);
    find_solution(&matrix, &b)
}