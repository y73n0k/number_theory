mod solver;

use std::io;
use std::io::Write;

use ndarray::Axis;

fn input_matrix() -> (solver::Matrix, solver::Vector) {
    let mut line = String::new();

    io::stdout().write(b"n: ").unwrap();
    io::stdout().flush().unwrap();
    line.clear();
    io::stdin().read_line(&mut line).unwrap();
    let n: usize = line.trim().parse().unwrap();

    io::stdout().write(b"m: ").unwrap();
    io::stdout().flush().unwrap();
    line.clear();
    io::stdin().read_line(&mut line).unwrap();
    let m: usize = line.trim().parse().unwrap();

    io::stdout().write(b"matrix: \n").unwrap();
    io::stdout().flush().unwrap();

    let arr: Vec<Vec<solver::ZZ>> = io::stdin().lines()
        .take(n)
        .map(|l| l.unwrap().split(char::is_whitespace)
            .take(m)
            .map(|num| num.parse().unwrap())
            .collect())
        .collect();

    let mut matrix = solver::Matrix::default((n, m));
    for (i, mut row) in matrix.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = arr[i][j];
        }
    }

    io::stdout().write(b"b vector: ").unwrap();
    io::stdout().flush().unwrap();
    line.clear();
    io::stdin().read_line(&mut line).unwrap();
    let b: solver::Vector = line.trim().split(char::is_whitespace)
        .map(|num| num.parse().unwrap())
        .collect();

    (matrix, b)
}

fn main() {
    let (matrix, b) = input_matrix();
    let (part, general) = solver::solve_system_of_linear_diophantine_equations(matrix.clone(), b.clone());
    if part.is_some() {
        let sol = part.unwrap();
        print!("{}", sol);
        for (i, x) in general.iter().enumerate() {
            print!(" + t{} * {}", i + 1, x)
        }
        println!();

        println!("{}", matrix.dot(&sol).eq(&b));
        println!("{}", matrix.dot(&(sol + 2 * general.get(0).unwrap())).eq(&b));
    }
    else {
        println!("no solution");
    }
}
