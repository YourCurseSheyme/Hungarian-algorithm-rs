# Assignment Solver

A high-performance, zero-allocation Rust library for solving the Linear Assignment Problem using the **Hungarian Algorithm (Kuhn-Munkres)**, enhanced with **LAPJV heuristics**.

## Overview

This crate provides a mathematically rigorous, domain-agnostic solver for bipartite matching problems (e.g., employee rostering, job-machine allocation). It operates strictly on a generic cost matrix and is designed for high-throughput backend environments.

## Key Features

*   **Zero Allocations:** No heap allocations occur within the $O(n^3)$ hot path.
*   **LAPJV Heuristics:** Implements Reduction Transfer for rapid initial matching, bypassing the heavy augmentation phase for up to 95% of rows.
*   **Rectangular Matrix Support:** Natively handles asymmetric assignments (e.g., 100 workers, 10 shifts) via virtual padding.
*   **Dual Objectives:** Supports both `Minimize` and `Maximize` optimization goals.
*   **Cache Optimized:** Utilizes flattened data slices and generation counters for $O(1)$ state resets.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
assignment_solver = { path = "../assignment_solver" }
```

## Quick Start

use assignment_solver::{CostMatrix, Objective, solve_assignment};

```rust
fn main() {
    // 1. Define a cost matrix (e.g., 3 workers, 3 tasks)
    let data = vec![
    8, 4, 7,
    5, 2, 3,
    9, 4, 8
    ];
    let matrix = CostMatrix::new(3, 3, data).unwrap();

    // 2. Solve for minimization
    let assignment = solve_assignment(&matrix, Objective::Minimize).unwrap();

    // 3. Process results
    println!("Total Cost: {}", assignment.total_cost());
    for (worker, task) in assignment.pairs() {
        println!("Worker {} assigned to Task {}", worker, task);
    }
}
```

## Architecture

This library strictly adheres to Data-Oriented Design principles. It does not rely on external dependencies to ensure minimal compile times and maximum portability.