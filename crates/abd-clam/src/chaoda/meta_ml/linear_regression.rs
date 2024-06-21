//! Meta-ML model using Linear Regression.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{
    chaoda::{Ratios, Vertex},
    Dataset, Instance, Tree,
};

use super::Model;

/// A Meta-ML model using Linear Regression.
#[derive(Serialize, Deserialize)]
pub struct LinearRegression {
    /// The coefficients of the linear regression model.
    coefficients: Ratios,
    /// The intercept of the linear regression model.
    intercept: f64,
}

impl<'a, U: Number> Model<'a, U, Vertex<U>> for LinearRegression {
    fn train<I: Instance, D: Dataset<I, U>>(tree: &Tree<I, U, D, Vertex<U>>, labels: &[bool]) -> Result<Self, String> {
        if tree.data().cardinality() != labels.len() {
            return Err("The number of labels must be equal to the cardinality of the data.".to_string());
        }

        todo!()
    }

    fn predict(&self, c: &Vertex<U>) -> f64 {
        self.intercept
            + c.ratios()
                .iter()
                .zip(&self.coefficients)
                .map(|(a, b)| a * b)
                .sum::<f64>()
    }
}
