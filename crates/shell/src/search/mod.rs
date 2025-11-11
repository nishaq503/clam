use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use abd_clam::cakes::Search;
use abd_clam::{DistanceValue, cakes};

#[derive(Debug, Clone, PartialEq)]
pub enum ShellQueryAlgorithm {
    String(QueryAlgorithm<u32>),
    F32(QueryAlgorithm<f32>),
    F64(QueryAlgorithm<f64>),
    U8(QueryAlgorithm<u8>),
    U16(QueryAlgorithm<u16>),
    U32(QueryAlgorithm<u32>),
    U64(QueryAlgorithm<u64>),
    I8(QueryAlgorithm<i8>),
    I16(QueryAlgorithm<i16>),
    I32(QueryAlgorithm<i32>),
    I64(QueryAlgorithm<i64>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnnParams {
    pub k: usize,
}

impl fmt::Display for KnnParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "k={}", self.k)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RnnParams<T: DistanceValue> {
    pub radius: T,
}

impl<T: DistanceValue> fmt::Display for RnnParams<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "radius={}", self.radius)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryAlgorithm<T: DistanceValue> {
    KnnLinear(KnnParams),
    KnnRepeatedRnn(KnnParams),
    KnnBreadthFirst(KnnParams),
    KnnDepthFirst(KnnParams),
    RnnLinear(RnnParams<T>),
    RnnClustered(RnnParams<T>),
}

impl<T: DistanceValue> std::fmt::Display for QueryAlgorithm<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            QueryAlgorithm::KnnLinear(params) => write!(f, "knn-linear({params})"),
            QueryAlgorithm::KnnRepeatedRnn(params) => write!(f, "knn-repeated-rnn(k={})", params.k),
            QueryAlgorithm::KnnBreadthFirst(params) => write!(f, "knn-breadth-first({params})"),
            QueryAlgorithm::KnnDepthFirst(params) => write!(f, "knn-depth-first({params})"),
            QueryAlgorithm::RnnLinear(params) => write!(f, "rnn-linear({params})"),
            QueryAlgorithm::RnnClustered(params) => write!(f, "rnn-clustered({params})"),
        }
    }
}

impl<T: DistanceValue + 'static> QueryAlgorithm<T> {
    pub fn get<Id, I, A, M>(&self) -> Box<dyn Search<Id, I, T, A, M>>
    where
        T: DistanceValue + 'static,
        M: Fn(&I, &I) -> T + 'static,
    {
        match self {
            QueryAlgorithm::KnnLinear(params) => Box::new(cakes::KnnLinear(params.k)),
            QueryAlgorithm::KnnRepeatedRnn(params) => Box::new(cakes::KnnRrnn(params.k)),
            QueryAlgorithm::KnnBreadthFirst(params) => Box::new(cakes::KnnBfs(params.k)),
            QueryAlgorithm::KnnDepthFirst(params) => Box::new(cakes::KnnDfs(params.k)),
            QueryAlgorithm::RnnLinear(params) => Box::new(cakes::RnnLinear(T::from(params.radius))),
            QueryAlgorithm::RnnClustered(params) => Box::new(cakes::RnnChess(T::from(params.radius))),
        }
    }
}

/// Parse parameter string into key-value pairs
fn parse_parameters(params_str: &str) -> Result<HashMap<String, String>, String> {
    let mut params = HashMap::new();

    if params_str.is_empty() {
        return Ok(params);
    }

    for pair in params_str.split(',') {
        let pair = pair.trim();
        if let Some((key, value)) = pair.split_once('=') {
            params.insert(key.trim().to_string(), value.trim().to_string());
        } else {
            return Err(format!("Invalid parameter format: '{pair}'. Expected 'key=value'"));
        }
    }
    Ok(params)
}

impl<T: DistanceValue> FromStr for QueryAlgorithm<T> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (algorithm, params_str) = if let Some((alg, params)) = s.split_once(':') {
            (alg, params)
        } else {
            (s, "")
        };
        let params = parse_parameters(params_str)?;

        match algorithm {
            "knn-linear" => Ok(QueryAlgorithm::KnnLinear(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "knn-repeated-rnn" => Ok(QueryAlgorithm::KnnRepeatedRnn(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "knn-breadth-first" => Ok(QueryAlgorithm::KnnBreadthFirst(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "knn-depth-first" => Ok(QueryAlgorithm::KnnDepthFirst(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "rnn-linear" => Ok(QueryAlgorithm::RnnLinear(RnnParams {
                radius: params
                    .get("radius")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid radius value")?,
            })),
            "rnn-clustered" => Ok(QueryAlgorithm::RnnClustered(RnnParams {
                radius: params
                    .get("radius")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid radius value")?,
            })),
            _ => Err(format!("Unknown algorithm: '{algorithm}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::metrics::euclidean;

    use super::*;

    #[test]
    fn test_parse_knn_linear() {
        let query: QueryAlgorithm<f64> = "knn-linear:k=3".parse().unwrap();
        assert_eq!(query, QueryAlgorithm::KnnLinear(KnnParams { k: 3 }));
    }

    #[test]
    fn test_parse_knn_repeated_rnn() {
        let query: QueryAlgorithm<f64> = "knn-repeated-rnn:k=5".parse().unwrap();
        assert_eq!(query, QueryAlgorithm::KnnRepeatedRnn(KnnParams { k: 5 }));
    }

    #[test]
    fn test_parse_rnn() {
        let query: QueryAlgorithm<f64> = "rnn-linear:radius=2.5".parse().unwrap();
        assert_eq!(query, QueryAlgorithm::RnnLinear(RnnParams { radius: 2.5 }));

        let query2: QueryAlgorithm<f64> = "rnn-clustered:radius=1.0".parse().unwrap();
        assert_eq!(query2, QueryAlgorithm::RnnClustered(RnnParams { radius: 1.0 }));
    }

    #[test]
    fn test_display() {
        let query: QueryAlgorithm<f64> = QueryAlgorithm::KnnLinear(KnnParams { k: 3 });
        assert_eq!(query.to_string(), "knn-linear(k=3)");
    }

    #[test]
    fn test_parse_errors() {
        assert!("unknown-algo:k=3".parse::<QueryAlgorithm<f64>>().is_err());
        assert!("knn-linear:k=invalid".parse::<QueryAlgorithm<f64>>().is_err());
        assert!("knn-linear:missing_equals".parse::<QueryAlgorithm<f64>>().is_err());
    }

    #[test]
    fn test_search_algorithm_wrapper_creation() {
        // Test that we can create a SearchAlgorithmWrapper from a QueryAlgorithm
        let query: QueryAlgorithm<f64> = QueryAlgorithm::KnnLinear(KnnParams { k: 5 });
        let alg = query.get::<usize, Vec<f64>, (), euclidean>();

        // Test that the wrapper implements the SearchAlgorithm trait correctly
        assert_eq!(alg.name(), "KnnLinear(k=5)");
    }

    #[test]
    fn test_rnn_wrapper_creation() {
        let query = QueryAlgorithm::RnnLinear(RnnParams { radius: 2.5 });
        let alg = query.get::<usize, Vec<f64>, (), euclidean>();

        assert_eq!(alg.name(), "RnnLinear(radius=2.5)");
    }
}
