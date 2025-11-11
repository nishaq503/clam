use core::str::FromStr;
use std::collections::HashMap;

use abd_clam::{
    DistanceValue,
    cakes::{KnnBfs, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear, Search},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellSearchAlgorithm {
    KnnLinear(HashMap<String, String>),
    KnnRepeatedRnn(HashMap<String, String>),
    KnnBreadthFirst(HashMap<String, String>),
    KnnDepthFirst(HashMap<String, String>),
    RnnLinear(HashMap<String, String>),
    RnnChess(HashMap<String, String>),
}

impl core::fmt::Display for ShellSearchAlgorithm {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Self::KnnLinear(params) => write!(f, "KnnLinear({params:?})"),
            Self::KnnRepeatedRnn(params) => write!(f, "KnnRrnn({params:?})"),
            Self::KnnBreadthFirst(params) => write!(f, "KnnBfs({params:?})"),
            Self::KnnDepthFirst(params) => write!(f, "KnnDfs({params:?})"),
            Self::RnnLinear(params) => write!(f, "RnnLinear({params:?})"),
            Self::RnnChess(params) => write!(f, "RnnChess({params:?})"),
        }
    }
}

impl FromStr for ShellSearchAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (alg, params_str) = s.trim().split_once(':').unwrap_or((s, ""));
        let params = parse_parameters(params_str)?;

        match alg.to_lowercase().as_str() {
            "knn-linear" | "knnlinear" => Ok(Self::KnnLinear(params)),
            "knn-repeated-rnn" | "knnrepeatedrnn" | "knn-rrnn" | "knnrrnn" => Ok(Self::KnnRepeatedRnn(params)),
            "knn-breadth-first" | "knnbreadthfirst" | "knn-bfs" | "knnbfs" => Ok(Self::KnnBreadthFirst(params)),
            "knn-depth-first" | "knndepthfirst" | "knn-dfs" | "knndfs" => Ok(Self::KnnDepthFirst(params)),
            "rnn-linear" | "rnnlinear" => Ok(Self::RnnLinear(params)),
            "rnn-chess" | "rnnchess" => Ok(Self::RnnChess(params)),
            _ => Err(format!("Unknown search algorithm: {alg}")),
        }
    }
}

impl ShellSearchAlgorithm {
    pub fn get<Id, I, T, A, M>(&self) -> Result<Box<dyn Search<Id, I, T, A, M>>, String>
    where
        T: DistanceValue + 'static,
        M: Fn(&I, &I) -> T,
    {
        match self {
            Self::KnnLinear(params) => {
                let k = params
                    .get("k")
                    .ok_or("Missing parameter 'k' for KnnLinear")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid value for 'k': {e}"))?;
                Ok(Box::new(KnnLinear(k)))
            }
            Self::KnnRepeatedRnn(params) => {
                let k = params
                    .get("k")
                    .ok_or("Missing parameter 'k' for KnnRepeatedRnn")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid value for 'k': {e}"))?;
                Ok(Box::new(KnnRrnn(k)))
            }
            Self::KnnBreadthFirst(params) => {
                let k = params
                    .get("k")
                    .ok_or("Missing parameter 'k' for KnnBreadthFirst")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid value for 'k': {e}"))?;
                Ok(Box::new(KnnBfs(k)))
            }
            Self::KnnDepthFirst(params) => {
                let k = params
                    .get("k")
                    .ok_or("Missing parameter 'k' for KnnDepthFirst")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid value for 'k': {e}"))?;
                Ok(Box::new(KnnDfs(k)))
            }
            Self::RnnLinear(params) => {
                let radius = params
                    .get("radius")
                    .ok_or("Missing parameter 'radius' for RnnLinear")?
                    .parse::<T>()
                    .map_err(|_| format!("Invalid value for 'radius': {params:?}"))?;
                Ok(Box::new(RnnLinear(radius)))
            }
            Self::RnnChess(params) => {
                let radius = params
                    .get("radius")
                    .ok_or("Missing parameter 'radius' for RnnChess")?
                    .parse::<T>()
                    .map_err(|_| format!("Invalid value for 'radius': {params:?}"))?;
                Ok(Box::new(RnnChess(radius)))
            }
        }
    }
}

/// Parse parameter string into key-value pairs
fn parse_parameters(params_str: &str) -> Result<HashMap<String, String>, String> {
    let mut params = HashMap::new();

    if params_str.is_empty() {
        return Err("No search parameters provided".to_string());
    }

    for pair in params_str.split(',') {
        let pair = pair.trim();
        if let Some((key, value)) = pair.split_once('=') {
            params.insert(key.trim().to_string(), value.trim().to_string());
        } else {
            return Err(format!("Invalid parameter format: '{pair}'. Expected 'key=value'"));
        }
    }

    if params.is_empty() {
        return Err("No valid search parameters found".to_string());
    }

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_knn_linear() {
        let query: ShellSearchAlgorithm = "knn-linear:k=3".parse().unwrap();
        let params = HashMap::from([("k".to_string(), "3".to_string())]);
        assert_eq!(query, ShellSearchAlgorithm::KnnLinear(params));
    }

    #[test]
    fn test_parse_knn_repeated_rnn() {
        let query: ShellSearchAlgorithm = "knn-repeated-rnn:k=5".parse().unwrap();
        let params = HashMap::from([("k".to_string(), "5".to_string())]);
        assert_eq!(query, ShellSearchAlgorithm::KnnRepeatedRnn(params));
    }

    #[test]
    fn test_parse_rnn() {
        let query: ShellSearchAlgorithm = "rnn-linear:radius=2.5".parse().unwrap();
        let params = HashMap::from([("radius".to_string(), "2.5".to_string())]);
        assert_eq!(query, ShellSearchAlgorithm::RnnLinear(params));

        let query2: ShellSearchAlgorithm = "rnn-clustered:radius=1.0".parse().unwrap();
        let params2 = HashMap::from([("radius".to_string(), "1.0".to_string())]);
        assert_eq!(query2, ShellSearchAlgorithm::RnnChess(params2));
    }

    #[test]
    fn test_display() {
        let query = ShellSearchAlgorithm::KnnLinear(HashMap::from([("k".to_string(), "3".to_string())]));
        assert_eq!(query.to_string(), "KnnLinear(k=3)");
    }

    #[test]
    fn test_parse_errors() {
        assert!("unknown-algo:k=3".parse::<ShellSearchAlgorithm>().is_err());
        assert!("knn-linear:k=invalid".parse::<ShellSearchAlgorithm>().is_err());
        assert!("knn-linear:missing_equals".parse::<ShellSearchAlgorithm>().is_err());
    }

    #[test]
    fn test_search_algorithm_wrapper_creation() {
        // Test that we can create a SearchAlgorithmWrapper from a ShellSearchAlgorithm
        let query = ShellSearchAlgorithm::KnnLinear(HashMap::from([("k".to_string(), "5".to_string())]));
        let alg = query.get::<usize, Vec<f64>, f64, (), fn(&Vec<f64>, &Vec<f64>) -> f64>();

        // Test that the wrapper implements the SearchAlgorithm trait correctly
        assert!(alg.is_ok());
        let alg = alg.unwrap();
        assert_eq!(alg.name(), "KnnLinear(k=5)");
    }

    #[test]
    fn test_rnn_wrapper_creation() {
        let query = ShellSearchAlgorithm::RnnLinear(HashMap::from([("radius".to_string(), "2.5".to_string())]));
        let alg = query.get::<usize, Vec<f64>, f64, (), fn(&Vec<f64>, &Vec<f64>) -> f64>();

        assert!(alg.is_ok());
        let alg = alg.unwrap();
        assert_eq!(alg.name(), "RnnLinear(radius=2.5)");
    }
}
