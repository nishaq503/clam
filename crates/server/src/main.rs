#[macro_use]
extern crate rocket;
use abd_clam::{
    cakes::{cluster::ParSearchable, Algorithm}, Metric,
};
use rocket::serde::json::Json;

mod config;
use config::{CodecDataType, Config, SquishyBallType};

#[get("/")]
fn index() -> &'static str {
    "Welcome to the CLAM server."
}

#[get("/search/rnn-clustered?<query>&<radius>")]
fn search_rnn_clustered(
    query: String,
    radius: u16,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let alg = Algorithm::RnnClustered(radius);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(codec_data, &query, alg);
    Json(results)
}

#[get("/search/knn-repeated-rnn?<query>&<k>&<max_radius_multiplier>")]
fn search_knn_repeated_rnn(
    query: String,
    k: usize,
    max_radius_multiplier: Option<u16>,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let r_m = max_radius_multiplier.unwrap_or(2);
    let alg = Algorithm::KnnRepeatedRnn(k, r_m);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(codec_data, &query, alg);
    Json(results)
}

#[get("/search/knn-breadth-first?<query>&<k>")]
fn search_knn_breadth_first(
    query: String,
    k: usize,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let alg = Algorithm::KnnBreadthFirst(k);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(codec_data, &query, alg);
    Json(results)
}

#[get("/search/knn-depth-first?<query>&<k>")]
fn search_knn_depth_first(
    query: String,
    k: usize,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let alg = Algorithm::KnnDepthFirst(k);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(codec_data, &query, alg);
    Json(results)
}

#[launch]
fn rocket() -> _ {
    let config = match std::env::var("CLAM_BOOTSTRAP") {
        Ok(_) => Config::from_bootstrap(),
        Err(_) => Config::from_env(),
    };
    let (squishy_ball, codec_data): (SquishyBallType, CodecDataType) = config.load(Metric::new(
        |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b),
        true,
    ));
    rocket::build().manage(squishy_ball).manage(codec_data).mount(
        "/",
        routes![
            index,
            search_rnn_clustered,
            search_knn_repeated_rnn,
            search_knn_breadth_first,
            search_knn_depth_first,
        ],
    )
}
