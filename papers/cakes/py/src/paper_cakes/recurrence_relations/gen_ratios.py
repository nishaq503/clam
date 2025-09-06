"""Generating memos and ratios for recurrence relations."""

import pathlib
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import numpy
import pandas
import typer
from tqdm import tqdm


def memo_for_k_n(k: int, n_max: int) -> numpy.ndarray:
    """Compute memo for a single k and multiple ns."""
    memo = numpy.zeros(n_max + 1, dtype=numpy.float32)
    memo[0] = 1.0
    memo[1] = 1.0
    for n in range(2, k + 1):
        memo[n] = n - 1.0
    for kn_i_1 in range(k + 1, n_max + 1):
        q = (kn_i_1 - 1) // k
        r = (kn_i_1 - 1) % k
        memo[kn_i_1] = 1.0 + r * memo[q + 1] + (k - r) * memo[q]
    return memo


def memo_for_ks_n(ks: list[int], n_max: int) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Compute memos and ratios for multiple ks and ns in parallel."""
    with ProcessPoolExecutor() as executor:
        futures: list[Future[numpy.ndarray]] = [
            executor.submit(memo_for_k_n, k, n_max) for k in ks
        ]

    memos = numpy.zeros((n_max + 1, len(ks)), dtype=numpy.float32)
    for i, future in tqdm(enumerate(as_completed(futures)), total=len(futures)):
        memos[:, i] = future.result()
    typer.echo("Computed all memos")

    memos = memos[1:]  # Remove n=0
    ns = numpy.arange(1, n_max + 1)
    ratios = (memos / ns[:, None]).astype(numpy.float32)

    memos_df = pandas.DataFrame(memos, index=ns, columns=list(map(str, ks)))
    ratios_df = pandas.DataFrame(ratios, index=ns, columns=list(map(str, ks)))

    return memos_df, ratios_df


def gen_ratios(
    data_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-d",
        "--data-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory to save memos and ratios parquet files",
    ),
    max_n: int = typer.Option(
        100_000,
        "-n",
        "--max-n",
        help="Maximum n to compute memos and ratios for",
    ),
    max_k: int = typer.Option(
        16,
        "-k",
        "--max-k",
        help="Maximum k to compute memos and ratios for",
    ),
    force: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "-f",
        "--force",
        help="Force recomputation even if files exist",
    ),
) -> None:
    """CLI entry point."""
    memos_path = data_dir / f"memos_{max_n}_{max_k}.parquet.gzip"
    ratios_path = data_dir / f"ratios_{max_n}_{max_k}.parquet.gzip"
    if force or not (memos_path.exists() and ratios_path.exists()):
        typer.echo("Computing memos and ratios")
        ks = list(range(2, max_k + 1))
        memos_df, ratios_df = memo_for_ks_n(ks, max_n)
        memos_df.to_parquet(memos_path)
        ratios_df.to_parquet(ratios_path)
    else:
        typer.echo(f"Loading memos and ratios from {data_dir}")
        memos_df = pandas.read_parquet(memos_path)
        ratios_df = pandas.read_parquet(ratios_path)

    typer.echo(f"Memos shape: {memos_df.shape}, Ratios shape: {ratios_df.shape}")
    typer.echo(f"Memos info:\n{memos_df.info()}")
    typer.echo(f"Ratios info:\n{ratios_df.info()}")


__all__ = ["gen_ratios", "memo_for_k_n", "memo_for_ks_n"]
