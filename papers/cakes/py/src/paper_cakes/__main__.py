"""Placeholder docstring."""

import pathlib
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import numpy
import pandas
import typer

cli = typer.Typer(add_completion=False)


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
    for i, future in enumerate(as_completed(futures)):
        memos[:, i] = future.result()
        typer.echo(f"Computed memo for k={ks[i]}")
    typer.echo("Computed all memos")

    memos = memos[1:]  # Remove n=0
    ns = numpy.arange(1, n_max + 1)
    ratios = (memos / ns[:, None]).astype(numpy.float32)

    memos_df = pandas.DataFrame(memos, index=ns, columns=ks)
    ratios_df = pandas.DataFrame(ratios, index=ns, columns=ks)

    return memos_df, ratios_df


@cli.command()
def main(
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-i",
        "--inp-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Input directory",
    ),
    out_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-o",
        "--out-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Output directory",
    ),
    fresh: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "-f",
        "--fresh",
        help="Force recomputation even if files exist",
    ),
) -> None:
    """CLI entry point."""
    memos_path = inp_dir / "memos.parquet.gzip"
    ratios_path = inp_dir / "ratios.parquet.gzip"
    if not fresh and memos_path.exists() and ratios_path.exists():
        memos_df = pandas.read_parquet(memos_path)  # type: ignore
        ratios_df = pandas.read_parquet(ratios_path)  # type: ignore
        typer.echo(f"Loaded memos and ratios from {inp_dir}")
    else:
        typer.echo("Computing memos and ratios")
        # Precompute memos and ratios for plots
        max_n = 10_000_000
        ks = list(range(2, 257))
        memos_df, ratios_df = memo_for_ks_n(ks, max_n)
        memos_df.to_parquet(memos_path)
        ratios_df.to_parquet(ratios_path)

    typer.echo(f"Memos shape: {memos_df.shape}, Ratios shape: {ratios_df.shape}")
    typer.echo(f"Memos info:\n{memos_df.info()}")
    typer.echo(f"Ratios info:\n{ratios_df.info()}")

    del memos_df, ratios_df  # Free memory

    typer.echo(f"Ready to write outputs to {out_dir}")


if __name__ == "__main__":
    cli()
