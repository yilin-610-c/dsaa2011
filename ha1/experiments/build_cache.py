import argparse
import time

from experiments.data_utils import build_or_load_tensor_cache


def main(parquet_path: str, cache_path: str) -> None:
    start = time.time()
    cache = build_or_load_tensor_cache(parquet_path=parquet_path, cache_path=cache_path)
    elapsed = time.time() - start
    images_shape = tuple(cache["images_uint8"].shape)
    print(f"Cache ready at {cache_path}")
    print(f"images_uint8 shape={images_shape}, labels={cache['labels'].shape[0]}")
    print(f"elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=str, default="data.parquet")
    parser.add_argument("--cache-path", type=str, default="results/cache/fashion_tensor_cache.pt")
    args = parser.parse_args()
    main(args.parquet_path, args.cache_path)
