from smart_open import register_compressor

__all__ = ["mk_compression"]


def mk_compression():
    def _handle_zst(file_obj, mode):
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError("zstandard is required for zstd support")

        return zstd.open(file_obj, mode)

    register_compressor(".zstd", _handle_zst)
    register_compressor(".zst", _handle_zst)
