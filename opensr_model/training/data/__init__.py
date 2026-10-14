"""TACO catalogs, paired datasets, and data modules."""

from .taco import (
    InvalidPairError,
    OpenSRDataModule,
    TACO_SUFFIX,
    TACO_SUFFIXES,
    TacoPairedDataset,
    TacoSampleRecord,
    resolve_taco_paths,
    scan_taco_catalog,
    split_taco_records,
)

__all__ = [
    "InvalidPairError",
    "OpenSRDataModule",
    "TACO_SUFFIX",
    "TACO_SUFFIXES",
    "TacoPairedDataset",
    "TacoSampleRecord",
    "resolve_taco_paths",
    "scan_taco_catalog",
    "split_taco_records",
]


def __getattr__(name):
    # Older benchmark scripts imported private helpers from the former data.py.
    from . import taco

    return getattr(taco, name)
