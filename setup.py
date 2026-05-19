from __future__ import annotations

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _openmp_args() -> tuple[list[str], list[str]]:
    if sys.platform.startswith("win"):
        return ["/O2"], []
    return ["-O3"], []


compile_args, link_args = _openmp_args()

ext_modules = [
    Pybind11Extension(
        "sfc._sfc_cpp",
        ["src/sfc/_cpp_sdf_contact.cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        optional=True,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
