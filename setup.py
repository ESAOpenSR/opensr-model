from setuptools import setup, find_packages

# read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="opensr-model",
    version="1.1.1",
    author="Simon Donike, Cesar Aybar, Luis Gomez Chova, Freddie Kalaitzis",
    author_email="accounts@donike.net",
    description="ESA OpenSR Diffusion model package for Super-Resolution of Senintel-2 Imagery",
    url="https://opensr.eu",
    project_urls={"Source Code": "https://github.com/ESAopenSR/opensr-model"},
    license="MIT",
    python_requires=">=3.12",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.26.0",
        "einops>=0.7.0",
        "rasterio>=1.4.0",
        "tqdm>=4.66.0",
        "torch>=2.2.0",
        "scikit-image>=0.22.0",
        "pytorch-lightning>=2.2.0,<3.0",
        "requests>=2.31.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.8.0",
    ],
    extras_require={
        "train": [
            "Pillow>=10.2.0",
            "tacoreader>=2.4.0,<3.0",
            "pandas>=2.2.0",
            "fsspec>=2024.2.0",
            "torchmetrics>=1.3.0",
            "torchvision>=0.17.0",
            "lpips>=0.1.4",
            "tensorboard>=2.16.0",
        ],
        "wandb": [
            "wandb>=0.17.0",
        ],
        "test": [
            "pytest>=8.0.0",
        ],
        "hpc": [
            "PyYAML>=6.0",
            "pyproj>=3.4",
            "pyshp>=2.3.1",
            "rioxarray>=0.15",
            "shapely>=2.0.2",
            "cubo>=0.3",
            "opensr-utils>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "opensr-hpc=deployment.opensr_hpc.cli:main",
            "opensr-train=opensr_model.training.cli:main",
            "opensr-train-autoencoder=opensr_model.training.cli:autoencoder_main",
            "opensr-train-diffusion=opensr_model.training.cli:diffusion_main",
        ],
    },
    package_data={
        "opensr_model": ["configs/*.yaml"],
        "deployment.opensr_hpc": ["slurm/*.sh"],
    },
)
