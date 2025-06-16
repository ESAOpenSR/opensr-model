from setuptools import setup, find_packages

# read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opensr-model',
    version='0.3.6',
    author = "Simon Donike, Cesar Aybar, Luis Gomez Chova, Freddie Kalaitzis",
    author_email = "accounts@donike.net",
    description = "ESA OpenSR Diffusion model package for Super-Resolution of Senintel-2 Imagery",
    url = "https://isp.uv.es/opensr/",
    project_urls={'Source Code': 'https://github.com/ESAopenSR/opensr-model'},
    license='MIT',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
	'numpy',
	'einops',
	'rasterio',
	'tqdm',
	'torch',
	'scikit-image',
    'pytorch-lightning',
    'requests',
    'omegaconf',],
)
