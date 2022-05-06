## Example breast density data

Download example data from https://drive.google.com/file/d/1Fd9GLUIzbZrl4FrzI3Huzul__C8wwzyx/view?usp=sharing.
Extract here.

## Data source
This example data is based on [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) from [TCIA](https://wiki.cancerimagingarchive.net/) [1].

We preprocessed all files using `code/pt/utils/preprocess_dicomdir.py` and generated train/val splits for each client
and separate testing split.

For more details on this example data, see [2,3].

## References
[1] Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: https://doi.org/10.1007/s10278-013-9622-7

[2] Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). Curated Breast Imaging Subset of DDSM [Dataset]. The Cancer Imaging Archive. DOI:  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

[3] Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi, Kanae Kawai Miyake, Mia Gorovoy & Daniel L. Rubin. (2017) A curated mammography data set for use in computer-aided detection and diagnosis research. Scientific Data volume 4, Article number: 170177 DOI: https://doi.org/10.1038/sdata.2017.177
