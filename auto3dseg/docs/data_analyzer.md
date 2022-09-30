## Data Analyzer

In **Auto3DSeg**, the data analyzer automatically analyzes given medical image dataset and reports the statistics of image intensity, shape, spacing, etc.

### Usage

The ussage of data analyzer is can be found [here](../notebooks/data_analyzer.ipynb)

### Customization

**Auto3DSeg** also provides the API for users to use their own customized data analyzing function as shown in this [notebook](../notebooks/data_analyzer_byoc.ipynb).

### Output

"datastats.yaml" is a summary of the dataset from the data analyzer. The summary report includes information such as data size, spacing, intensity distribution, etc., for a better understanding of the dataset. An example "datastats.yaml" is shown as follows.

```
...
stats_summary:
  image_foreground_stats:
    intensity: {max: 1326.0, mean: 353.68545989990236, median: 339.03333333333336,
      min: 0.0, percentile_00_5: 94.70366643269857, percentile_10_0: 210.9, percentile_90_0: 518.3333333333334,
      percentile_99_5: 734.7439453125, stdev: 122.72876790364583}
  image_stats:
    channels:
      max: 2
      mean: 2.0
      median: 2.0
      min: 2
      percentile: [2, 2, 2, 2]
      percentile_00_5: 2
      percentile_10_0: 2
      percentile_90_0: 2
      percentile_99_5: 2
      stdev: 0.0
    intensity: {max: 2965.0, mean: 307.1866872151693, median: 239.9, min: 0.0, percentile_00_5: 1.5333333333333334,
      percentile_10_0: 54.53333333333333, percentile_90_0: 649.3333333333334, percentile_99_5: 1044.0333333333333,
      stdev: 238.39599100748697}
    shape:
      max: [384, 384, 24]
      mean: [317.8666666666667, 317.8666666666667, 18.8]
...
```
