# Linear Obstacle Sampler & Aggregator
A command-line tool to split linear obstacles in small even parts, sample the ground elevation of the DEM and aggregate obstacle lines with similar crest levels.

## Assumptations
* The obstacles are split in even parts of 20 m.
* For every vertice on the line a 20 m long perpendicular line is created to each side of the obstacle with a resolution of 1m (a vertice atleast every 1m). The DEM is sampled along this line.
* The 95th percentile value of the sampled raster value is added as crest level.
* The user has control over which DEM values to filter out. For instance NoData values, but also burned values in the raster that are not representative of actual heights.
* The user has control over the aggregation threshold of neighbouring obstacles. If 0.1 m is set, crest level values within a margin of -0.1 and +0.1 are considered similar.
* After aggregation of lines, the mean value of all aggregated crest levels is added as the crest level for the aggregated line.

## Output
A geopackage containing:
* Input obstacles
* Splitted obstacles
* Sampled obstacles
* Aggregated obstacles

This output can be used to check the output of the tool.

## Usage
```
usage: obstacle_sampler_aggregator.py [-h] [--obstacles_layer_name OBSTACLES_LAYER_NAME]
                                      obstacles_path DEM_path aggregation_threshold dem_filter_values

positional arguments:
  obstacles_path        Required. Path to vector containing geometry of linear obstacles. Most common shape file
                        extensions supported.
  DEM_path              Required. Path to raster containing ground elevations. Most raster file extensions supported.
  aggregation_threshold
                        Required. Threshold for aggregation of neighbouring obstacle lines. Value in meters in terms
                        of +- elevation difference of crest levels. If 0.1 m is set, values within a margin of -0.1 and
                        +0.1 are considered similar.
  dem_filter_values     Required. Values that are not used to compute crest level from DEM sampling, like NoData
                        values and burn values of channels.

options:
  -h, --help            show this help message and exit
  --obstacles_layer_name OBSTACLES_LAYER_NAME, -s OBSTACLES_LAYER_NAME
                        Optional. Will be used as layer name in case obstacles_path is of format geopackage (.gpkg)
```

## Dependencies
[Dependencies](dep)



