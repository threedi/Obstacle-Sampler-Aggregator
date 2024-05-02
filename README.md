# Linear Obstacle Sampler & Aggregator
A command-line tool to split linear obstacles in small even parts, sample the ground elevation of the DEM and aggregate obstacle lines with similar crest levels.

## Assumptations
* The obstacles are split in even parts of a certain (user-defined) length.
* For every vertice on the line perpendicular line (with a user-defined length) is created to each side of the obstacle with a resolution of the pixel size of the raster (for isntance a vertice atleast every 0.5m). The DEM is sampled along this line.
* A certain percentile value (user-defined) of the sampled raster value is added as crest level.
* The user has control over which DEM values to filter out. For instance NoData values, but also burned values in the raster that are not representative of actual heights.
* The user has control over the aggregation threshold of neighbouring obstacles. If 0.1 m is set, crest level values within a margin of -0.1 and +0.1 are considered similar.
* After the aggregation of lines, the mean value of all aggregated crest levels is added as the crest level for the aggregated line.

## Output
A geopackage containing:
* Input obstacles
* Splitted obstacles
* Sampled obstacles
* Aggregated obstacles

The output can be used to validate the output of the tool.

## Usage
```
usage: obstacle_sampler_aggregator.py [-h] [--obstacles_layer_name OBSTACLES_LAYER_NAME] [--no_split NO_SPLIT]
                                      [--no_aggregate NO_AGGREGATE] [--aggregation_threshold AGGREGATION_THRESHOLD]
                                      [--splitting_segment_length SPLITTING_SEGMENT_LENGTH]
                                      [--sampling_buffer_size SAMPLING_BUFFER_SIZE] [--percentile PERCENTILE]
                                      obstacles_path DEM_path dem_filter_values

positional arguments:
  obstacles_path        Required. Path to vector containing geometry of linear obstacles. Most common shape file
                        extensions supported.
  DEM_path              Required. Path to raster containing ground elevations. Most raster file extensions supported.
  dem_filter_values     Required. Values that are not used to compute crest level from DEM sampling, like NoData
                        values and burn values of channels.

options:
  -h, --help            show this help message and exit
  --obstacles_layer_name OBSTACLES_LAYER_NAME
                        Optional. Will be used as layer name in case obstacles_path is of format geopackage (.gpkg)
  --no_split NO_SPLIT   Optional. If set, the input lines are not split into smaller segments.
  --no_aggregate NO_AGGREGATE
                        Optional. If set, the input lines are not aggregated.
  --aggregation_threshold AGGREGATION_THRESHOLD
                        Optional. Threshold for aggregation of neighbouring obstacle lines. Value in meters in terms
                        of +- elevation difference of crest levels.If 0.1 m is set, values within a margin of -0.1 and
                        +0.1 are considered similar. Default is 0.1 m.
  --splitting_segment_length SPLITTING_SEGMENT_LENGTH
                        Optional. Maximum length of line segments after splitting. Value in meters. Default is 20 m.
  --sampling_buffer_size SAMPLING_BUFFER_SIZE
                        Optional. Buffer size in meters around the line to sample the DEM. Default is 1 m.
  --percentile PERCENTILE
                        Optional. Percentile value to compute the crest level. Default is 0.95.
```

## Dependencies
[Dependencies](dep)



