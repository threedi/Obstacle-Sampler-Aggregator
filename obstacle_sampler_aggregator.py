import argparse
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
from shapely import make_valid
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

### Custom exceptions

class FileNotFoundError(Exception):
    pass

class LayerNotFoundError(Exception):
    pass

class LayerEmpty(Exception):
    pass

class Obstacle_Sampler_Aggregator:
    name = "Threedi Linear Obstacle Sampler & Aggregator"
    
    def __init__(self, obstacles_path, DEM_path, obstacles_layer_name=None, 
                 dem_filter_values = [-9999, 10], no_split=False, no_aggregate=False,
                 aggregation_threshold=0.1, splitting_segment_length=20, sampling_buffer_size=1, 
                 percentile=0.95):
        
        self.obstacles_path = Path(obstacles_path)
        self.DEM_path = Path(DEM_path)
        self.obstacles_layer_name = obstacles_layer_name
        self.dem_filter_values = dem_filter_values
        self.no_split = no_split
        self.no_aggregate = no_aggregate
        self.aggregation_threshold = aggregation_threshold
        self.splitting_segment_length = splitting_segment_length
        self.sampling_buffer_size = sampling_buffer_size
        self.percentile = percentile
        
        self.check_file_exists(self.obstacles_path)
        self.check_file_exists(self.DEM_path)
        
        # Read input data
        self.obstacles = self.read_to_gdf(self.obstacles_path, self.obstacles_layer_name)
        self.raster = self.read_raster(self.DEM_path)
        
        # Fix geometries
        print('Validating input geometries...')
        self.fixed_obstacles = self.fix_invalid_geometries(self.obstacles)
        
        # Prepare obstacles by splitting
        if not self.no_split:
            self.fixed_obstacles = self.fixed_obstacles.explode(index_parts=False) # Mutli to Single
            self.fixed_obstacles['geometry'] = self.fixed_obstacles['geometry'].apply(lambda geom: geom.simplify(0.5))
            self.splitted_obstacles = self.split_line_geometries(self.fixed_obstacles, self.splitting_segment_length)
        else:
            self.splitted_obstacles = self.fixed_obstacles.copy()
        
        # Sample raster values
        self.sampled_obstacles = self.extract_raster_values(self.splitted_obstacles, self.raster, 
                                                            filter_values=self.dem_filter_values)
        
        # Prepare recursive aggregation
        if not self.no_aggregate:
            initial_gdf = self.sampled_obstacles.explode(index_parts=False)
            initial_gdf["store_values"] = initial_gdf[f"{str(int(100*self.percentile))} percentile value"].apply(lambda x: [round(x, 2)])
            initial_gdf = initial_gdf.drop(columns=[f"{str(int(100*self.percentile))} percentile value"])
            
            # Recursive aggregation
            self.aggregated_obstacles = self.recursive_aggregation(initial_gdf)
            
            # Compute length and crest level       
            self.aggregated_obstacles['length'] = round(self.aggregated_obstacles['geometry'].length, 1)
            self.aggregated_obstacles['crest_level'] = self.aggregated_obstacles['store_values'].apply(lambda x: round(sum(x) / len(x), 2) if x else None)
            
            # Prepare output
            self.aggregated_obstacles['store_values'] = self.aggregated_obstacles['store_values'].apply(lambda x: str(x))
        
        # Write output               
        output_path = self.obstacles_path.parent / (self.obstacles_path.stem + '_crest_level_sampler' + self.obstacles_path.suffix)
        self.write(self.obstacles, output_path, layer='input obstacles')

        if not self.no_split:
            self.splitted_obstacles.drop(columns=['raster_values'], inplace=True)
            self.write(self.splitted_obstacles, output_path, layer='splitted obstacles')

        self.write(self.sampled_obstacles, output_path, layer='sampled obstacles')

        if not self.no_aggregate:
            self.write(self.aggregated_obstacles, output_path, layer='aggregated obstacles')
        
        
    def recursive_aggregation(self, gdf, buffer_distance=0.01, threshold=10, recursion_count=0):
        gdf.reset_index(inplace=True, drop=True)
        gdf.sindex
    
        aggregated_rows = []
        aggregated_indices = set()
    
        for idx, line in tqdm(gdf.iterrows(), total=len(gdf), unit='lines', desc=f'Aggregating lines (Recursion {recursion_count})'):
            if idx in aggregated_indices:
                continue  # Skip lines that have already been aggregated
    
            possible_connections = list(gdf.sindex.intersection(line.geometry.bounds))
            possible_connections.remove(idx)
    
            aggregated_indices.add(idx)  # Mark the current line as aggregated
    
            for candidate_idx in possible_connections:
                if candidate_idx in aggregated_indices:
                    continue  # Skip lines that have already been aggregated
    
                candidate_line = gdf.iloc[candidate_idx]
    
                if (
                        abs(max(line['store_values']) - min(candidate_line['store_values'])) < self.aggregation_threshold 
                        and 
                        abs(min(line['store_values']) - max(candidate_line['store_values'])) < self.aggregation_threshold
                    ):
                    # Create a small buffer around the line
                    buffered_line = line.geometry.buffer(buffer_distance)
    
                    # Check if the buffered line intersects with the candidate line
                    if buffered_line.intersects(candidate_line.geometry):
                        # Combine the geometries of intersecting lines
                        try:
                            multi_line = MultiLineString([line.geometry, candidate_line.geometry])
                            combined_geometry = linemerge(multi_line)
                        except TypeError as e:
                            if "MultiLineString" in str(e) and "not iterable" in str(e):
                                combined_geometry = line.geometry.union(candidate_line.geometry)
                            else:
                                raise
    
                        # Store the aggregated geometries
                        aggregated_row = {'geometry': combined_geometry}
    
                        # Store the values
                        store_values = line['store_values']
                        additional_values = candidate_line['store_values']
                        aggregated_row['store_values'] = store_values + additional_values
    
                        # Store the other information
                        other_columns = line.drop(['geometry', 'store_values'])
                        aggregated_row.update(other_columns)
    
                        # Store to list
                        aggregated_rows.append(aggregated_row)
    
                        # Mark the candidate line as aggregated
                        aggregated_indices.add(candidate_idx)
    
        aggregated_gdf = gpd.GeoDataFrame(aggregated_rows, crs=gdf.crs)
        aggregated_gdf.sindex
                
        individual_gdf = gdf.copy()
        individual_gdf.sindex
    
        to_drop_index = set()
    
        for idx, aggregated_line in tqdm(aggregated_gdf.iterrows(), total=len(aggregated_gdf),
                                         unit='lines', desc='Deleting the individual lines that have been aggregated'):
    
            possible_matches_index = list(individual_gdf.sindex.intersection(aggregated_line.geometry.buffer(buffer_distance).bounds))
            possible_matches = individual_gdf.iloc[possible_matches_index]
    
            to_drop = possible_matches[possible_matches['geometry'].within(aggregated_line.geometry.buffer(buffer_distance))].index
            to_drop_index.update(to_drop)
    
        individual_gdf = individual_gdf.drop(to_drop_index)

        '''
        for idx, aggregated_line in tqdm(aggregated_gdf.iterrows(), total=len(aggregated_gdf),
                                 unit='lines', desc='Deleting the individual lines that have been aggregated'):

            individual_gdf = individual_gdf[~individual_gdf['geometry'].within(aggregated_line.geometry.buffer(buffer_distance))]
        '''        
        merged_gdf = pd.concat([aggregated_gdf, individual_gdf])
    
        # Check the length difference
        length_difference = len(gdf) - len(merged_gdf)
    
        if length_difference < threshold:
            print(f"Succesfully aggregated {length_difference} lines during Recursion {recursion_count}. Quit aggregating.")
            return merged_gdf
        else:
            print(f"Succesfully aggregated {length_difference} lines during Recursion {recursion_count}. Keep on aggregating.")
            # Recursively call the function with the merged GeoDataFrame
            return self.recursive_aggregation(merged_gdf, buffer_distance, threshold, recursion_count + 1)

    def extract_raster_values(self, gdf, raster, filter_values=[-9999, 10]):
        """
        Extracts raster values around each linestring in a GeoDataFrame.
        Adds a new column 'percentile value' with the percentile value for each linestring.
    
        Parameters:
        - gdf (geopandas.GeoDataFrame): GeoDataFrame with linestring geometries.
        - raster (rasterio._io.RasterReader): Rasterio DatasetReader object.
    
        Returns:
        - gdf (geopandas.GeoDataFrame): Updated GeoDataFrame with the 'percentile value' column.
        """
     
        # Apply the function to get raster values for each row
        tqdm.pandas(desc="Sampling DEM for lines", unit="lines")
        gdf['raster_values'] = gdf['geometry'].progress_apply(lambda geom: self._get_raster_value(geom, raster, filter_values))
        
        # Calculate the percentile value for each row
        gdf[f"{str(int(100*self.percentile))} percentile value"] = gdf['raster_values'].apply(lambda x: np.percentile(x, self.percentile * 100) if x else None)
    
        # Drop temporary column
        gdf = gdf.drop(columns=['raster_values'])
    
        return gdf
            
    def _get_raster_value(self, geometry, raster, filter_values):
        values = []
        
        # Check if the line intersects with the raster extent
        bounds = geometry.bounds
        raster_extent = raster.bounds
        pixel_size = raster.res[0]
               
        if not (bounds[0] < raster_extent[2] and bounds[2] > raster_extent[0] and
                bounds[1] < raster_extent[3] and bounds[3] > raster_extent[1]):
            # Line is outside the raster extent, return NaN
            return [np.nan]
    
        # Create a line perpendicular to the input line at each vertex
        for point in geometry.coords:
            perpendicular_line = self.perpendicular_line_at_vertex(geometry, point, self.sampling_buffer_size).segmentize(pixel_size)
            new_values = self.sample_raster_values(perpendicular_line, raster, filter_values)
            if new_values:
                values.extend(new_values)
    
        if values:
            return values
        else:
            return [np.nan]
        
    def perpendicular_line_at_vertex(self, line, vertex, length):
        """
        Create a perpendicular line with a specified length at a given vertex of the input line.
    
        Parameters:
        - line (shapely.geometry.LineString): Input line.
        - vertex (tuple): Vertex coordinates (x, y).
        - length (float): Length of the perpendicular line on both sides of the original line.
    
        Returns:
        - perpendicular_line (shapely.geometry.LineString): Perpendicular line.
        """
        p1 = Point(vertex)
        angle = np.arctan2(line.coords[1][1] - line.coords[0][1], line.coords[1][0] - line.coords[0][0])
    
        # Calculate perpendicular line coordinates
        x_offset = length * np.sin(angle)
        y_offset = length * np.cos(angle)
    
        p2 = Point(vertex[0] + x_offset, vertex[1] - y_offset)
        p3 = Point(vertex[0] - x_offset, vertex[1] + y_offset)
    
        perpendicular_line = LineString([p2, p1, p3])
        return perpendicular_line
    
    def sample_raster_values(self, line, raster, filter_values):
        # Extract the coordinates from the LineString
        coords = list(line.coords)

        # Sample raster values along the line using rasterio's sample function
        values = list(raster.sample(coords))
        
        # Filter values
        values = [float(value) for value in values if value not in filter_values]
        
        if values:
            return values
        else:
            # All values where filtered out, return empty list
            return []
    
    def split_line_geometries(self, gdf, max_segment_length):
        """ check if lines are longer than threshold and if so, split them into smaller segments"""
        split_rows = []
    
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Splitting lines"):
            geometry = row['geometry']
    
            if isinstance(geometry, LineString) and geometry.length > max_segment_length:
                segments = self._split_line_by_length(geometry, max_segment_length)
                for segment in segments:
                    new_row = row.copy()
                    new_row['geometry'] = segment
                    split_rows.append(new_row)
            else:
                split_rows.append(row)
    
        return gpd.GeoDataFrame(split_rows, columns=gdf.columns, crs=gdf.crs)
        
    def _split_line_by_length(self, line_geometry, max_segment_length):
        """ check if line is longer than threshold and if so, split into smaller segments"""
        coords = list(line_geometry.coords)
        segments = []
    
        for i in range(len(coords) - 1):
            start = Point(coords[i])
            end = Point(coords[i + 1])
            segment_length = start.distance(end)
            
            if segment_length <= max_segment_length:
                segments.append(LineString([start, end]))
            else:
                num_subsegments = int(np.ceil(segment_length / max_segment_length))
                x_coords = np.linspace(start.x, end.x, num_subsegments + 1)
                y_coords = np.linspace(start.y, end.y, num_subsegments + 1)
                
                subsegment_points = [Point(x, y) for x, y in zip(x_coords, y_coords)]
                segments.extend([LineString([subsegment_points[j], subsegment_points[j + 1]]) for j in range(len(subsegment_points) - 1)])
    
        return segments

    def check_file_exists(self, file_path):
        if file_path.exists():
            return 
        else:
            raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    def read_to_gdf(self, path, layer_name):
        if path.suffix == '.gpkg':
            try:
                gdf = gpd.read_file(path, layer=layer_name)
            except:
                raise LayerNotFoundError(f"The file at '{path}' does not contain a layer named '{layer_name}'.")
        else:
            gdf = gpd.read_file(path)
            
        return gdf
    
    def fix_invalid_geometries(self, gdf):

        # Function to validate and fix LineString geometries
        def fix_line(line):
            valid_line = make_valid(line)
            if valid_line.is_empty:
                return line
            else:
                return valid_line
        
        # Counters for feedback
        invalid_features_removed = 0
        invalid_geometries_fixed = 0
        
        indices_to_remove = set()
        
        # Iterate through rows
        for index, row in gdf.iterrows():
            # Check if geometry is valid
            if not row['geometry'].is_valid:
                # Try to fix the geometry
                fixed_geometry = fix_line(row['geometry'])
                
                # If fixing is successful, update the geometry
                if fixed_geometry.is_valid:
                    gdf.at[index, 'geometry'] = fixed_geometry
                    invalid_geometries_fixed += 1
                else:
                    # If fixing is not possible, we will remove the row
                    indices_to_remove.add(index)
                    invalid_features_removed += 1
                    
            # Check if line is not too small
            if row['geometry'].length < 0.001:
                indices_to_remove.add(index)
                invalid_features_removed += 1   
        
        gdf = gdf.drop(indices_to_remove)
        
        # Print feedback
        print(f"{invalid_features_removed} invalid features removed.")
        print(f"{invalid_geometries_fixed} invalid geometries made valid.")
        
        return gdf
    
    def read_raster(self, raster_path):
        raster = rasterio.open(raster_path)
        return raster
    
    def write(self, gdf, path, layer=None):
        if path.suffix == '.gpkg':
            gdf.to_file(path, layer=layer, driver="GPKG")
        else:
            gdf.to_file(path)
                          
def run(
        obstacles_path, DEM_path, obstacles_layer_name=None, dem_filter_values=[-9999, 10],
        no_split=False, no_aggregate=False, aggregation_threshold=0.1, splitting_segment_length=20,
        sampling_buffer_size=1, percentile=0.95):

    obstacle_sampler_aggregator = Obstacle_Sampler_Aggregator(
        obstacles_path = obstacles_path, 
        obstacles_layer_name = obstacles_layer_name,
        DEM_path = DEM_path,
        dem_filter_values = dem_filter_values,
        no_split = no_split,
        no_aggregate = no_aggregate,
        aggregation_threshold=aggregation_threshold,
        splitting_segment_length = splitting_segment_length,
        sampling_buffer_size = sampling_buffer_size,
        percentile = percentile
    )
    
def get_parser():
    """Return argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument(
        "obstacles_path",
        help="Required. Path to vector containing geometry of linear obstacles. Most common shape file extensions supported.",
    )
               
    parser.add_argument(
        "--obstacles_layer_name", 
        help="Optional. Will be used as layer name in case obstacles_path is of format geopackage (.gpkg)",
    )
    
    parser.add_argument(
        "DEM_path",
        help="Required. Path to raster containing ground elevations. Most raster file extensions supported.",
    )    
        
    parser.add_argument(
        "dem_filter_values",
        help="Required. Values that are not used to compute crest level from DEM sampling, like NoData values and burn values of channels.",
    )    

    # argument for not splitting, save as True
    parser.add_argument(
        "--no_split", 
        help="Optional. If set, the input lines are not split into smaller segments.",
    )

    # argument for not aggregating, save as True
    parser.add_argument(
        "--no_aggregate", 
        help="Optional. If set, the input lines are not aggregated.",
    )

    parser.add_argument(
        "--aggregation_threshold",
        help=("Optional. Threshold for aggregation of neighbouring obstacle lines. Value in meters in terms of +- elevation difference of crest levels."
              "If 0.1 m is set, values within a margin of -0.1 and +0.1 are considered similar. Default is 0.1 m."),
    )       

    parser.add_argument(
        "--splitting_segment_length",
        help="Optional. Maximum length of line segments after splitting. Value in meters. Default is 20 m.",
    ) 

    parser.add_argument(
        "--sampling_buffer_size",
        help="Optional. Buffer size in meters around the line to sample the DEM. Default is 1 m.",
    )

    parser.add_argument(
        "--percentile",
        help="Optional. Percentile value to compute the crest level. Default is 0.95.",
    )
    
    return parser


def main():
    """Call extract_all with args from parser."""
    return run(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    exit(main())    




