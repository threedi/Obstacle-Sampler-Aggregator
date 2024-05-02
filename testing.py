from obstacle_sampler_aggregator import Obstacle_Sampler_Aggregator

run_it = Obstacle_Sampler_Aggregator(
    obstacles_path = r"N:\Kasper\obstacles_10m_split_clean_met_watergang_intersect.gpkg", 
    DEM_path = r"N:\Kasper\dem_ontsluitingsroute_ahn4.tif",
    obstacles_layer_name = "bgt_watervlakken_1m_buff_lines_diss_sp_split10m_crest_level__crest_level",
    dem_filter_values = [-9999, 10],
    no_split=True,
    no_aggregate=True,
    aggregation_threshold = 0.1,
    splitting_segment_length = 20,
    sampling_buffer_size = 1,
    percentile = 0.95,
)
# todo documentatie







