__usage = {}
bb = "\033[1m"
be = "\033[0m"

__usage['simplify_data'] = """
{}simplify_data{} (data=None, inplace=False, columns=None, save_csv=False,
                   file_name='data_simplified', save_dir='data/')
{}Description:{}
    Removes all columns except 'COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME', 'RESULT', and 'RESULT_UNITS'.
    If the user specifies additional columns in addition to the ones listed above, those columns will be kept.
    The function returns a dataframe and has an optional parameter to be able to save the dataframe to a csv file.
{}Parameters:{}
    {}data (dataframe):{} data to simplify
    {}inplace (bool):{} save data to current working dataset
    {}columns (list of strings):{} list of any additional columns on top of  ['COLLECTION_DATE', 'STATION_ID', 
        'ANALYTE_NAME', 'RESULT', and 'RESULT_UNITS'] to be kept in the dataframe.
    {}save_csv (bool):{} flag to determine whether or not to save the dataframe to a csv file.
    {}file_name (string):{} name of the csv file you want to save
    {}save_dir (string):{} name of the directory you want to save the csv file to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['get_MCL'] = """
{}get_MCL{} (analyte_name)
{}Description:{}
    Returns the Maximum Concentration Limit value for the specified analyte.
    Example: 'TRITIUM' returns 1.3
{}Parameters:{}
    {}analyte_name (string):{} name of the analyte to be processed
""".format(bb, be, bb, be, bb, be, bb, be)

__usage['get_unit'] = """
{}get_unit{} (analyte_name)
{}Description:{}
    Returns the unit of the analyte you specify.
    Example: 'DEPTH_TO_WATER' returns 'ft'
{}Parameters:{}
    {}analyte_name (string):{} name of the analyte to be processed
""".format(bb, be, bb, be, bb, be, bb, be)

__usage['filter_wells'] = """
{}filter_wells{} (units)
{}Description:{}
    Returns a list of the well names filtered by the unit(s) specified.
{}Parameters:{}
    {}units (list of strings):{}  Letter of the well to be filtered (e.g. [‘A’] or [‘A’, ‘D’])
""".format(bb, be, bb, be, bb, be, bb, be)

__usage['remove_outliers'] = """
{}remove_outliers{} (data, z_threshold=4)
{}Description:{}
    Removes outliers from a dataframe based on the z_scores and returns the new dataframe.
{}Parameters:{}
    {}data (dataframe):{} data for the outliers to removed from
    {}z_threshold (float):{} z_score threshold to eliminate.
""".format(bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['get_analyte_details'] = """
{}Description:{}
    Returns a csv file saved to save_dir with details pertaining to the specified analyte.
    Details include the well names, the date ranges and the number of unique samples.
{}Parameters:{}
    {}analyte_name (string):{} name of the analyte to be processed
    {}save_dir (string):{} name of the directory you want to save the csv file to
""".format(bb, be, bb, be, bb, be, bb, be)

__usage['get_data_summary'] = """
{}get_data_summary{} (analytes=None, sort_by='date', ascending=False)
{}Description:{}
    Returns a dataframe with a summary of the data for certain analytes.
    Summary includes the date ranges and the number of unique samples and other statistics for the analyte results.
{}Parameters:{}
    {}analytes (list of strings):{} list of analyte names to be processed. 
        If left empty, a list of all the analytes in the data will be used.
    {}sort_by (string):{} {{‘date’, ‘samples’, ‘wells’}} sorts the data by either the dates by entering: ‘date’, the 
        samples by entering: ‘samples’, or by unique well locations by entering ‘wells’.
    {}ascending (bool):{} flag to sort in ascending order.
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['query_data'] = """
{}query_data{} (well_name, analyte_name)
{}Description:{} 
    Filters data by passing the data and specifying the well_name and analyte_name
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}analyte_name (string):{} name of the analyte to be processed
""".format(bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_data'] = """
{}plot_data{} (well_name, analyte_name, log_transform=True,
     alpha=0, year_interval=2, plot_inline=True, save_dir='plot_data')
{}Description:{}
    Plot concentrations over time of a specified well and analyte with a smoothed curve on interpolated data points.
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}analyte_name (string):{} name of the analyte to be processed
    {}log_transform (bool):{} choose whether or not the data should be transformed to log base 10 values
    {}alpha (int):{} value between 0 and 10 for line smoothing
    {}year_interval (int):{} plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
    {}plot_inline (bool):{} choose whether or not to show plot inline
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be,
            bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_all_data'] = """
{}plot_all_data{} (log_transform=True, alpha=0,
            year_interval=2, plot_inline=True, save_dir='plot_data')
{}Description:{} 
    Plot concentrations over time for every well and analyte with a smoothed curve on interpolated data points.
{}Parameters:{}
    {}log_transform (bool):{} choose whether or not the data should be transformed to log base 10 values
    {}alpha (int):{} value between 0 and 10 for line smoothing
    {}year_interval (int):{} plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
    {}plot_inline (bool):{} choose whether or not to show plot inline
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_correlation_heatmap'] = """
{}plot_correlation_heatmap{} (well_name, show_symmetry=True, color=True,
     	          save_dir='plot_correlation_heatmap')
{}Description:{}
    Plots a heatmap of the correlations of the important analytes over time for a specified well.
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}show_symmetry (bool):{} choose whether or not the heatmap should show the same information twice over the diagonal
    {}color (bool):{} choose whether or not the plot should be in color or in greyscale
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_all_correlation_heatmap'] = """
{}plot_all_correlation_heatmap{} (show_symmetry=True, color=True, 
    save_dir='plot_correlation_heatmap')
{}Description:{}
    Plots a heatmap of the correlations of the important analytes over time for each well in the dataset.
{}Parameters:{}
    {}show_symmetry (bool):{} choose whether or not the heatmap should show the same information twice over the diagonal
    {}color (bool):{} choose whether or not the plot should be in color or in greyscale
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['interpolate_well_data'] = """
{}interpolate_well_data{} (well_name, analytes, frequency='2W')
{}Description:{} 
    Resamples the data based on the frequency specified and interpolates the values of the analytes.
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}analytes (list of strings):{} list of analyte names to use
    {}frequency (string):{} {{‘D’, ‘W’, ‘M’, ‘Y’}} frequency to interpolate. 
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. 
        (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_corr_by_well'] = """
{}plot_corr_by_well{}(well_name, interpolate=False, frequency='2W',
        save_dir='plot_correlation')
{}Description:{} 
    Plots the correlations with the physical plots as well as the correlations of the important analytes over time for a specified well.
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}remove_outliers (bool):{} choose whether or to remove the outliers.
    {}z_threshold (float):{} z_score threshold to eliminate outliers
    {}interpolate (bool):{} choose whether or not to interpolate the data
    {}frequency (string):{} {{‘D’, ‘W’, ‘M’, ‘Y’}} frequency to interpolate. 
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. 
        (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_all_corr_by_well'] = """
{}plot_all_corr_by_well{} (well_name, interpolate=False, frequency='2W',
               save_dir='plot_correlation')
{}Description:{} 
    Plots the correlations with the physical plots as well as the important analytes over time for each well in the dataset.
{}Parameters:{}
    {}remove_outliers (bool):{} choose whether or to remove the outliers.
    {}z_threshold (float):{} z_score threshold to eliminate outliers
    {}interpolate (bool):{} choose whether or not to interpolate the data
    {}frequency (string):{} {{‘D’, ‘W’, ‘M’, ‘Y’}} frequency to interpolate. 
    See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs.
    (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_corr_by_date'] = """
{}plot_corr_by_date{} (date, min_samples=48, save_dir=‘plot_corr_by_date’)
{}Description:{} 
    Plots the correlations with the physical plots as well as the correlations of the important 
    analytes for ALL the wells on a specified date.
{}Parameters:
    {}date (string):{} date to be analyzed
    {}min_samples (int):{} minimum number of samples the result should contain in order to execute.
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_corr_by_year'] = """
{}plot_corr_by_year{}(year, min_samples=500, save_dir=‘plot_corr_by_year’)
{}Description:{} 
    Plots the correlations with the physical plots as well as the correlations of the important 
    analytes for ALL the wells in specified year.
{}Parameters:{}
    {}year (int):{} year to be analyzed
    {}min_samples (int):{} minimum number of samples the result should contain in order to execute.
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_MCL'] = """
{}plot_MCL{} (well_name, analyte_name, year_interval=5, save_dir=‘plot_MCL’)
{}Description:{} 
    Plots the linear regression line of data given the analyte_name and well_name. 
    The plot includes the prediction where the line of best fit intersects with the 
    Maximum Concentration Limit (MCL).
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}analyte_name (string):{} name of the analyte to be processed
    {}year_interval (int):{} plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_PCA_by_date'] = """
{}plot_PCA_by_date{} (date, n_clusters=4, min_samples=48,
                        filter=False, filter_well_by=['D'], return_clusters=False,
                        show_labels=True, save_dir=‘plot_corr_by_date’)
{}Description:{} 
    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a date in the dataset.
    Only uses the 6 important analytes. The data is also clustered into n_clusters.
{}Parameters:{}
    {}date (string):{} date to be analyzed
    {}n_clusters (int):{} number of clusters to split the data into.
    {}filter (bool):{} Flag to indicate well filtering.
    {}filter_well_by (list of strings):{} Letter of the well to be filtered (e.g. [‘A’] or [‘A’, ‘D’])
    {}return_clusters (bool):{} Flag to return the cluster data to be used for spatial plotting.
    {}min_samples (int):{} minimum number of samples the result should contain in order to execute.
    {}show_labels (bool):{} choose whether or not to show the name of the wells.
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_PCA_by_year'] = """
{}plot_PCA_by_year{} (year, n_clusters=4, min_samples=48, filter=False,
                        filter_well_by=['D'], return_clusters=False, show_labels=True,
                        save_dir=‘plot_corr_by_year’)
{}Description:{} 
    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a year in the dataset. 
    Only uses the 6 important analytes. The data is also clustered into n_clusters.
{}Parameters:{}
    {}year (int):{} date to be analyzed
    {}n_clusters (int):{} number of clusters to split the data into.
    {}filter (bool):{} Flag to indicate well filtering.
    {}filter_well_by (list of strings):{} Letter of the well to be filtered (e.g. [‘A’] or [‘A’, ‘D’])
    {}return_clusters (bool):{} Flag to return the cluster data to be used for spatial plotting.
    {}min_samples (int):{} minimum number of samples the result should contain in order to execute.
    {}show_labels (bool):{} choose whether or not to show the name of the wells.
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_PCA_by_well'] = """
{}plot_PCA_by_well{} (well_name, interpolate=False, frequency='2W', min_samples=48,
            show_labels=True, save_dir=‘plot_PCA_by_well’)
{}Description:{} 
    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a well_name in the dataset. 
    Only uses the 6 important analytes.
{}Parameters:{}
    {}well_name (string):{} name of the well to be processed
    {}interpolate (bool):{} choose whether or to interpolate the data
    {}frequency (string):{} {{‘D’, ‘W’, ‘M’, ‘Y’}} frequency to interpolate. 
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. 
        (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    {}min_samples (int):{} minimum number of samples the result should contain in order to execute.
    {}show_labels (bool):{} choose whether or not to show the name of the wells.
    {}save_dir (string):{} name of the directory you want to save the plot to
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)

__usage['plot_coordinates_to_map'] = """
{}plot_coordinates_to_map{} (gps_data, center=[33.271459, -81.675873], zoom=14)
{}Description:{} 
    Plots the well locations on an interactive map given coordinates.
{}Parameters:{}
    {}gps_data (dataframe):{} Data frame with the following column names: station_id, latitude, longitude, color. 
        If the color column is not passed, the default color will be blue.
    {}center (list with 2 floats):{} latitude and longitude coordinates to center the map view.
    {}zoom (int):{} value to determine the initial scale of the map
""".format(bb, be, bb, be, bb, be, bb, be, bb, be, bb, be)




def get_FunctionDescription(function=None):
    if (function == None):
        for key, num in zip(list(__usage.keys()), range(len(list(__usage.keys())))):
            if(num==0):
                print("--------------------------------------------------------------------------------------")
            print(bb + str(num+1)+ ") "+ key + be)
            print("------------------------------------------")
            print(__usage[key])
            print("--------------------------------------------------------------------------------------")
    else:
        print(__usage[function])

def get_FunctionList():
    keys = list(__usage.keys())
    print(bb + "pyLEnM functions:" + be)
    for x in range(len(keys)):
        print("    " + bb + "{})".format(x+1) + be + " {}".format(keys[x]))
    print("\n")