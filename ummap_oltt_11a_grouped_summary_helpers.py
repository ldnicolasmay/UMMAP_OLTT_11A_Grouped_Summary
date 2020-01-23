##################
# Import Modules #

import re
import io
import os.path
import pandas as pd
from typing import List
from boxsdk import JWTAuth, Client


##################
# Useful Globals #


# Define columns that are are the source of summary stats
cols_recall: List[str] = ["trial num", " object", " env", " target X", " target Y",
                           " response X", " response Y", " deltatime", " error in px", " error in cm"]
cols_recogn: List[str] = ["trial num", " env", " obj", " location", " location chosen", " time"]
summ_cols_recall: List[str] = [" deltatime", " error in px", " error in cm"]
summ_cols_recogn: List[str] = [" time"]

stim_targ: pd.Series = \
    pd.Series([" dustpan", " clothespins", " wig", " cards", " mugs", " thread", " camcorder", " curler", " lotion",
               " radio", " pie", " plates", " purse", " money", " decanter"], name='target')
stim_rptd: pd.Series = \
    pd.Series([" pillow", " measuringcups"], name='repeated')
stim_foil: pd.Series = \
    pd.Series([" boot", " nuts", " silverware", " apples", " tweezers", " dvd", " detergent", " darts", " pens",
               " fabricsoftener", " pills", " skirt", " racket", " gift", " watch"], name='foil')
stim_all: pd.Series = \
    pd.Series(pd.concat([stim_targ, stim_rptd, stim_foil], ignore_index=True), name='all')


#############################
# Data Processing Functions #

def hstack_stats_dfs(df_x: pd.DataFrame,
                     type_str: str) -> pd.DataFrame:
    """
    Join sum DF, mean DF, median DF horizontally
    """
    if type_str == "recall":
        summ_cols = summ_cols_recall
        obj_str = " object"
    elif type_str == "recogn":
        summ_cols = summ_cols_recogn
        obj_str = " obj"
    else:
        raise ValueError("`type_str` arg in `hstack_stats_dfs()` needs to be either \"recall\" or \"recogn\"")

    df_sum = derive_df_x_sum(df_x, summ_cols, obj_str)
    df_mean = derive_df_x_mean(df_x, summ_cols, obj_str)
    df_median = derive_df_x_median(df_x, summ_cols, obj_str)

    return pd.concat([df_sum, df_mean, df_median], axis='columns')


def derive_df_x_sum(df_x: pd.DataFrame,
                    summ_cols: List[str],
                    obj_str: str) -> pd.DataFrame:
    """
    Combine sum Series into a DataFrame
    """
    s_sum_targ = derive_s_x_grp_sum(df_x, summ_cols, obj_str, stim_targ)
    s_sum_rptd = derive_s_x_grp_sum(df_x, summ_cols, obj_str, stim_rptd)
    s_sum_foil = derive_s_x_grp_sum(df_x, summ_cols, obj_str, stim_foil)
    s_sum_all = derive_s_x_grp_sum(df_x, summ_cols, obj_str, stim_all)

    return pd.concat([s_sum_targ, s_sum_rptd, s_sum_foil, s_sum_all], axis='columns').T


def derive_df_x_mean(df_x: pd.DataFrame,
                     summ_cols: List[str],
                     obj_str: str) -> pd.DataFrame:
    """
    Combine mean Series into a DataFrame
    """
    s_mean_targ = derive_s_x_grp_mean(df_x, summ_cols, obj_str, stim_targ)
    s_mean_rptd = derive_s_x_grp_mean(df_x, summ_cols, obj_str, stim_rptd)
    s_mean_foil = derive_s_x_grp_mean(df_x, summ_cols, obj_str, stim_foil)
    s_mean_all = derive_s_x_grp_mean(df_x, summ_cols, obj_str, stim_all)

    return pd.concat([s_mean_targ, s_mean_rptd, s_mean_foil, s_mean_all], axis='columns').T


def derive_df_x_median(df_x: pd.DataFrame,
                       summ_cols: List[str],
                       obj_str: str) -> pd.DataFrame:
    """
    Combine median Series into a DataFrame
    """
    s_median_targ = derive_s_x_grp_median(df_x, summ_cols, obj_str, stim_targ)
    s_median_rptd = derive_s_x_grp_median(df_x, summ_cols, obj_str, stim_rptd)
    s_median_foil = derive_s_x_grp_median(df_x, summ_cols, obj_str, stim_foil)
    s_median_all = derive_s_x_grp_median(df_x, summ_cols, obj_str, stim_all)

    return pd.concat([s_median_targ, s_median_rptd, s_median_foil, s_median_all], axis='columns').T


def derive_s_x_grp_sum(df_x: pd.DataFrame,
                       summ_cols: List[str],
                       obj_str: str,
                       stim_group: pd.Series) -> pd.Series:
    """
    Calculate the sum of the given DF for a given group
    """
    # Row-wise Boolean Series for rows in the given group
    is_stim_group = df_x[obj_str].isin(stim_group)

    s_x_grp_sum = df_x[is_stim_group][summ_cols].sum()
    s_x_grp_sum.name = stim_group.name
    s_x_grp_sum.index = [idx + " sum" for idx in s_x_grp_sum.index]

    return s_x_grp_sum


def derive_s_x_grp_mean(df_x: pd.DataFrame,
                        summ_cols: List[str],
                        obj_str: str,
                        stim_group: pd.Series) -> pd.Series:
    """
    Calculate the mean of the given DF for a given group
    """
    # Row-wise Boolean Series for rows in the given group
    is_stim_group = df_x[obj_str].isin(stim_group)

    s_x_grp_mean = df_x[is_stim_group][summ_cols].mean()
    s_x_grp_mean.name = stim_group.name
    s_x_grp_mean.index = [idx + " mean" for idx in s_x_grp_mean.index]

    return s_x_grp_mean


def derive_s_x_grp_median(df_x: pd.DataFrame,
                          summ_cols: List[str],
                          obj_str: str,
                          stim_group: pd.Series) -> pd.Series:
    """
    Calculate the median of the given DF for a given group
    """
    # Row-wise Boolean Series for rows in the given group
    is_stim_group = df_x[obj_str].isin(stim_group)

    s_x_grp_median = df_x[is_stim_group][summ_cols].median()
    s_x_grp_median.name = stim_group.name
    s_x_grp_median.index = [idx + " median" for idx in s_x_grp_median.index]

    return s_x_grp_median


########################
# Box Client Functions #

def get_authenticated_client(config_path):
    """Get an authenticated Box client for a JWT service account

    Arguments:
        configPath {str} -- Path to the JSON config file for your Box JWT app

    Returns:
        Client -- A Box client for the JWT service account

    Raises:
        ValueError -- if the configPath is empty or cannot be found.
    """
    if not os.path.isfile(config_path):
        raise ValueError(f"configPath must be a path to the JSON config file for your Box JWT app")
    auth = JWTAuth.from_settings_file(config_path)
    print("Authenticating...")
    auth.authenticate_instance()
    return Client(auth)


def get_subitems(client, folder, fields=["id", "name", "path_collection", "size"]):
    """Get a collection of all immediate folder items

    Arguments:
        client {Client} -- An authenticated Box client
        folder {Folder} -- The Box folder whose contents we want to fetch

    Keyword Arguments:
        fields {list} -- An optional list of fields to include with each item (default: {["id","name","path_collection"]})

    Returns:
        list -- A collection of Box files and folders.
    """
    items = []
    # fetch folder items and add subfolders to list
    for item in client.folder(folder_id=folder['id']).get_items(fields=fields):
        items.append(item)
    return items


def print_user_info(client):
    """Print the name and login of the current authenticated Box user

    Arguments:
        client {Client} -- An authenticated Box client
    """
    user = client.user('me').get()
    print("")
    print("Authenticated User")
    print(f"Name: {user.name}")
    print(f"Login: {user.login}")


########################################
# File Handling / Processing Functions #

def csv_file_id_to_df_raw(client, file_id, cols):
    """Get raw DataFrame using Box auth'd client and file ID

    :param client:
    :param file_id:
    :param cols:
    :return:
    """
    # Get file contents with auth'd Box client
    file_content = client.file(file_id).content()
    df_raw = pd.DataFrame()
    # Read file contents stream as DataFrame
    if cols == cols_recall:
        df_raw = pd.read_csv(io.BytesIO(file_content),
                             skiprows=4, usecols=cols,
                             dtype={'trial num': 'Int64', ' object': str, ' env': str,
                                    ' target X': 'Int64', ' target Y': 'Int64',
                                    ' response X': 'Int64', ' response Y': 'Int64',
                                    ' deltatime': 'Int64', }).dropna()
    elif cols == cols_recogn:
        df_raw = pd.read_csv(io.BytesIO(file_content),
                             skiprows=4, usecols=cols,
                             dtype={'trial num': 'Int64', ' env': str, ' obj': str, ' location': str,
                                    ' location chosen': str, ' time': 'Int64'}).dropna()

    return df_raw


# https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html?highlight=excelfile#writing-excel-files-to-memory
def dfs_to_excel_buffer(dict_dfs):
    # Instantiate a bytes buffer stream
    bio = io.BytesIO()
    # Instantiate and set the engine of ExcelWriter constructor
    with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
        # Write each DataFrame as a sheet in the Excel file to the ExcelWriter object
        for k, df in dict_dfs.items():
            df.to_excel(writer, sheet_name=k)
    # Save the ExcelWriter object contents to the bytes buffer stream
    writer.save()

    return bio


###################
# Driver Function #

def walk_dir_tree_process_files(client, folder,
                                ptrn_freercl, ptrn_cuedrcl, ptrn_recognt, ptrn_sumstat,
                                overwrite=False):
    """Count the number of files matching the regex pattern X

    Arguments:
        client {Client} -- An authenticated Box client
        folder {Folder} -- The Box folder whose contents we want to fetch
        ptrn_freercl {RegexObject} -- Regex pattern for "Free Recall" file
        ptrn_cuedrcl {RegexObject} -- Regex pattern for "Cued Recall" file
        ptrn_recognt {RegexObject} -- Regex pattern for "Recognition" file
        ptrn_sumstat {RegexObject} -- Regex pattern for "OLTT_11a_Grouped_Summary_Stats" file created by this script
        overwrite {Boolean} -- Flag determing whether to overwrite existing "OLTT_11a_Grouped_Summary_Stats" files
    """

    # Local variables for function control flow
    file_id_freercl, file_id_cuedrcl, file_id_recognt = "", "", ""
    bool_freercl_exists, bool_cuedrcl_exists, bool_recognt_exists = False, False, False
    file_id_sumstat = ""
    bool_sumstat_exists = False

    # Get list of items in current folder
    subitems = get_subitems(client, folder)

    # Recurse down with this function into every subfolder
    for subfolder in filter(lambda i: i.type == "folder", subitems):
        walk_dir_tree_process_files(client, subfolder,
                                    ptrn_freercl, ptrn_cuedrcl, ptrn_recognt, ptrn_sumstat,
                                    overwrite=overwrite)

    # Set file existence flags necessary for carrying data processing work
    for file in filter(lambda i: i.type == "file", subitems):
        if re.match(ptrn_freercl, file.name):
            file_id_freercl = file.id
            bool_freercl_exists = True
        if re.match(ptrn_cuedrcl, file.name):
            file_id_cuedrcl = file.id
            bool_cuedrcl_exists = True
        if re.match(ptrn_recognt, file.name):
            file_id_recognt = file.id
            bool_recognt_exists = True
        if re.match(ptrn_sumstat, file.name):
            file_id_sumstat = file.id
            bool_sumstat_exists = True

    if bool_freercl_exists and bool_cuedrcl_exists and bool_recognt_exists:
        print(folder.name, folder.id)

        # Read CSVs
        df_freercl_raw = csv_file_id_to_df_raw(client, file_id_freercl, cols_recall)
        df_cuedrcl_raw = csv_file_id_to_df_raw(client, file_id_cuedrcl, cols_recall)
        df_recognt_raw = csv_file_id_to_df_raw(client, file_id_recognt, cols_recogn)

        # Drive summary stats from CSVs
        df_freercl_stats = hstack_stats_dfs(df_freercl_raw, "recall")
        df_cuedrcl_stats = hstack_stats_dfs(df_cuedrcl_raw, "recall")
        df_recognt_stats = hstack_stats_dfs(df_recognt_raw, "recogn")

        # Write DataFrames to Excel file buffer
        dict_dfs_stats = {'freercl': df_freercl_stats, 'cuedrcl': df_cuedrcl_stats, 'recognt': df_recognt_stats}
        bio = dfs_to_excel_buffer(dict_dfs_stats)

        # Upload the bytes io stream as Excel to folder
        if bool_sumstat_exists and overwrite:
            file_sumstat = client.file(file_id_sumstat)
            file_info_sumstat = file_sumstat.get()
            print("Overwriting file:", file_info_sumstat.name, file_info_sumstat.id)
            file_sumstat.update_contents_with_stream(bio)
        # elif bool_sumstat_exists and not overwrite:
        #     print("File already exists:", folder.name + "OLTT_11a_Grouped_Summary_Stats.xlsx", file_id_sumstat)
        elif not bool_sumstat_exists:
            print("Uploading", folder.name + "-OLTT_11a_Grouped_Summary_Stats.xlsx")
            folder.upload_stream(bio, folder.name + "-OLTT_11a_Grouped_Summary_Stats.xlsx")
