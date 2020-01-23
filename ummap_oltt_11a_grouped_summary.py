#!/usr/bin/env python


##################
# Import Modules #

import re
import argparse
from colored import fg, attr

import ummap_oltt_11a_grouped_summary_helpers as hlps


########
# Main #

def main():

    #####################
    # Print Color Setup #

    clr_blu = fg('blue')
    clr_bld = attr('bold')
    clr_wrn = fg('red') + attr('bold')
    clr_rst = attr('reset')

    ##############
    # Parse Args #

    parser = argparse.ArgumentParser(description="Process grouped summary statistics of OLTT Set 11A data.")
    parser.add_argument('-j', '--jwt_cfg', required=True,
                        help=f"{clr_bld}required{clr_rst}: absolute path to JWT config file")
    parser.add_argument('-b', '--box_folder_id', required=True,
                        help=f"{clr_bld}required{clr_rst}: destination Box Folder ID")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help=f"print actions to stdout")
    args = parser.parse_args()

    #################
    # Configuration #

    # Access args.verbose once
    is_verbose = args.verbose

    # Set the path to your JWT app config JSON file
    jwt_cfg_path = args.jwt_cfg
    if is_verbose:
        print(f"{clr_blu}Path to Box JWT config{clr_rst}:", f"{jwt_cfg_path}")

    # Set the path to the folder that will be traversed
    box_folder_id = args.box_folder_id

    ############################
    # Establish Box Connection #

    # Get authenticated Box client
    client = hlps.get_authenticated_client(jwt_cfg_path)

    # Create Box Folder object with authenticated client
    folder = client.folder(folder_id=box_folder_id).get()

    ##########################################
    # Define Regex Patterns for Target Files #

    # Regex patterns for CSV filenames with raw data
    ptrn_freercl = re.compile(r'^\d{3,4}-Free Recall-\w{14}\.csv$', re.IGNORECASE)
    ptrn_cuedrcl = re.compile(r'^\d{3,4}-Cued Recall-\w{14}\.csv$', re.IGNORECASE)
    ptrn_recognt = re.compile(r'^\d{3,4}-Recognition-\w{14}\.csv$', re.IGNORECASE)
    ptrn_sumstat = re.compile(r'^\d{3,4}-OLTT_11a_Grouped_Summary_Stats\.xlsx$')

    ##################################################
    # Recurse Through Directories to Summarize Stats #

    # Walk the directory subtree from defined Box Folder object to process
    # Free Recall, Cued Recall, and Recognition files
    hlps.walk_dir_tree_process_files(client, folder,
                                     ptrn_freercl, ptrn_cuedrcl, ptrn_recognt, ptrn_sumstat,
                                     overwrite=False)


if __name__ == "__main__":
    main()
