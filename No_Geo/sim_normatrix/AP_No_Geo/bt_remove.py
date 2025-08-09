import numpy as np
import os
import glob
import re
from typing import List


def bt_remove_index(target_sic_path: str,
                    smos_father_path: str,
                    sic_format: str) -> List[bool]:
    """
    Scan a directory of SIC files and flag which columns to remove.
    
    Parameters
    ----------
    target_year_path : str
        Path to the folder containing SIC files for one year, 
        e.g. "D:/data/2012/".
    smos_father_path : str
        Base directory where SMOS BT subfolders are organized by year, 
        e.g. "D:/SMOS/".
    sic_format : str
        File extension or pattern for SIC files, e.g. ".nc" or "_sic.txt".
    
    Returns
    -------
    remove_column : List[bool]
        A boolean list of length (N_sic_files + 2). The first two entries
        correspond to latitude/longitude columns (always False), and each
        subsequent entry corresponds to one SIC file in sorted order.
        An entry is True if the matching SMOS BT file for that date is missing.
    """
    # find all SIC files matching the pattern
    search_pattern = os.path.join(target_sic_path, f"*{sic_format}")
    sic_paths = sorted(glob.glob(search_pattern))
    
    n_files = len(sic_paths)
    # +2 for the lat/lon columns
    remove_column = [False] * (n_files + 2)
    
    date_regex = re.compile(r"\d{8}")  # match YYYYMMDD
    
    for idx, sic_path in enumerate(sic_paths):
        sic_name = os.path.basename(sic_path)
        m = date_regex.search(sic_name)
        if not m:
            # if filename doesn't contain an 8-digit date, skip
            continue
        
        sic_date = m.group()         # e.g. "20121015"
        year = sic_date[:4]          # e.g. "2012"
        
        # construct the expected SMOS BT file path
        smos_bt_dir = os.path.join(smos_father_path, "daily_merge")
        smos_bt_file = os.path.join(smos_bt_dir, f"{sic_date}.txt")
        
        # flag removal if the file does not exist
        if not os.path.isfile(smos_bt_file):
            remove_column[idx + 2] = True
    
    return remove_column