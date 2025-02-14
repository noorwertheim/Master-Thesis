from src_cocoon_project.utils import read_settings
from src_cocoon_project.preprocess import butter_bandpass_filterab
import pandas as pd
import os
import wfdb
from tqdm import tqdm
from pathlib import Path
from typing import List, Union
import re
import numpy as np


settings_path = '/Users/AFischer/PycharmProjects/cocoon-project-new/references/settings'

file_paths = read_settings(settings_path, 'file_paths')

TPEHG_DATA_PATH = file_paths['tpehg_data_path']
TPEHGT_DATA_PATH = file_paths['tpehgt_data_path']
ICEHG_DS_DATA_PATH = file_paths['icehg_ds_data_path']
ICELANDIC_DATA_PATH = file_paths['icelandic_data_path']

COCOON_EHG_DATA_PATH = file_paths['cocoon_ehg_data_path']
CLINICAL_DATA_PATH = file_paths['cocoon_clinical_data_path']

SOURCE_DATASETS_PATH = file_paths['source_datasets_path']
OUTPUT_PATH = '/Users/AFischer/Documents/PhD_onderzoek/data_cocoon_studie/csv_files'
SAVED_MODELS_PATH = '/Users/AFischer/Documents/PhD_onderzoek/transfer_learning/saved_models'


def create_file_list_tpehg_database(path_to_signals: Union[str, Path]) -> List:
    """Create a list of all the record ids from the term-preterm database
    from PhysioNet.

    Parameters
    ----------
    path_to_signals : str
        Path to folder with EHG signal patient files.

    Returns
    -------
    filelist : List
        List containing all the record ids from the term preterm database.
    """
    # The data comes from PhysioNet and in order to read in the data
    # we need to use the WFDB python package.
    # The name of the WFDB record (record id) that has to be read must not contain any file extensions,
    # therefore we will strip of the .dat and .hea extension.
    filelist = os.listdir(path_to_signals)  # this list contains all wfdb records (including extension)
    filelist = [i.replace('.dat', "").replace('.hea', '') for i in filelist]  # strip of .dat and .hea

    # The final list will contain all record ids from the database (without the file extensions).
    filelist = list(dict.fromkeys(filelist))

    return filelist


def build_tpehg_dataframe(path_to_data: str) -> pd.DataFrame:
    """Loop over all patient EHG signal files from the TPEHG database and combine all EHG recordings into
    one dataframe and save it. Source: https://physionet.org/content/tpehgdb/1.0.1/

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.

    Returns
    -------
    df_final_signals : pd.DataFrame
        Dataframe that contains all EHG signals from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')

    # This filelist contains all the record ids from the database.
    filelist = create_file_list_tpehg_database(data_path)

    SIGNAL_COLUMN_NAMES = ['1', '1_DOCFILT-4-0.08-4', '1_DOCFILT-4-0.3-3', '1_DOCFILT-4-0.3-4',
                           '2', '2_DOCFILT-4-0.08-4', '2_DOCFILT-4-0.3-3', '2_DOCFILT-4-0.3-4',
                           '3', '3_DOCFILT-4-0.08-4', '3_DOCFILT-4-0.3-3', '3_DOCFILT-4-0.3-4']

    # In this dataframe we will store the EHG signals from all WFDB records.
    df_final_signals = pd.DataFrame(columns=['rec_id'] + SIGNAL_COLUMN_NAMES)

    for id_file in tqdm(filelist):
        # In the signals variable, the EHG data from all channels are saved and in
        # the fields variable all the record descriptors (e.g., age, gestation, etc.)
        # of the patients is stored.
        signals, fields = wfdb.rdsamp(f'{data_path}/{id_file}')
        df_signals = pd.DataFrame(signals, columns=SIGNAL_COLUMN_NAMES)

        # As we also want to know which record id belongs to which EHG data, we also save the id_file
        # in df_signals.
        df_signals = pd.concat([pd.Series([id_file] * len(signals)).to_frame('rec_id'), df_signals], axis=1)

        df_final_signals = pd.concat([df_final_signals, df_signals], ignore_index=True)

    df_final_signals = split_rec_id(df_final_signals, prefix='tpehg')
    df_final_signals.loc[:, 'rec_id'] = pd.to_numeric(df_final_signals['rec_id'])

    return df_final_signals


def create_file_list_tpehgt_database(path_to_signals: Union[str, Path]) -> List:
    """Create a list of all the record ids from the Term-Preterm EHG DataSet with Tocogram
    from PhysioNet. Source: https://www.physionet.org/content/tpehgt/1.0.0/

    Parameters
    ----------
    path_to_signals : str
        Path to folder with EHG signal patient files.

    Returns
    -------
    filelist : List
        List containing all the record ids from the Term-Preterm EHG DataSet with Tocogram database.
    """
    # The data comes from PhysioNet and in order to read in the data
    # we need to use the WFDB python package.
    # The name of the WFDB record (record id) that has to be read must not contain any file extensions,
    # therefore we will strip of the .dat and .hea extension.
    filelist = os.listdir(path_to_signals)  # this list contains all wfdb records (including extension)

    filelist = [file for file in filelist if 'tpehgt_p' in file or 'tpehgt_t' in file]

    # strip of .dat, .hea and .atr
    filelist = [i.replace('.dat', "").replace('.hea', '').replace('.atr', '') for i in filelist]

    # The final list will contain all record ids from the database (without the file extensions).
    filelist = list(dict.fromkeys(filelist))

    return filelist


def create_file_list_icehg_ds_database(path_to_signals: Union[str, Path]) -> tuple[List[str]]:
    """Create a list of all the record ids from the Induced Cesarean EHG DataSet (ICEHG DS)
    from PhysioNet. Source: https://www.physionet.org/content/icehg-ds/1.0.1/

    Parameters
    ----------
    path_to_signals : str
        Path to folder with EHG signal patient files.

    Returns
    -------
    filelist : List
        List containing all the record ids from the Term-Preterm EHG DataSet with Tocogram database.
    """
    # The data comes from PhysioNet and in order to read in the data
    # we need to use the WFDB python package.

    # There six folders containing EHG records and we will make 1 filelist containing the names of all records

    # this list contains all wfdb records (including extension)
    filelist_early_cesarean = os.listdir(os.path.join(path_to_signals, 'early_cesarean'))
    filelist_early_induced = os.listdir(os.path.join(path_to_signals, 'early_induced'))
    filelist_early_induced_cesarean = os.listdir(os.path.join(path_to_signals, 'early_induced-cesarean'))
    filelist_later_cesarean = os.listdir(os.path.join(path_to_signals, 'later_cesarean'))
    filelist_later_induced = os.listdir(os.path.join(path_to_signals, 'later_induced'))
    filelist_later_induced_cesarean = os.listdir(os.path.join(path_to_signals, 'later_induced-cesarean'))

    # The name of the WFDB record (record id) that has to be read must not contain any file extensions,
    # therefore we will strip of the .dat and .hea extension.
    # strip of .dat, .hea and .atr
    filelist_early_cesarean = [i.replace('.dat', "").replace('.hea', '').replace('_fltrd.jpg', '') for i in filelist_early_cesarean]
    filelist_early_induced = [i.replace('.dat', "").replace('.hea', '').replace('_fltrd.jpg', '') for i in filelist_early_induced]
    filelist_early_induced_cesarean = [i.replace('.dat', "").replace('.hea', '').replace('_fltrd.jpg', '') for i in filelist_early_induced_cesarean]
    filelist_later_cesarean = [i.replace('.dat', "").replace('.hea', '').replace('_fltrd.jpg', '') for i in filelist_later_cesarean]
    filelist_later_induced = [i.replace('.dat', "").replace('.hea', '').replace('_fltrd.jpg', '') for i in filelist_later_induced]
    filelist_later_induced_cesarean = [i.replace('.dat', "").replace('.hea', '').replace('_fltrd.jpg', '') for i in filelist_later_induced_cesarean]

    # The final list will contain all record ids from the database (without the file extensions).
    filelist_early_cesarean = list(dict.fromkeys(filelist_early_cesarean))
    filelist_early_induced = list(dict.fromkeys(filelist_early_induced))
    filelist_early_induced_cesarean = list(dict.fromkeys(filelist_early_induced_cesarean))
    filelist_later_cesarean = list(dict.fromkeys(filelist_later_cesarean))
    filelist_later_induced = list(dict.fromkeys(filelist_later_induced))
    filelist_later_induced_cesarean = list(dict.fromkeys(filelist_later_induced_cesarean))

    return filelist_early_cesarean, filelist_early_induced, filelist_early_induced_cesarean, filelist_later_cesarean, \
           filelist_later_induced, filelist_later_induced_cesarean


def create_file_list_icelandic_database(path_to_signals: Union[str, Path]) -> List:
    """Create a list of all the record ids from the Icelandic 16-electrode Electrohysterogram Database
    from PhysioNet. Source: https://www.physionet.org/content/ehgdb/1.0.0/

    Parameters
    ----------
    path_to_signals : str
        Path to folder with EHG signal patient files.

    Returns
    -------
    filelist : List
        List containing all the record ids from the Icelandic 16-electrode Electrohysterogram Database.
    """
    # The data comes from PhysioNet and in order to read in the data
    # we need to use the WFDB python package.
    # The name of the WFDB record (record id) that has to be read must not contain any file extensions,
    # therefore we will strip of the .dat and .hea extension.
    filelist = os.listdir(path_to_signals)  # this list contains all wfdb records (including extension)

    # Obtain only the files that start with 'ice' followed by an integer, e.g., ice001.
    filelist = [file for file in filelist if re.search('ice\d+', file)]

    # strip of .dat, .hea and .atr
    filelist = [i.replace('.dat', "").replace('.hea', '').replace('.atr', '').replace('.jpg', '') for i in filelist]

    # The final list will contain all record ids from the database (without the file extensions).
    filelist = list(dict.fromkeys(filelist))

    return filelist


def split_rec_id(df: pd.DataFrame, prefix: str):
    """Split the rec_id column such that it only contains the integer
    and not the prefix 'tpehg'. Example: Rec ids that have the form
    tpehg1007, will be split such that the returned rec id will be just
    1007.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the column 'rec_id'.

    prefix : str
        Prefix within the column 'rec_id'.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the column rec_id that only contains the integer.
    """
    # New data frame where the rec_id is split into 'tpehg' and the corresponding
    # integer
    if prefix == 'tpehg':
        new_df = df['rec_id'].str.split("tpehg", n=1, expand=True)
    elif prefix == 'icehg':
        new_df = df['rec_id'].str.split("icehg", n=1, expand=True)
    elif prefix == 'ice':
        new_df = df['rec_id'].str.split("ice", n=1, expand=True)

    # We need to make sure that the replacement of the rec_ids is correct. If the length
    # of the new rec_ids does not match the length of the old rec_ids we know something
    # went wrong.
    assert (len(df['rec_id']) == len(new_df[1])), "The rec_ids are not properly split!"

    # The second column of the new dataframe contains the integer, and we replace
    # the old 'rec_id' of df with the new 'rec_id' that only consists of the integer.
    df['rec_id'] = new_df[1]

    return df


def build_tpehgt_dataframe(path_to_data: str) -> pd.DataFrame:
    """Loop over all patient EHG signal files from the TPEHGT dataset and combine all EHG recordings into
    one dataframe and save it.

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.

    Returns
    -------
    df_final_signals : pd.DataFrame
        Dataframe that contains all EHG signals from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')

    # This filelist contains all the record ids from the database.
    filelist = create_file_list_tpehgt_database(data_path)

    SIGNAL_COLUMN_NAMES = ['EHG1', 'EHG1_Butter-4-bi-0.08-5', 'EHG2', 'EHG2_Butter-4-bi-0.08-5',
                           'EHG3', 'EHG3_Butter-4-bi-0.08-5', 'TOCO', 'TOCO_Butter-4-bi-0.08-5']

    # In this dataframe we will store the EHG signals from all WFDB records.
    df_final_signals = pd.DataFrame(columns=['rec_id'] + SIGNAL_COLUMN_NAMES)

    for id_file in tqdm(filelist):
        # In the signals variable, the EHG data from all channels are saved and in
        # the fields variable all the record descriptors (e.g., age, gestation, etc.)
        # of the patients is stored.
        signals, fields = wfdb.rdsamp(f'{data_path}/{id_file}')
        df_signals = pd.DataFrame(signals, columns=SIGNAL_COLUMN_NAMES)

        # As we also want to know which record id belongs to which EHG data, we also save the id_file
        # in df_signals.
        df_signals = pd.concat([pd.Series([id_file] * len(signals)).to_frame('rec_id'), df_signals], axis=1)

        df_final_signals = pd.concat([df_final_signals, df_signals], ignore_index=True)

    df_final_signals = split_rec_id(df_final_signals, prefix='tpehg')

    return df_final_signals


def build_icehg_ds_dataframe(path_to_data: str) -> pd.DataFrame:
    """Loop over all patient EHG signal files from the ICEHG DS dataset and combine all EHG recordings into
    one dataframe and save it.

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.

    Returns
    -------
    df_final_signals : pd.DataFrame
        Dataframe that contains all EHG signals from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')

    # This filelist contains all the record ids from the database.
    filelist_early_cesarean, filelist_early_induced, filelist_early_induced_cesarean, filelist_later_cesarean, \
    filelist_later_induced, filelist_later_induced_cesarean = create_file_list_icehg_ds_database(data_path)

    final_list = [filelist_early_cesarean, filelist_early_induced, filelist_early_induced_cesarean,
                  filelist_later_cesarean, filelist_later_induced, filelist_later_induced_cesarean]

    final_list_names = ['early_cesarean', 'early_induced', 'early_induced-cesarean',
                        'later_cesarean', 'later_induced', 'later_induced-cesarean']

    SIGNAL_COLUMN_NAMES = ['S1', 'S1_DOCFILT-4-0.08-5', 'S2', 'S2_DOCFILT-4-0.08-5',
                           'S3', 'S3_DOCFILT-4-0.08-5']

    df_final_signals = pd.DataFrame(columns=['rec_id'] + SIGNAL_COLUMN_NAMES)
    for filelist, filelist_name in zip(final_list, final_list_names):
        # In this dataframe we will store the EHG signals from all WFDB records.
        df_signals_per_group = pd.DataFrame(columns=['rec_id'] + SIGNAL_COLUMN_NAMES)
        for id_file in filelist:
            # In the signals variable, the EHG data from all channels are saved and in
            # the fields variable all the record descriptors (e.g., age, gestation, etc.)
            # of the patients is stored.
            signals, fields = wfdb.rdsamp(f'{os.path.join(data_path, filelist_name)}/{id_file}')
            df_signals = pd.DataFrame(signals, columns=SIGNAL_COLUMN_NAMES)

            # As we also want to know which record id belongs to which EHG data, we also save the id_file
            # in df_signals.
            df_signals = pd.concat([pd.Series([id_file] * len(signals)).to_frame('rec_id'), df_signals], axis=1)

            df_signals_per_group = pd.concat([df_signals_per_group, df_signals], ignore_index=True)

        df_final_signals = pd.concat([df_final_signals, df_signals_per_group], ignore_index=True)

    df_final_signals = split_rec_id(df_final_signals, prefix='icehg')

    return df_final_signals


def build_icelandic_dataframe(path_to_data: str) -> pd.DataFrame:
    """Loop over all patient EHG signal files from the TPEHGT dataset and combine all EHG recordings into
    one dataframe and save it.

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.

    Returns
    -------
    df_final_signals : pd.DataFrame
        Dataframe that contains all EHG signals from all record ids.
    """
    # Create a Path object in order to convert to the correct OS path.
    data_path = Path(f'{path_to_data}')

    # This filelist contains all the record ids from the database.
    filelist = create_file_list_icelandic_database(data_path)

    SIGNAL_COLUMN_NAMES = ['EHG1', 'EHG10', 'EHG11', 'EHG12', 'EHG13', 'EHG14', 'EHG15', 'EHG16', 'EHG2', 'EHG3',
                           'EHG4', 'EHG5', 'EHG6', 'EHG7', 'EHG8', 'EHG9']

    # In this dataframe we will store the EHG signals from all WFDB records.
    df_final_signals = pd.DataFrame(columns=['rec_id'] + SIGNAL_COLUMN_NAMES)

    for id_file in tqdm(filelist):
        # In the signals variable, the EHG data from all channels are saved and in
        # the fields variable all the record descriptors (e.g., age, gestation, etc.)
        # of the patients is stored.
        signals, fields = wfdb.rdsamp(f'{data_path}/{id_file}')
        df_signals = pd.DataFrame(signals, columns=SIGNAL_COLUMN_NAMES)

        # As we also want to know which record id belongs to which EHG data, we also save the id_file
        # in df_signals.
        df_signals = pd.concat([pd.Series([id_file] * len(signals)).to_frame('rec_id'), df_signals], axis=1)

        df_final_signals = pd.concat([df_final_signals, df_signals], ignore_index=True)

    return df_final_signals


def create_filtered_channels(df_signals: pd.DataFrame, list_of_channels: List[str],
                             list_of_bandwidths: List[List[np.array]],
                             fs: int = 20, order: int = 4) -> pd.DataFrame:
    """Create filtered channels based on the Butterworth filtering scheme.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Original signal data.
    list_of_channels : List[str]
        List with the names of the channels you want to filter.
    list_of_bandwidths : List[List[np.array]]
        List containing the bandwidths (low cut and high cut) for which you want to filter.
        Example: [[0.08, 4], [0.3, 3]].
    fs : int
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    df_signals_new : pd.DataFrame
        Dataframe that contains both the original signals as the filtered signals of each channel.
    """
    # The filtered signals will be stored in df_signals_new and this df will be returned
    df_signals_new = pd.DataFrame()

    rec_ids = df_signals['rec_id'].unique()

    for rec_id in tqdm(rec_ids):

        df_tmp_rec_id = df_signals[df_signals['rec_id'] == rec_id].copy()
        df_filtered_signals = pd.DataFrame(df_tmp_rec_id[['rec_id']], columns=['rec_id'])

        for channel in list_of_channels:
            # The original (unfiltered) signal data is also added to the final dataframe
            df_filtered_signals = pd.concat([df_filtered_signals, df_tmp_rec_id[[f'{channel}']]], axis=1)

            for bandwidth in list_of_bandwidths:
                # We filter the signal data for each given bandwidth
                filtered_signals = butter_bandpass_filterab(data=df_tmp_rec_id[f'{channel}'],
                                                            lowcut=bandwidth[1],
                                                            highcut=bandwidth[0],
                                                            fs=fs,
                                                            order=order)

                index_tmp = df_tmp_rec_id[['rec_id']].index

                df_filtered_signals = pd.concat(
                    [df_filtered_signals, pd.DataFrame(filtered_signals,
                                                       columns=[f'{channel}_filt_{bandwidth[0]}_{bandwidth[1]}_hz'],
                                                       index=index_tmp)], axis=1)

        df_signals_new = pd.concat([df_signals_new, df_filtered_signals], ignore_index=True)

    return df_signals_new


df_tpehg = build_tpehg_dataframe(TPEHG_DATA_PATH)
df_tpehg_signals = create_filtered_channels(df_tpehg, list_of_channels=['1', '2', '3'],
                                            list_of_bandwidths=[[0.34, 1]], fs=20, order=4)
# We save this file so that we do not have to repeat the creation of dataset and filtering process each time
df_tpehg_signals.to_csv(os.path.join(SOURCE_DATASETS_PATH, 'df_tpehg_filtered.csv'))


df_tpehgt = build_tpehgt_dataframe(TPEHGT_DATA_PATH)
df_tpehgt_signals = create_filtered_channels(df_tpehgt, list_of_channels=['EHG1', 'EHG2', 'EHG3'],
                                             list_of_bandwidths=[[0.34, 1]], fs=20, order=4)
# We save this file so that we do not have to repeat the creation of dataset and filtering process each time
df_tpehgt_signals.to_csv(os.path.join(SOURCE_DATASETS_PATH, 'df_tpehgt_filtered.csv'))


df_icehg_ds = build_icehg_ds_dataframe(ICEHG_DS_DATA_PATH)
df_icehg_ds_signals = create_filtered_channels(df_icehg_ds, list_of_channels=['S1', 'S2', 'S3'],
                                               list_of_bandwidths=[[0.34, 1]], fs=20, order=4)
df_icehg_ds_signals.to_csv(os.path.join(SOURCE_DATASETS_PATH, 'df_icehg_ds_filtered.csv'))


# We save this file so that we do not have to repeat the creation of dataset and filtering process each time
df_icelandic = build_icelandic_dataframe(ICELANDIC_DATA_PATH)
df_icelandic.to_csv(os.path.join(SOURCE_DATASETS_PATH, 'df_icelandic.csv'))

df_icelandic = pd.read_csv(os.path.join(ICELANDIC_DATA_PATH, 'df_icelandic.csv'))

icelandic_rec_ids = df_icelandic['rec_id'].unique()

df_icelandic_1 = df_icelandic.loc[df_icelandic['rec_id'].isin(icelandic_rec_ids[0:61])].copy()
df_icelandic_1 = df_icelandic_1.reset_index(drop=True)
df_icelandic_2 = df_icelandic.loc[df_icelandic['rec_id'].isin(icelandic_rec_ids[61:])].copy()
df_icelandic_2 = df_icelandic_2.reset_index(drop=True)

# To save memory
del df_icelandic


df_icelandic_signals_1 = create_filtered_channels(df_icelandic_1, list_of_channels=['EHG1', 'EHG10', 'EHG11', 'EHG12',
                                                                                    'EHG13', 'EHG14', 'EHG15', 'EHG16',
                                                                                    'EHG2', 'EHG3', 'EHG4', 'EHG5',
                                                                                    'EHG6', 'EHG7', 'EHG8', 'EHG9'],
                                                  list_of_bandwidths=[[0.34, 1]], fs=200, order=4)

# We save this file so that we do not have to repeat the creation of dataset and filtering process each time
df_icelandic_signals_1.to_csv(os.path.join(SOURCE_DATASETS_PATH, 'df_icelandic_filtered_1.csv'))
# Save memory
del df_icelandic_1

df_icelandic_signals_2 = create_filtered_channels(df_icelandic_2, list_of_channels=['EHG1', 'EHG10', 'EHG11', 'EHG12',
                                                                                    'EHG13', 'EHG14', 'EHG15', 'EHG16',
                                                                                    'EHG2', 'EHG3', 'EHG4', 'EHG5',
                                                                                    'EHG6', 'EHG7', 'EHG8', 'EHG9'],
                                                  list_of_bandwidths=[[0.34, 1]], fs=200, order=4)

df_icelandic_signals_2.to_csv(os.path.join(SOURCE_DATASETS_PATH, 'df_icelandic_filtered_2.csv'))
# Save memory
del df_icelandic_2
