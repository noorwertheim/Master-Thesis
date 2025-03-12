import scipy.io
import numpy as np
import json
from typing import Dict


class EHGRecord:
    """
    Represents an EHG record of one person.

    Attributes
    ----------
    record_name : str
        Path to the EHG record
    mat_file : str
        Returns dictionary with variable names as keys, and loaded matrices as values.
    header : np.array
        Header containing all the metadata
    polarization_voltage : numpy.ndarray
        Lead polarization voltage for each electrode (Ref, 1, 2, 3). Consists of four channels.
        Polarization is the accumulation of waste on the cathode of the battery which blocks the
        movement of charges hence reducing the efficiency of the battery.
    acc_signals : numpy.ndarray
        3-axis accelerometer data
    clipping_signals : numpy.ndarray
        Clipping indicator for each EHG channel.
    ehg_signals : numpy.ndarray
        3-channel EHG data. Measured in microvolt (muV).
    specs_dict : dict
        Dictionary containing all the metadata present in the header.
    fs_ehg : float
        Sampling frequency of the EHG signals.
    fs_acc : float
        Sampling frequency of the acc data.
    fs_clipping : float
        Sampling frequency of the clipping indicator.
    num_channels_ehg : int
        Number of channels of the EHG signals.
    num_channels_acc : int
        Number of channels of the acc data.
    num_channels_clipping : int
        Number of channels of the clipping indicator.
    sig_len_ehg : int
        Length of the EHG signal data, i.e., the total number of data points for each channel.
    sig_len_acc : int
        Length of the acc data, i.e., the total number of data points for each channel.
    sig_len_clipping : int
        Length of the clipping indicator data, i.e., the total number of data points for each channel.
    unit_ehg : str
        Unit of the EHG signals.
    unit_acc : str
        Unit of the acc data.

    Methods
    -------
    get_spec(fs_dict, name):
        Gets the specification present in the metadata. For example: get the sampling frequency of the EHG signals.
    create_specs_dict():
        Creates the specification dictionary of all relevant data in the metadata. Such as the sampling frequency,
        unit, number of channels, etc.
    create_fourth_channel():
        Derive fourth EHG channel from the other three channels that are present.
    create_fifth_channel():
        Derive fifth EHG channel from the other three channels that are present.
    create_sixth_channel():
        Derive sixth EHG channel from the other three channels that are present.
    calculate_recording_duration_in_minutes():
        Calculate original duration of EHG recording in minutes.
    """
    def __init__(self, record_name):
        self.record_name = record_name
        self.mat_file = scipy.io.loadmat(self.record_name)
        self.header = self.mat_file['header']
        self.polarization_voltage = self.mat_file['polarization_voltage_sam4sd32c_adc']
        self.acc_signals = self.mat_file['acceleration_adxl362']
        self.clipping_signals = self.mat_file['clipping_afe2q']

        self.ehg_signals = self.mat_file['exg_afe2q']

        self.ehg_signals = self.create_fourth_channel()
        self.ehg_signals = self.create_fifth_channel()
        self.ehg_signals = self.create_sixth_channel()

        self.specs_dict = self.create_specs_dict()

        self.fs_ehg = self.get_spec(self.specs_dict, 'fs_ehg')
        self.fs_acc = self.get_spec(self.specs_dict, 'fs_acc')
        self.fs_clipping = self.get_spec(self.specs_dict, 'fs_clipping')

        self.num_channels_ehg = self.get_spec(self.specs_dict, 'num_channels_ehg')
        self.num_channels_acc = self.get_spec(self.specs_dict, 'num_channels_acc')
        self.num_channels_clipping = self.get_spec(self.specs_dict, 'num_channels_clipping')

        self.sig_len_ehg = self.get_spec(self.specs_dict, 'sig_len_ehg')
        self.sig_len_acc = self.get_spec(self.specs_dict, 'sig_len_acc')
        self.sig_len_clipping = self.get_spec(self.specs_dict, 'sig_len_clipping')

        self.unit_ehg = self.get_spec(self.specs_dict, 'unit_ehg')
        self.unit_acc = self.get_spec(self.specs_dict, 'unit_acc')

        self.orig_recording_duration = self.calculate_recording_duration_in_minutes()

    def get_spec(self, fs_dict: Dict, name: str):
        """Return the requested specification from the dictionary"""
        return fs_dict[name]

    def create_specs_dict(self):
        """Create a dictionary with the specifications from the header file, which contains the metadata.
        An example of the metadata: sample rate of ehg signals data, unit of the accelerometer data.
        """
        header_dict = json.loads(self.header[0])
        signals_specs_list = header_dict['data_file_header']['payload_info']['signals']
        specs_dict = {}
        for i, signal_spec in enumerate(signals_specs_list):
            signals_dict = signals_specs_list[i]

            # Specifications of the EHG data
            if signals_dict.get('name') == 'exg':
                fs_ehg_dict = signals_dict['sample_rate']
                fs_ehg = fs_ehg_dict['numerator'] / fs_ehg_dict['denominator']
                specs_dict['fs_ehg'] = fs_ehg
                specs_dict['num_channels_ehg'] = signals_dict['number_of_channels']
                specs_dict['sig_len_ehg'] = len(self.ehg_signals)
                specs_dict['unit_ehg'] = signals_dict['unit']

            # Specifications of the acc data
            elif signals_dict.get('name') == 'acceleration':
                fs_acc_dict = signals_dict['sample_rate']
                fs_acc = fs_acc_dict['numerator'] / fs_acc_dict['denominator']
                specs_dict['fs_acc'] = fs_acc
                specs_dict['num_channels_acc'] = signals_dict['number_of_channels']
                specs_dict['sig_len_acc'] = len(self.acc_signals)
                specs_dict['unit_acc'] = signals_dict['unit']

            # Specifications of the clipping indicator data
            elif signals_dict.get('name') == 'clipping':
                fs_clipping_dict = signals_dict['sample_rate']
                fs_clipping = fs_clipping_dict['numerator'] / fs_clipping_dict['denominator']
                specs_dict['fs_clipping'] = fs_clipping
                specs_dict['num_channels_clipping'] = signals_dict['number_of_channels']
                specs_dict['sig_len_clipping'] = len(self.clipping_signals)

        return specs_dict

    def create_fourth_channel(self):
        """
        Derive the fourth channel from channel 3 and 1. This channel will represent the
        difference in action potentials between the yellow and red electrode. The channel will be
        added to the already present ehg signals.
        """
        # Channel 4 is derived from channel 3 and 1
        channel4 = self.ehg_signals[:, 2] - self.ehg_signals[:, 0]
        self.ehg_signals = np.concatenate((self.ehg_signals, channel4.reshape(-1, 1)), axis=1)

        return self.ehg_signals

    def create_fifth_channel(self):
        """
        Derive the fifth channel from channel 2 and 1. This channel will represent the
        difference in action potentials between the green and red electrode. The channel will be
        added to the already present ehg signals.
        """
        # Channel 5 is derived from channel 2 and 1
        channel5 = self.ehg_signals[:, 1] - self.ehg_signals[:, 0]
        self.ehg_signals = np.concatenate((self.ehg_signals, channel5.reshape(-1, 1)), axis=1)

        return self.ehg_signals

    def create_sixth_channel(self):
        """
        Derive the sixth channel from channel 3 and 2. This channel will represent the
        difference in action potentials between the yellow and green electrode. The channel will be
        added to the already present ehg signals.
        """
        # Channel 6 is derived from channel 3 and 2
        channel6 = self.ehg_signals[:, 2] - self.ehg_signals[:, 1]
        self.ehg_signals = np.concatenate((self.ehg_signals, channel6.reshape(-1, 1)), axis=1)

        return self.ehg_signals

    def calculate_recording_duration_in_minutes(self):
        """Calculate original duration of the recording in minutes. """
        original_recording_duration = self.sig_len_ehg / (self.fs_ehg * 60)

        return original_recording_duration