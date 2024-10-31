#! /usr/bin/env python
# _*_ coding: utf-8 -*-
# exportNWB.py
"""
exportNWB.py
Takes data from a dump.rdb and a graph to export it as an NWB file for
analysis in Python and MATLAB
@author Sam Nason-Tomaszewski, adapted for supervisor by Mattia Rigotti
"""

import argparse
import json
import logging
import numbers
import numpy as np
import os
import signal
import sys
import yaml

from brand.redis import RedisLoggingHandler

from datetime import datetime

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position

from redis import ConnectionError, Redis

###############################################
# Initialize script
###############################################

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--nickname', type=str, required=True)
ap.add_argument('-i', '--redis_host', type=str, required=True)
ap.add_argument('-p', '--redis_port', type=int, required=True)
ap.add_argument('-s', '--redis_socket', type=str, required=False)
args = ap.parse_args()

NAME = args.nickname
redis_host = args.redis_host
redis_port = args.redis_port
redis_socket = args.redis_socket

loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{NAME}] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)

def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(1)

# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)


###############################################
# Helper functions
###############################################
def add_stream_sync_timeseries(nwbfile, stream, time_data):
    """
    Creates an sync timeseries representing the
    system (monotonic), redis, and other sync times
    at which each stream had an entry
    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile in which to store the sync timeseries
    stream : str
        The name of the stream
    stream_data : dict
        The stream's data containing times
    """

    time_data_stack = {
        k: time_data[k]
        for k in time_data if k != 'sync_timestamps'
    }
    column_order_string = ','.join(
        [k for k in time_data.keys() if k != 'sync_timestamps'])

    sync_timeseries = TimeSeries(
        name=f'{stream}_ts',
        data=np.stack(list(time_data_stack.values()), axis=1),
        unit='seconds',
        timestamps=time_data['sync_timestamps'],
        comments=f'columns=[{column_order_string}]',
        description=f'Syncing timestamps for the {stream} stream')

    nwbfile.add_acquisition(sync_timeseries)


def create_nwb_trials(nwbfile, stream, stream_data, var_params, comments):
    """
    Adds trials to the nwbfile
    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile in which to store the trials
    stream : str
        The name of the stream sourcing the data
    stream_data : dict
        The stream's data
    var_params : dict
        NWB storage parameters for the data
    """

    # get key for the trial_state variable
    for var in var_params:
        if 'nwb' in var_params[var] and var_params[var][
                'nwb'] is not None and 'trial_state' in var_params[var]['nwb']:
            trial_state = var_params[var]['nwb']['trial_state']

    # first get trial indicators
    start_trial_indicators = var_params[trial_state]['nwb'][
        'start_trial_indicators']
    end_trial_indicators = var_params[trial_state]['nwb'][
        'end_trial_indicators']
    other_trial_indicators = var_params[trial_state]['nwb'][
        'other_trial_indicators']

    # get data indices corresponding to trial states
    starts = np.isin(stream_data['state']['data'], start_trial_indicators)
    start_inds = stream_data['state']['data'][starts]
    ends = np.isin(stream_data['state']['data'], end_trial_indicators)
    end_inds = stream_data['state']['data'][ends]
    others = {
        k: np.isin(stream_data['state']['data'], k)
        for k in other_trial_indicators
    }

    # get sync timestamps
    start_times = stream_data[trial_state]['sync_timestamps'][starts[:, 0]]
    end_times = stream_data[trial_state]['sync_timestamps'][ends[:, 0]]
    end_times = end_times[end_times >= start_times[0]] # remove end_times before the first start time
    other_times = {
        k: stream_data[trial_state]['sync_timestamps'][others[k][:, 0]]
        for k in others
    }

    # remove final start_time and relevant other_times if stopped during trial
    # do the if statement because will likely be faster considering the loop for other_times
    if start_times.shape[0] != end_times.shape[0]:
        start_times = start_times[:end_times.shape[0]]
        other_times = {
            k: other_times[k][other_times[k] < end_times[-1]]
            for k in other_times
        }

    # add a column for our other trial milestones
    for k in other_times:
        nwbfile.add_trial_column(
            name=k,
            description=var_params[trial_state]['nwb'][k + '_description'])

    for s_time, e_time, s_ind, e_ind in zip(start_times, end_times, start_inds,
                                            end_inds):
        # first find the other_times corresponding to the current trial
        trial_other_times = {
            k: other_times[k][np.logical_and(other_times[k] >= s_time,
                                             other_times[k] <= e_time)]
            for k in other_times
        }
        trial_other_times = {
            k: np.nan
            if trial_other_times[k].size == 0 else trial_other_times[k][0]
            for k in trial_other_times
        }

        # now create the trial
        nwbfile.add_trial(start_time=s_time,
                          stop_time=e_time,
                          start_indicator=s_ind,
                          stop_indicator=e_ind,
                          **trial_other_times)


def add_nwb_trial_info(nwbfile, stream, stream_data, var_params, comments):
    """
    Adds trial information to the trials table. The code
    below assumes the trials table has already been
    generated in nwbfile.
    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile in which to store trial information
    stream : str
        The name of the stream sourcing the data
    stream_data : dict
        The stream's data
    var_params : dict
        NWB storage parameters
    """

    num_trials = len(nwbfile.trials)

    for var in stream_data:
        var_data_per_trial = [None] * num_trials

        # loop through trials to get one entry of var per trial
        for id, trial in enumerate(nwbfile.trials):
            var_data_in_trial = stream_data[var]['data'][np.logical_and(
                stream_data[var]['sync_timestamps'] >= trial.start_time.values,
                stream_data[var]['sync_timestamps'] <= trial.stop_time.values)]
            var_data_per_trial[
                id] = np.nan if var_data_in_trial.size == 0 else var_data_in_trial[
                    0].astype(stream_data[var]['data'].dtype).item()

        nwbfile.add_trial_column(
            name=stream + '_' + var,
            description=var_params[var]['nwb']['description'],
            data=var_data_per_trial)


def create_nwb_position(nwbfile, stream, stream_data, var_params, comments):
    """
    Generates a position container
    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile in which to store the series
    stream : str
        The name of the stream sourcing the data
    stream_data : dict
        The stream's data
    var_params : dict
        NWB storage parameters for each variable
    """
    pos = Position(name=stream)

    for var in stream_data:
        pos.create_spatial_series(
            name=var,
            data=stream_data[var]['data'],
            timestamps=stream_data[var]['sync_timestamps'],
            comments=json.dumps(comments),
            **var_params[var]['nwb'])
    nwbfile.add_acquisition(pos)


def create_nwb_unitspiketimes(nwbfile, stream, stream_data, var_params, comments):
    """
    Adds spike times to pre-existing units
    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile in which to store the series
    stream : str
        The name of the stream sourcing the data
    stream_data : dict
        The stream's data
    var_params : dict
        NWB storage parameters for each variable
    """

    # get key for the trial_state variable
    for var in var_params:
        if var_params[var]['nwb'] is not None and 'crossings' in var_params[
                var]['nwb']:
            crossings_var = var_params[var]['nwb']['crossings']

    for electrode in range(stream_data[crossings_var]['data'].shape[1]):
        nwbfile.add_unit(
            electrodes=[electrode],
            electrode_group=nwbfile.electrodes.group[electrode],
            spike_times=stream_data[crossings_var]['sync_timestamps'][
                stream_data[crossings_var]['data'][:, electrode].astype(bool)],
            stream=stream,
            obs_intervals=stream_data[crossings_var]['sync_timestamps'][[0,-1]].reshape(1,-1))


def create_nwb_timeseries(nwbfile, stream, stream_data, var_params, comments):
    """
    Generates a time series container
    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile in which to store the series
    stream : str
        The name of the stream sourcing the data
    stream_data : dict
        The stream's data
    var_params : dict
        NWB storage parameters for each variable
    """

    for var in stream_data:
        # need new TimeSeries object for each stream/key
        timeseries = TimeSeries(name=f'{stream}_{var}',
                                data=stream_data[var]['data'],
                                unit=var_params[var]['nwb']['unit'],
                                timestamps=stream_data[var]['sync_timestamps'],
                                description=var_params[var]['nwb']['description'],
                                comments=json.dumps(comments))

        nwbfile.add_acquisition(timeseries)


###############################################
# Connect to redis
###############################################
try:
    logging.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    r = Redis(redis_host, redis_port, redis_socket, retry_on_timeout=True)
    r.ping()
except ConnectionError as e:
    logging.error(f"Error with Redis connection, check again: {e}")
    sys.exit(1)
except:
    logging.error('Failed to connect to Redis. Exiting.')
    sys.exit(1)

logging.getLogger().addHandler(RedisLoggingHandler(r, NAME))

logging.info('Redis connection successful.')

###############################################
# Load all stream and NWB info
# do outside of main loop for eventual translation to real-time
###############################################

try:
    model_stream_entry = r.xrevrange(b'supergraph_stream', '+', '-', 1)[0]
except IndexError as e:
    logging.error(
        f"No model published to supergraph_stream in Redis. Exiting.")
    sys.exit(1)

entry_id, entry_dict = model_stream_entry
model_data = json.loads(entry_dict[b'data'].decode())

if NAME in model_data['derivatives']:
    graph_params = model_data['derivatives'][NAME]['parameters']
    if 'devices_file' in graph_params:
        devices_path = graph_params['devices_file']
    else:
        devices_path = os.path.join(os.getenv('BRAND_BASE_DIR'),
                                '../Data/devices.yaml')
else:
    logging.warning(
        f"No {NAME} derivative configuration in the current graph. Exiting.")
    sys.exit(0)

# Get graph name
graph_name = model_data['graph_name']

## Get exportnwb_io
if 'streams' in model_data:
    stream_dict = model_data['streams']
else:
    logging.info('No streams in supergraph. Exiting.')
    sys.exit(1)
exportnwb_dict = graph_params['streams']

for stream in exportnwb_dict:
    if stream not in stream_dict:
        raise ValueError(f"Could not find definition of '{stream}' stream in "
                         "node YAML files")
    if 'sync' in exportnwb_dict[stream]:
        stream_dict[stream]['sync'] = exportnwb_dict[stream]['sync']
    else:
        logging.warning(
            f'Invalid NWB parameters in graph YAML. \'sync\' is required for each stream. Stream: {stream}'
        )
        stream_dict[stream]['enable_nwb'] = False

    if 'comments' in exportnwb_dict[stream]:
        stream_dict[stream]['comments'] = exportnwb_dict[stream]['comments']
    else:
        stream_dict[stream]['comments'] = {}

    if 'rate' in exportnwb_dict[stream]:
        stream_dict[stream]['comments']['rate'] = exportnwb_dict[stream]['rate']
        stream_dict[stream]['comments']['rate_type'] = 'graph'
        
    if 'enable_nwb' in stream_dict[stream] and 'type_nwb' in stream_dict[stream]:
        if 'enable' in exportnwb_dict[stream]:
            stream_dict[stream]['enable_nwb'] = exportnwb_dict[stream]['enable']
        else:
            logging.info(f'Using default NWB enable. Stream: {stream}')
    else:
        logging.warning(
            f'Invalid NWB parameters in node YAML. \'enable_nwb\' and \'type_nwb\' are required. Stream: {stream}'
        )
        stream_dict[stream]['enable_nwb'] = False

# Get timing keys
sync_key = graph_params['sync_key'].encode()
time_key = graph_params['time_key'].encode()
if 'sync_timing_hz' in graph_params:
    sync_timing_hz = graph_params['sync_timing_hz']
else:
    sync_timing_hz = {}
if isinstance(sync_timing_hz, dict) and 'default_sync_rate' in sync_timing_hz:
    DEFAULT_SYNC_RATE = sync_timing_hz['default_sync_rate']

sync_dict = {}
sync_list = []
for name, stream in exportnwb_dict.items():
    if 'sync' in stream:
        for s in stream['sync']:
            sync_list.append(s)
    else:
        logging.warning(f"Sync key not found in '{name}' stream config, ignoring it.")
        stream_dict[name]['enable_nwb'] = False

for sync in set(sync_list):
    if isinstance(sync_timing_hz, dict):
        if sync in sync_timing_hz:
            if isinstance(sync_timing_hz, numbers.Number):
                sync_dict[sync] = sync_timing_hz[sync]
            else:
                sync_dict[sync] = DEFAULT_SYNC_RATE
                logging.warning(f'Unsupported \'sync_timing_hz\' parameter. It should be a number. Using \'{NAME}\' default sync rate of {DEFAULT_SYNC_RATE} Hz. Sync: {sync}')
        elif 'default_sync_rate' in sync_timing_hz:
            if isinstance(sync_timing_hz['default_sync_rate'], numbers.Number):
                sync_dict[sync] = sync_timing_hz['default_sync_rate']
            else:
                sync_dict[sync] = DEFAULT_SYNC_RATE
                logging.warning(f'Unsupported \'default_sync_rate\' parameter. It should be a number. Using \'{NAME}\' default sync rate of {DEFAULT_SYNC_RATE} Hz. Sync: {sync}')
        else:
            sync_dict[sync] = DEFAULT_SYNC_RATE
            logging.warning(f'Sync rate undefined. Using \'{NAME}\' default sync rate of {DEFAULT_SYNC_RATE} Hz. Sync: {sync}')
    elif isinstance(sync_timing_hz, numbers.Number):
        sync_dict[sync] = sync_timing_hz
    else:
        sync_dict[sync] = DEFAULT_SYNC_RATE
        logging.warning(f'Unsupported \'sync_timing_hz\' parameter. It should be either a \'dict\' or a number. Using \'{NAME}\' default sync rate of {DEFAULT_SYNC_RATE} Hz. Sync: {sync}')
    pass


# find 'Trial', 'Trial_Info', and 'Spike_Times' streams
trial_stream = None
trial_info_stream = None
spike_times_streams = []
for stream in exportnwb_dict:
    # if the stream is of type 'Trial'
    if stream_dict[stream]['enable_nwb'] and stream_dict[stream]['type_nwb'] == 'Trial':
        if trial_stream == None:
            trial_stream = stream
        else:
            logging.error(
                f'Multiple Trial streams, only one allowed! Stream: {stream}')
    # if the stream is of type 'Trial_Info'
    elif stream_dict[stream]['enable_nwb'] and stream_dict[stream]['type_nwb'] == 'TrialInfo':
        trial_info_stream = stream
    elif stream_dict[stream]['enable_nwb'] and stream_dict[stream]['type_nwb'] == 'SpikeTimes':
        spike_times_streams.append(stream)

# guarantee there is a 'Trial' stream if we have a 'Trial_Info' stream
if trial_info_stream is not None and trial_stream == None:
    logging.error('Trial_Info stream exists but no Trial stream!')

# guarantee the 'Trial' stream is processed first
has_trial_stream = False
if trial_stream is not None:
    key_order = [k for k in exportnwb_dict if k not in [trial_stream]]
    key_order.insert(0, trial_stream)
    exportnwb_dict = {k: exportnwb_dict[k] for k in key_order}
    has_trial_stream = True

###############################################
# Prepare NWB file
###############################################

# get metadata
participant_metadata_file = graph_params['participant_file']
with open(participant_metadata_file, 'r') as f:
    yamlData = yaml.safe_load(f)
    participant_metadata = yamlData['metadata']
    participant_implants = yamlData['implants']

# get devices information
with open(devices_path, 'r') as f:
    devices = yaml.safe_load(f)

# TODO autogenerate these inputs to represent block information
nwbfile = NWBFile(session_description=graph_params['description'],
                  identifier=graph_name,
                  session_start_time=datetime.today(),
                  file_create_date=datetime.today())

# add trial column containing the stream's name that sourced each trial state change
if has_trial_stream:
    nwbfile.add_trial_column(
        name='start_indicator',
        description='list of the indicators used to start the trial')
    nwbfile.add_trial_column(
        name='stop_indicator',
        description='list of the indicators used to stop the trial')

# add unit column containing the stream's name that sourced the crossings
if spike_times_streams:
    nwbfile.add_unit_column(
        name='stream',
        description='Name of stream providing threshold crossings')
    nwbfile.add_unit_column(
        name='obs_intervals',
        description='Time period over which these threshold crossings were observed'
    )

# create devices, create electrode groups, and create electrodes
n_electrodes = 0
for implant in participant_implants:
    for device_entry in devices:
        if implant['device'] == device_entry['name']:
            device = device_entry
            break

    if device['name'] in nwbfile.devices:
        nwb_device = nwbfile.devices[implant['device']]
    else:
        nwb_device = nwbfile.create_device(name=device['name'],
                                           description=device['description'],
                                           manufacturer=device['manufacturer'])

    nwb_group = nwbfile.create_electrode_group(
        name=implant['name'],
        description=f'{implant["device"]} connected to {implant["connector"]}',
        location=implant['location'],
        device=nwb_device,
        position=implant['position'])

    # TODO autogenerate from subject implant (array files from Blackrock)
    # for now, just dummy electrode assignment
    for electrode in range(device['electrode_qty']):
        nwbfile.add_electrode(x=float(electrode),
                              y=float(electrode),
                              z=float(electrode),
                              imp=float(electrode),
                              location=implant['location'],
                              filtering='0.3 Hz 1st-order highpass Butterworth and 7.5 kHz 3rd-order lowpass Butterworth',
                              group=nwb_group)
        
    n_electrodes += device['electrode_qty']

###############################################
# Pull data from streams and write to NWB
###############################################

# find ID of find_reset_stream stream where a reset happens, if at all
start_id = '-'
reset_stream_final_ts = None
reset_stream_start_ts = None
reset_stream_incl_ts = None
unusual_ts_nsp = []
unusual_ts_mono = []
unusual_ts_mono_diff = []
if 'find_reset_stream' in graph_params:
    find_reset_stream = graph_params['find_reset_stream']
    if isinstance(find_reset_stream, str):
        resp = r.xrevrange(find_reset_stream, min='-', max='+')

        if 'reset_keys' in graph_params:
            reset_keys = graph_params['reset_keys']

            if isinstance(reset_keys, str):
                reset_keys = [reset_keys]

            if not isinstance(reset_keys, list):
                logging.warning(f'\'reset_keys\' must be a string or a list, but was {reset_keys}. Ignoring timestamp matching validation')
                reset_keys = [stream_dict[find_reset_stream]['sync'][0]]

            for k in reset_keys:
                if not isinstance(k, str) and not isinstance(k, numbers.Number):
                    logging.warning(f'\'reset_keys\' items must be strings or numbers, but one was {k}. Ignoring timestamp matching validation')
                    reset_keys = [stream_dict[find_reset_stream]['sync'][0]]
                    break

        else:
            reset_keys = [stream_dict[find_reset_stream]['sync'][0]]

        if find_reset_stream in stream_dict and 'rate' in stream_dict[find_reset_stream]['comments']:
            find_reset_stream_rate = stream_dict[find_reset_stream]['comments']['rate']
        else:
            find_reset_stream_rate = None

        # loop backwards through the entries
        done = False
        last_ts = {k:0xFFFFFFFF for k in stream_dict[find_reset_stream]['sync']}
        last_mono = np.frombuffer(resp[0][1][time_key], dtype=np.uint64).astype(np.float64) + 1.e9/np.array(find_reset_stream_rate, dtype=np.float64)
        for idx, (id, entry) in enumerate(resp):
            sync_data = json.loads(entry[sync_key])

            sync_val = sync_data[reset_keys[0]]
            for k in stream_dict[find_reset_stream]['sync']:
                # find first instance where timestamp increases from the previous timestamp or sync values are different
                if sync_data[k] > last_ts[k] or (k in reset_keys and sync_val != sync_data[k]):
                    logging.info(f'Timestamp reset found in {find_reset_stream} at Redis ID {id}, sync_dict {sync_data}. Ignoring {len(resp)-idx} entries')
                    done = True
                    break # if so, then the prior ID is where we should start, so break before updating the start_id
                    
                if k in reset_keys:
                    sync_val = sync_data[k]
                    
                last_ts[k] = sync_data[k]

            if done:
                break

            mono_time = np.frombuffer(entry[time_key], dtype=np.uint64).astype(np.float64)
            mono_diff_s = (last_mono - mono_time) / 1.e9
            if (find_reset_stream_rate is not None and np.abs(mono_diff_s - 1./find_reset_stream_rate) > 0.5/find_reset_stream_rate):
                unusual_ts_nsp.insert(0, sync_val / sync_timing_hz)
                unusual_ts_mono.insert(0, mono_time)
                unusual_ts_mono_diff.insert(0, mono_diff_s)

            last_mono = mono_time

            start_id = id

        reset_stream_final_ts = np.frombuffer(resp[0][1][time_key], dtype=np.uint64)
        reset_stream_start_ts = np.frombuffer(resp[-1][1][time_key], dtype=np.uint64)
        reset_stream_incl_ts = np.frombuffer(resp[idx][1][time_key], dtype=np.uint64)

        if unusual_ts_nsp:
            unusual_timeseries = TimeSeries(name=f'unusual_monotonic_ts',
                                 data=unusual_ts_mono_diff,
                                 unit='s',
                                 timestamps=unusual_ts_nsp,
                                 description='Timestamps in monotonic time that are not 1/rate apart')
            nwbfile.add_acquisition(unusual_timeseries)
    else:
        logging.warning(f'\'find_reset_stream\' must be a string, but was {find_reset_stream}. Skipping finding reset')

# set up dictionary of NWB writing functions
nwb_funcs = {
    'Trial': create_nwb_trials,
    'TrialInfo': add_nwb_trial_info,
    'Position': create_nwb_position,
    'SpikeTimes': create_nwb_unitspiketimes,
    'TimeSeries': create_nwb_timeseries
}

# loop through streams to extract data
for stream in exportnwb_dict:
    # checks the ENABLE_NWB parameter is set to true
    if stream_dict[stream]['enable_nwb']:

        strm = stream_dict[stream]  # shortcut to use later

        logging.info(f'Extracting data. Stream: {stream}')

        ###################################
        # first extract the data from redis
        ###################################
        stream_read = r.xrange( stream,
                                min=start_id)
        
        stream_len = len(stream_read)
        sync_name = strm['sync'][0]

        # stream_data:
        #   data:               store extracted data
        #                       dim0 = number of stream entries * number of samples per entry
        #                       dim1 = number of channels
        #   sync_timestamps:    store sync timestamps for each piece of extracted data
        #                       dim0 = number of stream entries * number of samples per entry
        #   sample_count:       track how many samples have been counted for each key
        stream_data = {
            k: {
                'data':
                np.empty((stream_len *
                          strm[k]['samp_per_stream'],
                          strm[k]['chan_per_stream']),
                         dtype=object
                         if strm[k]['sample_type'] == 'str' 
                         else strm[k]['sample_type']),  # ugly, but need to handle strings
                'sync_timestamps':
                np.empty(stream_len *
                         strm[k]['samp_per_stream'],
                         dtype=np.double),
                'sample_count':
                0
            }
            for k in strm
            if (k not in ['enable_nwb', 'type_nwb', 'source_nickname', 'sync', 'last_id']
                and 'nwb' in strm[k])
        }

        # time_data:
        #   sync_timestamps:    store sync timestamps for each piece of extracted data
        #                       dim0 = number of stream entries
        #   monotonic_ts:       store the monotonic clock timestamp
        #                       dim0 = number of stream entries
        #   redis_ts:           store the redis clock timestamp
        #                       dim0 = number of stream entries
        #   <other>:            store the sync timestamps for other, non-blocking incoming syncs
        time_data = {
            'sync_timestamps': np.empty(stream_len,
                                        dtype=np.double),  # sync timestamps
            'monotonic_ts': np.empty(stream_len,
                                     dtype=np.double),  # monotonic timestamp
            'redis_ts': np.empty(stream_len, dtype=np.double)
        }  # redis timestamp
        time_data.update({
            k: np.empty(stream_len, dtype=np.double)
            for k in strm['sync']
            if k != strm['sync'][0]
        })

        for ind, entry in enumerate(stream_read):
            sync_data = json.loads(entry[1][sync_key])
            time_data['sync_timestamps'][ind] = float(
                sync_data[sync_name]
            ) / sync_dict[sync_name]  # get blocked sync timestamp in ms, convert to seconds
            time_data['monotonic_ts'][ind] = np.frombuffer(
                entry[1][time_key], dtype=np.uint64)
            time_data['redis_ts'][ind] = float(
                entry[0].decode('utf-8').split('-')[0]) / 1000

            # get other sync signals for this entry
            for sync in sync_data:
                if sync == sync_name:
                    continue
                time_data[sync][ind] = float(sync_data[sync]) / sync_dict[sync]

            for var in stream_data:
                if 'nwb' not in strm[var]:
                    continue
                var_config = strm[var]
                batch_idx = list(
                    range(
                        stream_data[var]['sample_count'],
                        stream_data[var]['sample_count'] +
                        var_config['samp_per_stream']))
                if strm[var]['sample_type'] == 'str':
                    stream_data[var]['data'][batch_idx, :] = entry[1][
                        var.encode()].decode('utf-8')
                else:
                    stream_data[var]['data'][batch_idx, :] = np.frombuffer(
                        entry[1][var.encode()],
                        dtype=stream_data[var]['data'].dtype).reshape([
                            var_config['samp_per_stream'],
                            var_config['chan_per_stream']
                        ])
                stream_data[var]['sync_timestamps'][batch_idx] = time_data[
                    'sync_timestamps'][ind]
                stream_data[var]['sample_count'] += var_config[
                    'samp_per_stream']
                
        # check for non-monotonic sync
        # We should have excluded all IDs before resets, so if we get any warnings from here, that's a major issue
        if np.any(np.diff(time_data['sync_timestamps']) < 0):
            # remove time_data before the sync resets
            reset = np.argwhere(np.diff(time_data['sync_timestamps']) < 0)
            logging.warning(f'Sync was non-monotonic at sample {reset}, ignoring all data before sample {reset[-1]+1}. Stream: {stream}')
            for k in time_data:
                time_data[k] = time_data[k][int(reset[-1]+1):]
            
            # remove stream_data before the sync resets
            for k in stream_data:
                reset = np.argwhere(np.diff(stream_data[k]['sync_timestamps']) < 0)
                stream_data[k]['data'] = stream_data[k]['data'][int(reset[-1]+1):, :]
                stream_data[k]['sync_timestamps'] = stream_data[k]['sync_timestamps'][int(reset[-1]+1):]

        # estimate stream's sampling rate if not provided
        if 'rate' not in strm['comments']:
            stream_keys = list(stream_data.keys())
            time_between_samples = np.diff(stream_data[stream_keys[0]]['sync_timestamps'])
            bin_size = np.round(np.median(time_between_samples), decimals=6)
            strm['comments']['rate'] = 1./bin_size
            strm['comments']['rate_type'] = 'inferred'
        
        #####################################
        # now append the data to the NWB file
        #####################################
        add_stream_sync_timeseries(nwbfile, stream, time_data)

        nwb_funcs[strm['type_nwb']](
            nwbfile, stream, stream_data, {
                k: strm[k]
                for k in strm
                if k not in ['enable_nwb', 'type_nwb', 'source_nickname', 'sync', 'last_id', 'comments']
            },
            strm['comments'])

        logging.info(f'Export completed. Stream: {stream}')
        

##############################
# now add other electrode info
##############################

def get_params_from_stream(stream_key, search_keys, start_id):
    """
    Gets parameters from a stream or list of streams

    Parameters
    ----------
    stream_key : str or list
        The exportNWB parameter containing the stream or list of streams to search for parameters
    search_keys : str or bytes or list
        The keys to search for
    start_id : str
        The latest Redis ID from which to xrevrange

    Returns
    -------
    tuple
        The values of the parameters requested
    """

    if isinstance(search_keys, str) or isinstance(search_keys, bytes):
        search_keys = [search_keys]
                
    # overwritten if values found
    values = tuple(None for _ in search_keys)

    if stream_key in graph_params:
        stream_name = graph_params[stream_key]

        if isinstance(stream_name, str):
            stream_name = [stream_name]
        
        for s in stream_name:
            stream_entry = r.xrevrange(s, start_id, '-', count=1)
            if stream_entry:
                entry = stream_entry[-1]
                if set(search_keys).intersection(set(entry[1].keys())) == set(search_keys):
                    values = tuple(np.frombuffer(entry[1][k], dtype=np.float64) for k in search_keys)
                    sk_string = b", ".join(search_keys).decode("utf-8") if isinstance(search_keys[0], bytes) else ", ".join(search_keys)
                    logging.info(f'Loaded {sk_string} from {s} stream')
                    break
                else:
                    logging.warning(f'Could not find at least one of {", ".join(search_keys)} keys in {s} stream')
            else:
                logging.warning(f'No entries in {s} stream')

    return values[0] if len(search_keys) == 1 else values

def get_params_from_yaml_file(file_key, search_keys):
    """
    Gets parameters from a YAML file

    Parameters
    ----------
    file_key : str
        The exportNWB parameter containing the full path to the YAML file
    search_keys : str or list
        The keys to search for

    Returns
    -------
    tuple
        The values of the parameters requested
    """

    if isinstance(search_keys, str):
        search_keys = [search_keys]
                
    # overwritten if values found
    values = tuple(None for _ in search_keys)

    if file_key in graph_params:
        filepath = graph_params[file_key]
        
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f'Could not find file at {filepath}')
            return values[0] if len(search_keys) == 1 else values

        if set(search_keys).intersection(set(data.keys())) == set(search_keys):
            values = tuple(np.array(data[k]) for k in search_keys)
            logging.info(f'Loaded {", ".join(search_keys)} from {filepath}')
        else:
            logging.warning(f'Could not find at least one of {", ".join(search_keys)} keys in {filepath}')

    return values[0] if len(search_keys) == 1 else values

# for the remaining start_id uses, this ensures stream entries are read
if start_id == '-':
    start_id == '+'

# add electrode column containing its channel mask status
if 'ch_mask_stream' in graph_params:
    ch_mask_entry = r.xrevrange(graph_params['ch_mask_stream'], start_id, '-', count=1)
    if ch_mask_entry:
        id, ch_mask_entry = ch_mask_entry[0]
        ch_mask = np.frombuffer(ch_mask_entry[b'channels'], np.uint16)
    else:
        logging.warning(f'\'ch_mask_stream\' was set to {model_data["derivatives"][NAME]["parameters"]["ch_mask_stream"]}, but there were no entries. Ignoring channel mask')
        ch_mask = None
else:
    ch_mask = None

if ch_mask is None:
    ch_mask = ['N/A'] * len(nwbfile.electrodes)
else:
    ch_mask = [ch in ch_mask for ch in range(len(nwbfile.electrodes))]

nwbfile.add_electrode_column(
    name='mask',
    description='whether the channel was enabled or disabled in the mask',
    data=ch_mask)

# add electrode column containing its threshold
if 'thresh_stream' in graph_params:
    thresholds = get_params_from_stream('thresh_stream', b'thresholds', start_id)
else:
    # None if not there
    thresholds = get_params_from_yaml_file('thresh_file', 'thresholds')

if thresholds is None:
    thresholds = ['N/A'] * len(nwbfile.electrodes)

if len(thresholds) == len(nwbfile.electrodes):
    nwbfile.add_electrode_column(
        name='threshold',
        description='threshold used for extracting spike timings, units bits',
        data=thresholds)
else:
    logging.warning('The number of thresholds did not equal the number of electrodes, skipping')

# add electrode column containing its mean and STD
if 'norm_stream' in graph_params:
    means, stds = get_params_from_stream('norm_stream', [b'means', b'stds'], start_id)
else:
    # Nones if not there
    means, stds = get_params_from_yaml_file('norm_file', ['means', 'stds'])

if means is None:
    means = ['N/A'] * len(nwbfile.electrodes)

if stds is None:
    stds = ['N/A'] * len(nwbfile.electrodes)

if len(means) == len(nwbfile.electrodes):
    nwbfile.add_electrode_column(
        name='mean',
        description='mean used for normalizing binned spikes',
        data=means)
else:
    logging.warning('The number of means did not equal the number of electrodes, skipping')

if len(stds) == len(nwbfile.electrodes):
    nwbfile.add_electrode_column(
        name='std',
        description='standard deviation used for normalizing binned spikes',
        data=stds)
else:
    logging.warning('The number of stds did not equal the number of electrodes, skipping')

# add electrode column containing its denormalization mean and STD
if 'denorm_stream' in graph_params:
    de_means, de_stds = get_params_from_stream('denorm_stream', [b'means', b'stds'], start_id)
else:
    # Nones if not there
    de_means, de_stds = get_params_from_yaml_file('denorm_file', ['means', 'stds'])

if de_means is None:
    de_means = ['N/A'] * len(nwbfile.electrodes)

if de_stds is None:
    de_stds = ['N/A'] * len(nwbfile.electrodes)

if len(de_means) == len(nwbfile.electrodes):
    nwbfile.add_electrode_column(
        name='denorm_mean',
        description='mean used for denormalizing normalized binned spikes',
        data=de_means)
else:
    logging.warning('The number of denormalizing means did not equal the number of electrodes, skipping')

if len(de_stds) == len(nwbfile.electrodes):
    nwbfile.add_electrode_column(
        name='denorm_std',
        description='standard deviation used for denormalizing normalized binned spikes',
        data=de_stds)
else:
    logging.warning('The number of denormalizing STDs did not equal the number of electrodes, skipping')

# log how much time lost to unusual timing
if reset_stream_incl_ts is not None and reset_stream_final_ts is not None and reset_stream_start_ts is not None:
    logging.info(f'Lost {(reset_stream_incl_ts.item() - reset_stream_start_ts.item())/1e9:.3f} of {(reset_stream_final_ts.item() - reset_stream_start_ts.item())/1e9:.3f} seconds of data due to NSP clock reset')
if unusual_ts_nsp:
    logging.info(f'The latest unusual timestamp was at {unusual_ts_nsp[-1]} seconds NSP time and {(unusual_ts_mono[-1].item() - reset_stream_start_ts.item())/1e9:.3f} seconds monotonic time')


###############################################
# Save the NWB object to file
###############################################
save_filename = r.config_get('dbfilename')['dbfilename']
save_filename = os.path.splitext(save_filename)[0] + '.nwb'
save_filepath = r.config_get('dir')['dir']
save_filepath = os.path.dirname(save_filepath)
save_filepath = os.path.join(save_filepath, 'NWB')

if not os.path.exists(save_filepath):
    os.makedirs(save_filepath)

save_path = os.path.join(save_filepath, save_filename)

# save the file
logging.info(f'Saving NWB file to: {save_path}')
with NWBHDF5IO(save_path, 'w') as io:
    io.write(nwbfile)

r.xadd('nwb_file_stream', {'file': save_path})