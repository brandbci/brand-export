# Guidelines for Writing BRAND Nodes for NWB Compatibility

This guide will detail best practices for writing BRAND nodes to be compatible with general NWB exporting. `exportNWB` requires using Emory's version of `supervisor`. To make nodes comply with `exportNWB`, it is recommended that the node's author be familiar with the NWB data structure. Useful links can be found here:

* [Neurodata Without Borders website](https://www.nwb.org/)
* [PyNWB Documentation](https://pynwb.readthedocs.io/en/latest/index.html)
* [MatNWB Documentation](https://neurodatawithoutborders.github.io/matnwb/doc/index.html)

## Graph YAML

When launching a graph with Emory's version of `supervisor`, the supergraph contains compiled stream definitions. These stream definitions are used by the `exportNWB` script. To ensure these definitions are present in the supergraph, information about the streams needs to occur in two places in the graph YAML file: within a node's instantiation and within the `derivatives` section.

```yaml
nodes:
  - name:                       <node name>
    nickname:                   <node nickname>
    module:                     <node module>
    redis_outputs:
      <graph_stream_1>:         <node_yaml_stream_1>
      <graph_stream_2>:         <node_yaml_stream_2>
      ...
    parameters:
      ...

derivatives:
  - exportNWB:
      parameters:
        participant_file:       <path to participant YAML file>
        devices_file:           <path to devices YAML file> # optional, defaults to ../Data/devices.yaml
        description:            description of the graph
        sync_key:               <the sync key used throughout the graph>
        time_key:               <the time key used throughout the graph>
        sync_timing_hz:                                     # optional, the rate at which each sync label has entries
                                                            # made in its corresponding stream, can be a numeric
                                                            # value to apply to all sync labels or a dict defining a
                                                            # rate for each sync label. If not provided, exportNWB
                                                            # applies a default 1000 Hz to each sync

          <sync_label_1>:       <sync rate>                 # include one key/value pair for each label

          default_sync_rate:    <sync rate>                 # optional, set a default rate if a sync label is not
                                                            # explicitly defined

        streams:
          <graph_stream_1>:                                 # stream name as defined in the graph
            
            enable:             <True/False>                # optional, defaults to the node's YAML value if not
                                                            # defined here

            sync:               <list of sync labels>       # the 0th item should be the blocking sync, add others to
                                                            # include in NWB
                                                            
            rate:               <sample rate of stream>     # optional, the rate at which entries are entered in this stream

          <graph_stream_2>:                                 # stream name as defined in the graph

            enable:             <True/False>                # optional, defaults to the node's YAML value if not
                                                            # defined here

            sync:               <list of sync labels>       # the 0th item should be the blocking sync, add others to
                                                            # include in NWB

            rate:               <sample rate of stream>     # optional, the rate at which entries are entered in this stream
```

Here is an example `exportNWB` configuration:

```yaml
nodes:
  - name:                     func_generator
    nickname:                 func_generator
    module:                   ../brand-modules/brand-test
    redis_outputs:
      func_generator:         func_generator
      func_generator2:        func_generator                # multiple streams may point to the same structure in the node's YAML file
    parameters:
      sample_rate:            1000
      n_features:             96
      n_targets:              2
      log:                    INFO   

derivatives:
  - exportNWB:
      parameters:
        participant_file:     ../Data/t0/t0.yaml
        devices_file:         ../Data/devices.yaml
        description:          test graph
        sync_key:             sync
        time_key:             ts
        sync_timing_hz:       # 1000                        # set a default value for all sync labels
          # or set a rate for each label
          i:                  1000
          default_sync_rate:  1000
        streams:
          func_generator:
            enable:           True
            sync:             ['i']
            rate:             1000
          func_generator2:
            # no enable, so uses the default from the node's YAML
            sync:             ['i']
```

The `redis_outputs` "connectors" in a node's instantiation indicate to `supervisor` which stream structure in the node's YAML file (value) corresponds to the stream in the Redis database (key, since stream names must be unique). The information in the `derivatives` section in the graph YAML file contains necessary sync information (see [Data Alignment](#data-alignment) below). It is highly recommended to use YAML anchors and aliases to define stream names so spellings do not differ between the "connector" in the node's instantiation and the `exportNWB` instantiation in the `derivatives` section of the graph (see [here](https://stackoverflow.com/a/51834925) for basic tutorial on using anchors and aliases for YAML keys).

As [described below](#node-yaml), the YAML file of the node that generates a stream will also contain an `enable_nwb` parameter for whether that stream is enabled for logging by default. The `enable_nwb` parameter in the `exportNWB` derivative instantiation in the graph YAML is the priority setting for a given stream. If the `enable` key is not included under a stream's parameters within the `exportNWB` derivative instantiation, then `exportNWB` pulls the parameter from the node's YAML. A stream's optional `rate` parameter can be optionally set to indicate the stream's entry rate within the NWB file, stored as a serialized JSON dictionary within each NWB acquisition's `comments` field. If the `rate` parameter is absent for a stream, `exportNWB` attempts to infer the rate from the entries in the stream.

As [mentioned below](#participant-metadata), `exportNWB` also requires two YAML files, one with data on the participant and the other with all possible devices used for recording. Templates for each can be found in the [templates](./templates/) directory. The path to the participant file is specified in the `participant_file` field, and the optional path to the devices file is specified in the `devices_file` field.

## Data Alignment

Since BRAND runs asynchronous graphs, there is a need to track the flow of data through a graph for data integrity. See the [Guidelines for Data Alignment](./DataSyncGuidelines.md) for a description of how this is implemented for `exportNWB`. This is therefore critical to the functionality of `exportNWB`, since it must take that data and store it in a deterministic way. To do so, data from all streams in a graph are logged using the `sync` key contained in each stream entry as the timestamp for that entry. `exportNWB` also generates a `<stream_name>_ts` container that uses the NWB `TimeSeries` container. The `<stream_name>_ts` `TimeSeries` has one item entered for each `sync` key value in the stream. In each item, `exportNWB` logs the `monotonic` timestamps, which are required according to the [Guidelines for Data Alignment](./DataSyncGuidelines.md), and Redis timestamps at which each entry was logged. Additionally, if the stream is composed of multiple input streams, any additional `sync` labels are included in a separate column of `<stream_name>_ts`.

## Node YAML

Information regarding how a stream should be exported in an NWB file should be contained in the YAML file for the node that generates that stream. This implies the author of the node must write information regarding how that stream should be logged in NWB, as the author knows the stream's content best. Within a node's YAML, the NWB writing information is included in the stream's definition:

```yaml
RedisStreams:
  Inputs:
    ...

  Outputs:
    <output_stream_1>:
      enable_nwb:             <True/False>
      type_nwb:               <TimeSeries/Position/SpikeTimes/Trial/TrialInfo>
      <output_stream_1_key_1>:
        chan_per_stream:      [number of channels in each stream entry]
        samp_per_stream:      [number of samples per channel in each stream entry]
        sample_type:          [datatype of the sample]
        nwb:
          <nwb_parameter_1>:  value
          <nwb_parameter_2>:  value
          ...
      <output_stream_1_key_2>:
        ...
    <output_stream_2>:
      ...
```

Datatypes supported by `exportNWB` include any numeric datatype inherently supported by `numpy`, and strings. In the stream's definition in the node YAML, strings must have the datatype `str` to be logged in the NWB file without crashing the script.

`enable_nwb` is a required parameter that represents the default behavior for whether the stream `output_stream_1` should be exported to NWB if the `enable_nwb` parameter is not defined for `output_stream_1` at the graph level, which is the priority value. `type_nwb` is a required parameter that represents what NWB mechanism (see Rules below) should be used to fit all the data of the stream. Within each stream's keys, one can enable logging of that key by including an `nwb:` parameter field even if no parameters are required. Omitting the `nwb:` parameter field disables that key from being logged. Current supported mechanisms are `TimeSeries`, `Position`, `SpikeTimes`, `Trial`, and `TrialInfo`. See sections below for required `nwb` parameters for each mechanism.

### Variable Stream Dimensions

The exact dimensions of a stream often depend on parameters given to the node (i.e. a quantity of recording channels). To facilitate variable dimensions of a stream, Emory's `supervisor` will search for the reserved `$` character in `chan_per_stream`, `samp_per_stream`, and `sample_type`. `supervisor` assumes the text following a `$` character is the name of a parameter of that node, and `supervisor` will replace the parameter name and prefix `$` with the parameter value when the graph is loaded. Note that only top-level parameters may be used with the `$` character (i.e. parameters nested in a dictionary cannot be referenced with `$`).

Setting the `chan_per_stream` and `samp_per_stream` can also include basic mathematical operators, including `+`, `-`, `*`, `/`, `%`, `^`, `(`, and `)`. This is particularly useful if a stream dimension relates to multiple parameters. Note that any usage of `$` with `chan_per_stream` and `samp_per_stream` must resolve to a parameter of type `numbers.Number`.

Variably-defining `sample_type` has two requirements:
1. It must be a single parameter (no mathematical operators) of type `str`
1. Usage of `$` must resolve to a `str` type

See below for some example usages of `$`:

```yaml
RedisStreams:
  Inputs:
    ...

  Outputs:
    func_generator:
      enable_nwb:           True
      type_nwb:             TimeSeries
      samples:
        chan_per_stream:    $n_features
        samp_per_stream:    ($n_features*2 + 10)/$n_targets # won't actually work with current func_generator
        sample_type:        $input_dtype
        nwb:
          unit: n/a
          description: arbitrary generated features
```

### Rules for NWB Logging Mechanisms

Please follow each mechanism's rules to guarantee your data is properly logged in the NWB file.

#### `TimeSeries`

The `TimeSeries` mechanism creates a `TimeSeries` NWB container with data stored as a generic time series. This is equivalent to storing a `numpy.ndarray` of size `[num_time, num_dimensions]`. `exportNWB` logs sample times using the `sync` key included in the entry (see DataSyncGuidelines.md). Required `nwb` parameters in the stream's definition are those required for a [`TimeSeries`](https://pynwb.readthedocs.io/en/stable/pynwb.behavior.html#pynwb.behavior.TimeSeries) object in the NWB format, namely:

* `unit`: (`str`) the base unit of measurement (should be SI unit)

Note: `name`, `data`, and one of `timestamps` or `rate` are required, but these are generated automatically by `exportNWB`.

#### `Position`

The `Position` mechanism creates a `Position` NWB container with data stored as a time series. This is equivalent to storing a `numpy.ndarray` of size `[num_time, num_dimensions]`. `exportNWB` logs sample times using the `sync` key included in the entry (see DataSyncGuidelines.md). Required `nwb` parameters in the stream's definition are those required for a [`SpatialSeries`](https://pynwb.readthedocs.io/en/stable/pynwb.behavior.html#pynwb.behavior.SpatialSeries) object in the NWB format, namely:

* `reference_frame`: (`str`) description defining what the zero-position is

Note: `name`, `data`, and one of `timestamps` or `rate` are required, but these are generated automatically by `exportNWB`.

#### `Spike_Times`

The `Spike_Times` mechanism adds new units to the `units` table in the NWB file containing the spiking times of the electrode. Each stream entry should be of the format `[1, num_electrodes]` where each element is a boolean indicator of a spike having occurred on that channel for that entry. Spiking times logged in the `units` table are the `sync` key values at which the entries were added to the stream. The only required `nwb` parameter is:

* `crossings`: (`str`) not a formal NWB parameter, but indicates to `exportNWB` which key in the stream's entries contains the threshold crossing indicators.

#### `Trial`

The `Trial` mechanism creates a `trials` table in the NWB file. Each entry in the `trials` table must have start and stop times. The `trials` table forces no structure on continuously acquired data (such as data stored via the `Position` mechanism). Customized variables can be added to this table, i.e. movement onset times (see `other_trial_indicators` below). If a custom variable does not have a value for a given trial, that table entry is automatically filled with `NaN` (i.e. consider a monkey that fails a trial yielding no reward, so the reward time would be `NaN`). **At most one** stream in the entire graph should have its `type_nwb` set to `Trial`, and `exportNWB` automatically processes the `Trial` stream first if one is enabled (see `Trial_Info` for reasoning). An `indicators` column is automatically generated in the `trials` table to indicate which `start` and `end` trial indicators resulted in each trial. Required `nwb` parameters for the `Trial` mechanism are as follows:

* `trial_state`: (`str`) not a formal NWB parameter, but the name of the stream key containing trial states.
* `start_trial_indicators`: (`list` of (`str` or `numeric`)) not a formal NWB parameter, but elements indicating the start of a trial. For example, this could be a list such as `['start_trial']` where presence of this string indicates the time at which a trial was started. Note, `numeric` elements in the list have not been tested.
* `end_trial_indicators`: (`list` of (`str` or `numeric`)) not a formal NWB parameter, but elements indicating the end of a trial. For example, this could be a list such as `['stop_trial']` or with more abstract trial states, such as `['failure', 'between_trials']`, where presence of these strings indicate the time at which a trial was concluded. Note, `numeric` elements in the list have not been tested.
* `other_trial_indicators`: (`list` of (`str` or `numeric`)) not a formal NWB parameter, but elements indicating significant trial milestones. For example, this could be a list such as `['movement', 'reward']`. This parameter is required, though it can be an empty list indicated as `[]`.

Optional `nwb` parameters for the `Trial` mechanism are as follows:

* `<other_indicator>_description`: (`str`) descriptions of trial columns are required in the NWB file format. Every element of `other_trial_indicators` must have its own description parameter under `nwb` and the parameter name must match the exact name as included in `other_trial_indicators` with `_description` appended to the end.

#### `Trial_Info`

The `Trial_Info` mechanism adds trial information as columns to the `trials` table. The `Trial_Info` mechanism requires a pre-existing `trials` table, so `exportNWB` automatically processes the *only* `Trial` stream first. The data from each key in a `Trial_Info` stream is logged as a separate trial column. The column name in the `trials` table is autogenerated to be `<stream_name>_<key_name>`. If multiple data samples exist in a key for a given trial, only the first sample belonging to the trial is logged in the table. Note that data samples for a trial may be a vector of samples from multiple channels, where the whole vector will be logged as the entry for the trial. If no data points exist in a key for a given trial, the table entry is filled with a `NaN`. The only required `nwb` parameter is:

* `description`: (`str`) description of the trial column to be added, which is required in the NWB file format.

## Participant Metadata

It is helpful to log de-identified participant information alongside data without having to manually enter it into each NWB file. To this end, a `<participant>.yaml` (i.e. named `T14.yaml`), containing de-identified participant metadata, is required to store data in an NWB file using `exportNWB`. Atemplate for this file is included at `derivatives/exportNWB/templates/t0.yaml`.

The `<participant>.yaml` file should be structured as follows, and all elements are required:

```
metadata:
  participant_id:         T<num>
  cortical_implant_date:  [date of cortical implant surgery]

implants:
  - name:       [name of implant, i.e. medial hand knob]
    location:   [anatomical location of implant]
    position:   [stereotaxic position of implant]
    device:     [name of device, i.e. NeuroPort10x10]
    connector:  [name of connector, i.e. left anterior]
    serial:     [serial number of the device]
    array_map:  [path to the map file of the device provided by manufacturer]
  
  - name:       
    ...
```

To store electrophysiology recordings in the NWB file format, information about the devices used for recording is required. This is done by creating `device` objects along with their corresponding `electrodes`. Rather than repeating this information for each implant within the `<participant>.yaml` file, the `device` parameter in the `<participant>.yaml` file should be a name pointing to a device entry in a `devices.yaml`. A template for this file is included at `derivatives/exportNWB/templates/devices.yaml`. This `devices.yaml` should be structured as follows:

```
- name:           [name of device]
  electrode_qty:  [quantity of wired electrodes on the device]
  description:    [text description of the device]
  manufacturer:   [manufacturer of the device]

- name:           NeuroPort10x10
  electrode_qty:  96
  description:    10x10 96 channel NeuroPort array
  manufacturer    Blackrock Neurotech

- name:
  ...
```

The `<devices>.yaml` should exist in a `Data` directory one level above the execution path (assumed to be `realtime_rig_dev` at the moment). The `<participant>.yaml` should exist within the corresponding participant directory within this `Data` directory (e.g. `../Data/t0`). Templates for both files are located at `derivaties/exportNWB/templates` and can be copied to those respective folders and filled in as needed.

## File Storage

To stop a graph and save an NWB file, the `stopGraphAndSaveNWB` command must be sent to the `supervisor_ipstream` stream. Saved files are currently stored in a `../Data` folder one level above the execution path (assumed to be `realtime_rig_dev` at the moment). It will automatically create a new folder within `Session/Data` for the participant called `T<num>`, a new folder for the session within that `ses-<session_number>`, and a new folder within that called `NWB`. Within this folder, a file named `T<num>_ses-<session_number>.nwb` is saved. Future versions of the save functionality will use date and block information from the session metadata to create the directory structure and file name.
