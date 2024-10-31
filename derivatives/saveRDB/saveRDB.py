#! /usr/bin/env python
"""
Saves Redis data to a dump.rdb with a customizable name,
the option to remove streams before saving, and the
option to delete streams with exceptions after saving.
"""

import argparse
import json
import logging
import os
import signal
import sys

from brand.redis import RedisLoggingHandler
 
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
## setting up clean exit code
###############################################
def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(0)

# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)


###############################################
# Connect to redis and pull supergraph
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

try:
    supergraph_entry = r.xrevrange(b'supergraph_stream', '+', '-', 1)[0]
except IndexError as e:
    logging.error(
        f"No model published to supergraph_stream in Redis. Exiting.")
    sys.exit(1)

entry_id, entry_dict = supergraph_entry
supergraph = json.loads(entry_dict[b'data'].decode())

graph_params = supergraph['derivatives'][NAME]['parameters']


###############################################
# Load parameters
###############################################

# streams to flush before saving the RDB
if 'flush_streams_before_save' in graph_params:
    flush_streams_before_save = graph_params['flush_streams_before_save']
    if isinstance(graph_params['flush_streams_before_save'], str):
        flush_streams_before_save = [flush_streams_before_save]

    if isinstance(flush_streams_before_save, list):
        stream_to_rm = []
        for s in flush_streams_before_save:
            if not isinstance(s, str):
                logging.warning(f'{s} entered in \'flush_streams_before_save\' was not a string, ignoring')
                stream_to_rm.append(s)
        
        for s in stream_to_rm:
            flush_streams_before_save.remove(s)

    else:
        logging.warning(f'\'flush_streams_before_save\' must be a string or a list of strings, but was {type(flush_streams_before_save)}, ignoring')
        flush_streams_before_save = []
else:
    flush_streams_before_save = []

# whether to flush after saving the RDB
if 'flush_rdb_after_save' in graph_params:
    flush_rdb_after_save = graph_params['flush_rdb_after_save']
    if not isinstance(flush_rdb_after_save, bool):
        logging.warning(f'\'flush_rdb_after_save\' must be boolean, but was {type(flush_rdb_after_save)}, defaulting to not flush the RDB')
        flush_rdb_after_save = False
else:
    flush_rdb_after_save = False

# any streams to not flush
if 'flush_rdb_except_streams' in graph_params:
    flush_rdb_except_streams = graph_params['flush_rdb_except_streams']
    if isinstance(graph_params['flush_rdb_except_streams'], str):
        flush_rdb_except_streams = [flush_rdb_except_streams]

    if isinstance(flush_rdb_except_streams, list):
        stream_to_rm = []
        for s in flush_rdb_except_streams:
            if not isinstance(s, str):
                logging.warning(f'{s} entered in \'flush_rdb_except_streams\' was not a string, ignoring')
                stream_to_rm.append(s)
        
        for s in stream_to_rm:
            flush_rdb_except_streams.remove(s)

    else:
        logging.warning(f'\'flush_rdb_except_streams\' must be a string or a list of strings, but was {type(flush_rdb_except_streams)}, ignoring')
        flush_rdb_except_streams = []
else:
    flush_rdb_except_streams = []

# don't flush streams with this prefix
if 'no_flush_prefix' in graph_params:
    no_flush_prefix = graph_params['no_flush_prefix']
    if not isinstance(no_flush_prefix, str):
        logging.warning(f'\'no_flush_prefix\' must be a string, but was {type(no_flush_prefix)}, defaulting to not flush streams with "session:" prefix')
        no_flush_prefix = 'session:'
else:
    no_flush_prefix = 'session:'


###############################################
# Flush specified streams before saving
###############################################

flushed_streams = []
for s in flush_streams_before_save:
    r.delete(s)
    flushed_streams.append(s)

r.memory_purge()

if flushed_streams:
    logging.info(f'Flushed the following streams before saving: {flushed_streams}')


###############################################
# Save the RDB
###############################################
r.xadd('supervisor_ipstream', {'commands':'saveRdb'})


###############################################
# Flush streams except those specified
###############################################

if flush_rdb_after_save:
    no_flush_streams = [k.decode('utf-8') for k in r.keys(f'{no_flush_prefix}*')] if no_flush_prefix else []
    no_flush_streams += flush_rdb_except_streams
    if no_flush_streams:
        cs_streams = ','.join(no_flush_streams)
        r.xadd('supervisor_ipstream', {'commands':'flushGraphStreams',
                                       'except':cs_streams})
    else:
        r.xadd('supervisor_ipstream', {'commands':'flushGraphStreams'})