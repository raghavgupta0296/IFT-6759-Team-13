from utilities.sequencer_utils import MIN_TIME
from utilities.sequencer_utils import next_day
from utilities.sequencer_utils import time_in_seconds

from random import shuffle

import pickle as pkl
import numpy as np


# A Class used for reading sequences of data
class Sequencer:

    # Init method
    def __init__(self, stations_names, stations_mappings, offset = 1, seq_length = 6, batch_size = 50, shuffle = True):
        self._CHUNK_SIZE = 1000
        self._stations_names = stations_names
        self._stations_mappings = stations_mappings
        self._offset = offset
        self._seq_length = seq_length
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._current_indexes = {name : 0 for name in stations_names}
        self._current_seq_num = {name : 0 for name in stations_names}
        self._memory_segments = {name : [] for name in stations_names}

    # Computes the next sequence number.
    def _next_seq(self, station_name):
        # check seq number
        seq_number = self._current_seq_num[station_name]
        if seq_number == -1:
            return -1
        else:
            seq_number += 1

        # check sequence number
        if seq_number not in self._stations_mappings[station_name]:
            self._current_seq_num[station_name] = -1 # No more data to load
            return -1

        # Valid sequence number
        self._current_seq_num[station_name] = seq_number
        return seq_number



    # Loading memory segments (lists) and updates sequences tracking mechanism
    def _load_segment(self, station_name):
        # check station name
        if station_name not in self._stations_names:
            print('ERROR: station %s not found' %station_name)
            return -1

        seq_number = self._next_seq(station_name)
        if seq_number == -1:
            print('WARNING: no more data to load for station %s' % station_name)
            return 0

        mem_file = None
        filepath = self._stations_mappings[station_name][seq_number]
        try:
            mem_file = pkl.load(open(filepath, 'rb'))
        except Exception:
            print('Could not load file \'%s\'' %filepath)
            return -1

        # Empty list: should not happen
        if not mem_file:
            print('WARNING: loaded empty list from %s' %filepath)
            return -1

        # No change in index here
        if not self._memory_segments[station_name]:
            self.current_indexes[station_name] = 0

        # Updating the data
        self._memory_segments[station_name] += mem_file

        # number of memory chunks successfully loaded
        return 1



    # Truncates unwanted data for better memory management
    def _truncate_memory_segment(self, station_name, size):
        mem_list = self._memory_segments[station_name]
        if mem_list:
            del(mem_list[:size])
        # Update the index
        self.current_indexes[station_name] = 0


    # Reads epoch time from samples
    def _read_epoch_time(self, cell):
        return cell['iso_datetime']

    # Prepares data segment
    def _validate_data_segment(self, station_name):
        mem_list = self._memory_segments[station_name]
        # No more data to read
        if not mem_list:
            return 0

        seq = self._current_seq_num[station_name]
        curr_index = self.current_indexes[station_name]
        start = self._read_epoch_time(self, mem_list[curr_index])
        #minimum_length = max(start + 24 * self._offset * 4 * 60, self._offset * self._seq_length * 60)
        minimum_length = self._offset * self._seq_length * 60
        while seq != -1:
            end = self._read_epoch_time(self, mem_list[-1])
            diff = end - start # Difference in seconds
            if diff > minimum_length:
                return 1
            print('Loading segment for station \'%s\'' %station_name)
            if self._load_segment(station_name) != 1:
                return 0
            # Continuing
            seq = self._current_seq_num[station_name]
            start = end
        return 0


    # Function that computes all possible sequences
    def _detect_sequences_in_data(self, station_name):
        seq_list = []
        # First check
        test = self._validate_data_segment(station_name)
        # No more data
        if test != 1:
            return []

        seen = {}
        index = self._current_indexes[station_name]
        # First pass: collect the values
        for i, elem in enumerate(self._memory_segments[station_name]):
            seen[self._read_epoch_time(elem)] = i

        # Loop
        end_times = seen.keys.copy()
        for in in range(self._seq_length):


        for i in range(len(end_times)):
            end_times[i] += self._offset * 60 * self._seq_length
        # Third pass: traverse the list and report the largest number in the previous list
        for i, elem in enumerate(end_times):
            if elem not in seen:
                # No need to continue
                break
            last = seen[self._read_epoch_time(elem)]
        return first, last


    # Public method that generates a batch of sequences
    def generate_batch(self):
        # Initialization
        list_of_sequences = []
        batch_size = self._batch_size
        batch = []

        # Stations Ids
        station_Ids = [i for i in range(len(self._stations_names))]

        # Main loop
        i = 0
        while i < batch_size:
            if self._shuffle:
                shuffle(station_Ids)

            station_id = 0
            for j in enumerate(station_Ids):
                station_id = j
                station_name = self._stations_names[station_id]
                indexes = self._detect_sequences_in_data(station_name)
                if not indexes:
                    # No sequence found
                    continue
                # Generating new sequence
                sequence = []
                station_name = self._stati
                ons_names[station_id]
                memory_segment = self._memory_segments[station_name]
                sequence += memory_segment[first:last + 1] # TODO: Copy??

                # Update the current index
                new_index = last + 1
                if new_index > self._CHUNK_SIZE:
                    print('INFO: Truncating memory of station %s' % station_name)
                    self._truncate_memory_segment(station_name, new_index)
                    self._current_indexes[station_name] = 0
                else:
                    self._current_indexes[station_name] = new_index

                # Adding the sequence to the batch
                batch.append(sequence)
        # End
        return batch








