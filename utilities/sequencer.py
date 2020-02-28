from utilities.sequencer_utils import intersect_sorted_lists
from utilities.sequencer_utils import ONE_HOUR
from utilities.sequencer_utils import ONE_DAY
from random import shuffle
import pickle as pkl


# A Class used for reading sequences of data
# A Class used for reading sequences of data
class Sequencer:

    # Init method
    def __init__(self, stations_names, stations_mappings, offset=1800, seq_length=0, max_batch_size=50):
        self._MAX_MEM_SIZE = 1000
        self._stations_names = stations_names
        self._stations_mappings = stations_mappings
        self._offset = offset
        self._seq_length = seq_length
        self._max_batch_size = max_batch_size
        self._current_indexes = {name: 0 for name in stations_names}
        self._current_seq_num = {name: 0 for name in stations_names}
        self._memory_segments = {name: [] for name in stations_names}
        self._number_of_memory_loads = 0
        print('Calling sequencer with offset = %d, seq_length = %d and batch size = %d' %( offset, seq_length, max_batch_size))

    # Checks whether we have more
    def _get_seq_number(self, station_name):
        seq_number = self._current_seq_num[station_name]
        return seq_number

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
            self._current_seq_num[station_name] = -1  # No more data to load
            return -1

        # Valid sequence number
        self._current_seq_num[station_name] = seq_number
        return seq_number

    # Loading memory segments (lists) and updates sequences tracking mechanism
    def _load_segment(self, station_name):
        # check station name
        if station_name not in self._stations_names:
            print('ERROR: station %s not found' % station_name)
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
            print('Could not load file \'%s\'' % filepath)
            return -1

        # Empty list: should not happen
        if not mem_file:
            print('WARNING: loaded empty list from %s' % filepath)
            return -1

        # Updating internal counter
        self._number_of_memory_loads += 1

        # No change in index here
        if not self._memory_segments[station_name]:
            self._update_index(station_name, 0)

        # Updating the data
        self._memory_segments[station_name] += mem_file

        # number of memory chunks successfully loaded
        return 1

    # Truncates unwanted data for better memory management
    def _truncate_memory_segment(self, station_name, size):
        mem_list = self._memory_segments[station_name]
        if mem_list:
            self._memory_segments[station_name] = mem_list[size:]
            

    # Avoids the lists to grow exponentially
    def _adjust_memory_segment(self, station_name):
        index = self._current_indexes[station_name]
        if index > self._MAX_MEM_SIZE:
            print('INFO: Memory cleanup...')
            self._truncate_memory_segment(station_name, index)
            # Update the index
            self._update_index(station_name, 0)

    # Reads epoch time from sample
    def _read_epoch_time(self, cell):
        return cell['iso-datetime']

    # Reads day from sample
    def _read_day(self, cell):
        return cell['day']

    # Prepares 24h data segment. Returns the last index if segment found, -1 otherwise
    def _get_valid_data_segment(self, station_name):
        mem_list = self._memory_segments[station_name]
        seq = self._current_seq_num[station_name]
        if seq == -1:
            # No more data to read
            print('WARNING: no more data available for station %s' % station_name)
            return -1

        # Data segment not loaded yet
        if not mem_list:
            print('INFO: Loading segment for station \'%s\'' % station_name)
            test = self._load_segment(station_name)
            if test != 1:
                print('WARNING: Failed to get valid segment for station \'%s\'' % station_name)
                return -1

        # Checking first if we have enough data
        curr_index = self._current_indexes[station_name]
        size = len(mem_list)
        now = self._read_epoch_time(mem_list[curr_index])
        then = self._read_epoch_time(mem_list[size - 1])
        if (then - now) < ONE_DAY:
            print('INFO: Loading segment for station \'%s\'' % station_name)
            attempt = self._load_segment(station_name)
            if attempt != 1:
                print('WARNING: Failed to get valid segment for station \'%s\'' % station_name)
                # Might be the last memory chunk? 
                return size - 1
            print('INFO: sequence number for station %s is now %d' % (station_name, seq))

        # Getting 1 day segment
        next_day = -1
        today = self._read_day(mem_list[curr_index])
        last_index = 0
        for last_index in range(curr_index + 1, len(mem_list)):
            next_day = self._read_day(mem_list[last_index])
            if next_day != today:
                break
        # Even though we're still in the same day, we do return the segment
        return last_index

    # Function that computes all possible sequences
    def _detect_sequences_in_segment(self, start_index, segment):
        seq_list = []
        seen = {}
        minimum_length = 4

        # First: collect the values
        for i, elem in enumerate(segment):
            seen[self._read_epoch_time(elem)] = start_index + i

        seen_keys = sorted(list(seen.keys()))
        end_times = seen_keys.copy()
        # Second: Add offset k times and eliminate values
        for k in range(1,4):
            # Adding the offset length
            for i in range(len(end_times)):
                end_times[i] += ONE_HOUR * k

            # Computing intersection
            end_times = intersect_sorted_lists(end_times, seen_keys)

        # Last: pickup indexes
        for e in end_times:
            sequence = []
            for k in [0, 3, 5, 6]:
                val = e - ONE_HOUR * k
                sequence.insert(0, seen[val])
            seq_list.append(sequence)
        return seq_list

    # Updates the index
    def _update_index(self, station_name, new_index):
        self._current_indexes[station_name] = new_index

    # Public method that generates a batch of sequences
    def generate_batch(self):
        # Initialization
        list_of_sequences = []
        batch_size = self._max_batch_size
        batch = []
        # Stations Ids
        #station_Ids = [0]
        station_Ids = [i for i in range(len(self._stations_names))]

        # Main loop
        oos_stations = 0
        while len(batch) < batch_size and oos_stations < len(station_Ids):
            for station_id in station_Ids:
                station_name = self._stations_names[station_id]
                print('INFO: preparing data for station %s...' %station_name)
                memory_segment = self._memory_segments[station_name]
                start_index = self._current_indexes[station_name]
                #print('INFO: station %s. Initial index is %d...' % (station_name, start_index))
                end_index = self._get_valid_data_segment(station_name)
                if end_index == -1:
                    self._update_index(station_name, len(memory_segment) - 1)
                    # self._adjust_memory_segment(station_name)
                    print('INFO: station %s: NO sequence found, updating index to %d and continuing...'
                          % (station_name, self._current_indexes[station_name]))
                    oos_stations += 1
                    continue

                print('INFO: station %s, reading segment[%d, %d]...' % (station_name, start_index, end_index-1))
                view = memory_segment[start_index:end_index]

                print('INFO: station %s: sequence detection...' % (station_name))
                sequences_list = self._detect_sequences_in_segment(start_index, view)
                if not sequences_list:
                    # No sequence found
                    self._update_index(station_name, end_index)
                    # self._adjust_memory_segment(station_name)
                    print('INFO: station %s: NO sequence found, updating index to %d and continuing...'
                          % (station_name, end_index))
                    continue
                print('INFO: station %s: found %d sequences...' % (station_name, len(sequences_list)))
                # updating current index
                self._update_index(station_name, end_index)
                print('INFO: station %s new index is %d...' % (station_name, self._current_indexes[station_name]))

                # Cleaning up memory in case list have became very large
                # self._adjust_memory_segment(station_name)

                # Append sequence to the batch
                print('INFO: station %s: Adding the data to the batch...' % (station_name))
                # Copying over the data
                for seq in sequences_list:
                    sequence = []
                    for s in seq:
                        sequence.append(memory_segment[s])
                    batch.append(sequence)

                # Memory cleanup
                #for station_id in station_Ids:
                #    self._adjust_memory_segment(station_name)
        return batch
