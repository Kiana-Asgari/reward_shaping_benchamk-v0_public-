import numpy as np
"""
Buffer of Trajectories
=[
    [ [s0,s1,...,sT],
      [a0,....,aT-1],
      [R1,...,RT]     ]
      [[s0,a0],[s1,a1],...],
      .
      .
      .
  ]
"""


class replay_buffer():
    def __init__(self, max_size):
         self.max_size = max_size
         self.data = []
         self.ptr = 0


    def append(self, value):
        if self.full():
            self.data[self.ptr] = value
        else:
            self.data.append(value)

        self.ptr = (self.ptr + 1) % self.max_size



    def sample(self, sample_size):
        idxes = np.random.choice(len(self.data), sample_size)
        mini_batch = []

        for idx in idxes:
            mini_batch.append(self.data[idx])

        return mini_batch

    def __len__(self): return len(self.data)

    def full(self): return len(self.data) == self.max_size

