
class StateTrajBuffer():
    def __init__(self, capacity_each_task, task_names):
        self.capacity_each_task = capacity_each_task
        self.buffer = {}
        self.traj_pointers = {}

        # Initialize buffer.
        for each_name in task_names:
            self.buffer[each_name] = []
            # Always point to an empty idx.
            self.traj_pointers[each_name] = 0

    def add_trajectory(self, traj, task_name):
        # each traj should be like [length, dimension]

        idx = self.traj_pointers[task_name]



        self.traj_pointers[task_name] = (idx+1) % self.capacity_each_task
