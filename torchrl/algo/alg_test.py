import torch 

state_trajectory = {}

# means task 0 has 2 success trajs.
state_trajectory[0] = [torch.randn((3,13)),torch.randn((3,13))]
state_trajectory[3] = [torch.randn((3,13)),torch.randn((3,13))]

def get_traj_availability(state_trajectory):
    key_list = []
    for key,value in state_trajectory.items():
        if len(value) > 0:
            key_list.append(key)

    return key_list

def clip_by_window(list_of_trajs,window_length):
    if len(list_of_trajs) > window_length:
        return list_of_trajs[-window_length:]
    return list_of_trajs


one_hot_map = {}
for i in range(10):
    new = torch.zeros((1,10))
    new[0][i]=1
    one_hot_map[i] = new


def update_masks(state_trajectory):
        # First, update encoder

    recent_window = 5
    recent_traj = {}
    tasks = get_traj_availability(state_trajectory)
    print("tasks",tasks)
    for each_task in tasks:
        recent_traj[each_task] = clip_by_window(state_trajectory[each_task],recent_window)

    # Okay, here you will have a dict of recent_success trajs for some of the tasks.

    batch_traj_list = []
    batch_id_list = []
    for key,value in recent_traj.items():
        task_traj_amount = len(value)
        batch_traj_list += value
        
        for _ in range(task_traj_amount):
            batch_id_list.append(one_hot_map[key])

    assert len(batch_traj_list) == len(batch_id_list)

    task_traj_batch = torch.cat(batch_traj_list).float()#.to(self.device)
    task_onehot_batch = torch.cat(batch_id_list)


    print(task_onehot_batch)

update_masks(state_trajectory)