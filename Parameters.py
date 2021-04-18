import os
import h5py
import numpy as np

def save(mode, iteration=None, parameters=None):
    if mode == 'w':
        os.makedirs('parameters', exist_ok=True)
        hf = h5py.File('parameters/parameters.h5', 'w'); hf.close()
    else:
        assert isinstance(parameters, dict)
        assert isinstance(iteration, int)
        hf = h5py.File('parameters/parameters.h5', 'a')
        group = hf.create_group(f"iter-{str(iteration).rjust(4, '0')}")
        for k in parameters.keys():
            group.create_dataset(k, data=parameters[k])
        hf.close()

def load(iteration):
    parameters = {}
    hf = h5py.File('parameters/parameters.h5', 'r')
    group = hf.get(f"iter-{str(iteration).rjust(4, '0')}")
    for k in group.keys():
        parameters[k] = np.array(group.get(k))
    hf.close()
    return parameters

def load_last():
    last_key = None
    
    with h5py.File('parameters/parameters.h5', 'r') as h:
        keys =  list(h.keys())
        keys.sort(key=lambda x: int(x.split('-')[1]))
        last_key = keys[-1]
        
    last_iteration = last_key.split('-')[1]
    print(f"[GRABBING] grabbing last parameters | Last Iteration: {last_key}")
    return (last_iteration, load(last_iteration))
    
    
    

def view():
    with h5py.File('parameters/parameters.h5', 'r') as h:
        print("Parameters File Keys:", h.keys())
    
    parameters = load(100)
    print("Keys:", parameters.keys())

if __name__ == '__main__':
    # Parameters File View
    view()
    
    # Grab Last Parameters
    iteration, parameters = load_last()
    print(f"Grabbing parameters for iteration {iteration}", parameters.keys())