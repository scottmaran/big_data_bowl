'''
File that defines several useful functions
'''

import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import os
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data


'''
Function that splits the data into a training, validation, and test set
Input: 
    dataset - pandas dataframe
    train_split - proportion of dataset to use for training (e.g. 0.7)
    seed - integer specifying the seed to use for random shuffle
Output:
    train, validation, and test pandas dataframes
'''
def split_data(dataset, train_split, seed):
    np.random.seed(seed)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    train_num = int(len(dataset)*train_split)
    val_num = (len(dataset) - int(len(dataset)*train_split))//2

    train_indices = indices[0:train_num]
    val_indices = indices[train_num:train_num+val_num]
    test_indices = indices[train_num+val_num:]

    #check to make sure slices correct
    assert len(dataset) == len(train_indices) + len(val_indices) + len(test_indices)

    #dataset = help.normalize(train_indices, dataset)

    train_data = dataset.iloc[train_indices,:]
    val_data = dataset.iloc[val_indices,:]
    test_data = dataset.iloc[test_indices,:]

    return train_data, val_data, test_data


'''
Function that plot's the given history
metric (str): e.g. 'accuracy', 'loss'
'''
def plot_history(history_list, metric, filename):
    
    fig, ax = plt.subplots()

    # plotting
    ax.plot(list(range(1,len(history_list)+1)), history_list)
    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric}")
    plt.show()

    #file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    fig.savefig(filename)


    
'''
returns list of losses
'''
def train_sarsa(NUM_EPOCHS, BATCH_SIZE, train_data, verbose=True, automatic_stop=False):

    loss_list = []

    num_batches = int(len(train_data)/BATCH_SIZE)+1

    # define when to update the target network
    COPY_TARGETS_INDEX = int(num_batches/3)

    for k in range(NUM_EPOCHS):
        eval_net.train()
            
        for i in range(num_batches):
        
            # update weights of target network
            if i % COPY_TARGETS_INDEX == 0:
                target_net.load_state_dict(eval_net.state_dict())

            # get sample from data
            start_index = i*BATCH_SIZE
            end_index = min(len(train_data), (i+1)*BATCH_SIZE)
            training_batch = train_data.iloc[list(range(start_index, end_index)), :]

            '''
            Move all of data to torch tensors on proper device with proper type (float32)
            Extract state vectors and properly cast them into corrrect shape
            '''
            state = torch.tensor(np.stack(training_batch['state'].values), dtype=torch.float32).to(device=device)
            action = torch.tensor(np.stack(training_batch['action'].values), dtype=torch.float32).to(device=device)
            reward = torch.tensor(training_batch['reward'].values, dtype=torch.float32).to(device=device)
            next_state = torch.tensor(np.stack(training_batch['next_state'].values), dtype=torch.float32).to(device=device)
            next_action = torch.tensor(np.stack(training_batch['next_action'].values), dtype=torch.float32).to(device=device)
            
            # do this because forget to replace nans in next state in dataset construction
            next_state = torch.nan_to_num(next_state, nan=0)
            
            # Update Q
            eval_input = torch.cat((state, action), 1)
            q_eval = eval_net(eval_input)

            target_input = torch.cat((next_state, next_action), 1)
            q_target = torch.unsqueeze(reward,1) + gamma*target_net(target_input)

            loss = loss_fn(q_eval, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % int(num_batches/3) == 0) and (i != 0):
                avg_val_loss = test_loop(val_data, eval_net, F.mse_loss, device)
                loss_list.append((k,avg_val_loss))
                if verbose:
                    print(f"At epoch {k}, iter {i}: avg. val loss = {avg_val_loss}")
                eval_net.train()
                
            if automatic_stop:
                if len(loss_list) > 2:
                    # stops training if the loss doesn't decrease for two consecutive times
                    if (loss_list[-1][1] > loss_list[-2][1]) and (loss_list[-1][1] > loss_list[-3][1]):
                        return loss_list


 


'''
Unvectorized function that computes the loss for each row in the given dataset
'''
def test_loop(test_df, model, test_loss_fn, gamma, device):
    size = len(test_df)
    test_loss = 0
    model.eval()
    
    with torch.no_grad():
        for row_index in range(0,len(test_df)):
            #try:
            row = test_df.iloc[row_index, :]

            state = torch.tensor(row['state'], dtype=torch.float32).to(device=device)
            action = torch.tensor(row['action'], dtype=torch.float32).to(device=device)
            reward = torch.tensor(row['reward'], dtype=torch.float32).to(device=device)
            next_state = torch.tensor(row['next_state'], dtype=torch.float32).to(device=device)
            next_action = torch.tensor(row['next_action'], dtype=torch.float32).to(device=device)
            
            # do this because forget to replace nans in next state in dataset construction
            next_state = torch.nan_to_num(next_state, nan=0)

            # Update Q
            eval_input = torch.cat((state, action), 0)
            eval_input = torch.unsqueeze(eval_input, 0)
            q_eval = model(eval_input)

            target_input = torch.cat((next_state, next_action), 0)
            target_input = torch.unsqueeze(target_input, 0)
            q_target = reward + gamma*model(target_input)

            loss = test_loss_fn(q_eval, q_target)
            test_loss += loss
            #except Exception as e:
                #print(f"Row index: {row_index}")
                #print(F"Exception: {e}")
                #print(row)
    
        test_loss /= size
            #print(f"Avg loss: {test_loss:>8f} \n")

    return test_loss


'''
Given a play, we estimate the 'efficiency of the play' with respect to our trained Q-function.
We sample num_simulations number of potential actions that could have been taken, and compute
their Q_values. We then see what quantile the true action fell in
'''
def test_quantile_for_play(df, row_index, num_simulations, verbose=False):

    sample_row = df.iloc[row_index,:]
    state = torch.tensor(sample_row['state'], dtype=torch.float32).to(device=device)
    action = torch.tensor(sample_row['action'], dtype=torch.float32).to(device=device)
    q_value_dict = {}

    for k in range(num_simulations):

        '''
        Generate a simulated action
        '''
        radius = np.linalg.norm(sample_row['action']) * np.sqrt(np.random.uniform())
        theta = np.random.uniform()*2*np.pi
        sampled_action = np.array([radius*np.cos(theta), radius*np.sin(theta)])
        sampled_action = torch.tensor(sampled_action, dtype=torch.float32).to(device=device)
        # calculate simulated Q-value
        simulated_input = torch.cat((state, sampled_action), 0)
        simulated_input = torch.unsqueeze(simulated_input, 0)
        sim_q_value = eval_net(simulated_input)

        #print(f"Simulated Q-value: {output}")

        q_value_dict[sim_q_value] = sampled_action

    true_value_input = torch.cat((state, action), 0)
    true_value_input = torch.unsqueeze(true_value_input, 0)
    true_output = eval_net(true_value_input)

    if verbose:
        max_q = max(q_value_dict.keys())
        print(f"True Q-value: {true_output}")
        print(f"True action: {sample_row['action']}")
        print(f"Max q-value from simulations: {max_q}")
        print(f"Max Q action: {q_value_dict[max_q]}")

    cpu_true_output = true_output.cpu().detach().numpy()
    cpu_keys = torch.stack(list(q_value_dict.keys())).reshape(-1).detach().cpu()

    percentile = stats.percentileofscore(cpu_keys, cpu_true_output, kind='weak')
    percentile_rank = stats.percentileofscore(cpu_keys, cpu_true_output, kind='rank')
    
    if verbose:
        print(f"Percentile (weak): {percentile}")
        print(f"Percentile (rank): {percentile_rank}")

        print(f"99th percentile {np.percentile(cpu_keys, 99)}")
        print(f"100th percentile {np.percentile(cpu_keys, 100)}")
        print(f"True q-val: {true_output}")
        
    return percentile_rank