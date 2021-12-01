'''
Main file where training is done
'''
import helper_functions as hf
from models import Qnet


'''
Import and clean dataset
'''
data_df = pd.read_pickle('datasets/all_players_rel_all_actions_group_team.pkl')
data_df.loc[:, 'next_action'] = data_df.groupby('playIndex').action.shift(-1)
# drop na if using sarsa model
sarsa_df = data_df.dropna(axis=0, how='any', inplace=False)

'''
Get dataset splits
'''
dataset_seed = 2430
training_proportion = 0.7
train_data, val_data, test_data = hf.split_data(sarsa_df, training_proportion, dataset_seed)

'''
Define hyperparameters
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 2
BATCH_SIZE = 64
state_size = len(data_df.loc[0,'state'])
action_size = 2
gamma = 0.99

'''
Create models/optimizers/loss function
'''
eval_net = Qnet(state_size, action_size).to(device=device)
# make sure to reset parameters (? - don't really need)
for layer in eval_net.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
# define target net
target_net = type(eval_net)(state_size, action_size).to(device=device)
target_net.load_state_dict(eval_net.state_dict())
# define loss function
loss_fn = nn.MSELoss()
# define optimizers
optimizer = optim.Adam(eval_net.parameters())

'''
Train model
'''
training_loss_list = hf.train_sarsa(NUM_EPOCHS, BATCH_SIZE, train_data):

#### get test loss and validation loss plot
test_loss_model = test_loop(test_data, eval_net, F.mse_loss, device)
print(f"Trained model test MSE: {test_loss_model}")
hf.plot_history(loss_list, 'MSE Loss', 'training_plots/val_loss.png')

#### Save model
torch.save(eval_net.state_dict(), 'model_files/sarsa_two.pt')

#test_loss_rb = test_loop(test_data, untrained_net, F.mse_loss, device)
#print(f"Random baseline model test MSE: {test_loss_rb}")

'''
Get percentile of true action
'''
NUM_TEST_ROWS = 1000
NUM_SIMULATIONS = 100
percentile_list = []
for i in range(NUM_TEST_ROWS):
    
    percentile_rank = test_quantile_for_play(test_data, i, NUM_SIMULATIONS)
    
    #if percentile_rank < 100:
    #    print(f"at index {i} percentile rank {percentile_rank}") 
    percentile_list.append(percentile_rank)








