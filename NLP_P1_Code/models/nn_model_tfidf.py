import sys, os
from collections import OrderedDict
from torch import nn, optim
from torchtext.legacy import data
from torchtext.vocab import Vectors, GloVe
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
#sys.path.insert('path') #Modify these
from basics import makedir, dump_pickle, read_pickle #Modify this

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#tensor = tensor.to("cuda:0" if torch.cuda.is_available() else "cpu") # needs assignment
#model.to('cuda:0')
class TFIDFDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self,index):
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.X_data)

def linear_block(in_features, out_features, p_drop, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p = p_drop)
        )

class dense_model(nn.Module):
    """
    Function defines a fully connected neural network architecture with optional Dropout layers,
    :param hidden layers: how many linear layers we want of whcih size
    :param num_features: number of tfidf or other features
    :param num_class: how many classes we are predicting
    :param p_drop: dropout layer to prevent overfitting
    :return: Keras model ready for compilation
    """
    def __init__(self, hidden_layers, num_features, num_classes, p_drop = .15):
        super(dense_model, self).__init__()
        
        linear_blocks = []
        for i, layer in enumerate(hidden_layers):
            if i == 0:
                linear_blocks.append(linear_block(num_features, layer, p_drop))
            else:
                linear_blocks.append(linear_block(hidden_layers[i-1], layer, p_drop))
        
        self.linear = nn.Sequential(
            *linear_blocks
        )
        #For Binary Classificaiton
        if num_classes == 2:
            num_classes = 1
        
        self.out = nn.Sequential(
            nn.Linear(hidden_layers[-1], num_classes)
        )
    
    def forward(self, x):
        x = self.linear(x)
        return self.out(x)
    

    
def train_nn(model, train_loader, valid_loader, NUM_EPOCHS, binary = False, LR = .001, LR_patience = None,
             early_stopping_callback = None):
    model.to(device)
    """
    trains model definedf by dense_model
    :param model: pass in the model created by dense model
    :param NUM_EPOCHS: decide how many epochs this model will be trained for
    :param criterion: use crossEntropyLoss unless a new application is suitable (like a binary deision with sigmoid)
    :param LR: initial learning rate
    :LR_patience: patience before downgrading learning rate
    :return: Keras model ready for compilation
    """
    
    if binary:
        criterion = nn.BCEwithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    accuracy_stat = {"train": [], "test" : []}
    loss_stat = {"train": [], "test" : []}
    stats = {"accuracy": accuracy_stat, "loss":loss_stat}
    optimizer = optim.Adam(model.parameters(), lr = LR)
    if LR_patience != None:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = LR_patience, factor = .1)
    
    for progress in tqdm(range(1, NUM_EPOCHS+1)):
        train_epoch_loss = 0
        train_epoch_acc = 0
        
        model.train()
        #Loop over training dataset using batches (use DataLoader to loaddata with batches)
        for x_train_batch, y_train_batch in train_loader:
            x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)

            #Clear Gradients
            optimizer.zero_grad()

            #Forward pass ->>>>
            #print(x_train_batch)
            y_train_pred = model(x_train_batch.float())

            #for binary classification
            if binary:
                y_train_pred = y_train_pred.flatten()
                y_train_batch = y_train_batch.float()

            #Find Loss and backporpogate gradients
            #print(y_train_pred)
            train_loss = criterion(y_train_pred, y_train_batch.long())
            train_acc = acc_calc(y_train_pred, y_train_batch, binary = binary)

            #backward <-------
            train_loss.backward()

            #update parameters (weights and biases)
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        
        #Now we validate out model
        with torch.no_grad():

            test_epoch_loss = 0
            test_epoch_acc = 0
            
            model.eval()
            for X_test_batch, y_test_batch in valid_loader:
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
                
                y_test_pred = model(X_test_batch.float())
                
                if binary == True:
                    y_test_pred = y_test_pred.flatten()
                    y_test_batch = y_test_batch.float()
                    
                test_loss = criterion(y_test_pred, y_test_batch.long())
                test_acc = acc_calc(y_test_pred, y_test_batch, binary = binary)
                
                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()
                
        if early_stopping_callback != None:
            early_stopping_callback(test_epoch_loss / len(valid_loader))
            if early_stopping_callback.stop_training:
                print(f'Training stopped -> Eearly Stopping Callback : test_loss: {test_epoch_loss/len(valid_loader)}')
                break
        stats['loss']['train'].append(train_epoch_loss/len(train_loader))
        stats['loss']['test'].append(test_epoch_loss/len(valid_loader))
        stats['accuracy']['train'].append(train_epoch_acc/len(train_loader))
        stats['accuracy']['test'].append(test_epoch_acc/len(valid_loader))
        
        # NN optimization
        clr = optimizer.param_groups[0]['lr']
        if LR_patience != None:
            scheduler.step(test_epoch_acc/len(valid_loader))
        
        print(f'EPOCH {progress + 0:03}: Loss: [Train: {train_epoch_loss/len(train_loader):.5f} | test {test_epoch_loss/len(valid_loader):.5f} ] Accuracy: [Train: {train_epoch_acc/len(train_loader):.3f} | test {test_epoch_acc/len(valid_loader):.3f}] LR: {clr}')
        
    return(model,stats)

#get accuracy for binary or multi-class
def acc_calc(y_pred, y_true, binary = False):
    if binary == False:
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        correct_pred = (y_pred_tags == y_true).float()
    elif binary == True:
        m = nn.Sigmoid()
        y_pred = m(y_pred)
        correct_pred = (y_pred>.5).bool() == y_true
    acc = correct_pred.sum() / len(correct_pred)
    n_digits = 3
    acc = torch.round(acc * 100 * 10**n_digits)/(10**n_digits)
    return acc

class EarlyStoppingCallback:
    def __init__(self, min_delta = 0.1, patience = 5):
        
        self.min_delta = min_delta
        self.patience = patience
        self.best_epoch_score = 0
        
        self.attempt = 0
        self.best_score = None
        self.stop_training = False
        
    def __call__(self, validation_loss):
        self.epoch_score = validation_loss
        
        if self.best_epoch_score == 0:
            self.best_epoch_score = self.epoch_score
        elif self.epoch_score > self.best_epoch_score - self.min_delta:
            self.attempt += 1
            #print(round(self.epoch?score,3),round(self.best_epoch_score,3)-self.delta)
            print(f'Message from callback (Early Stopping) counter: {self.attempt}/{self.patience}')
            if self.attempt >= self.patience:
                print('here')
                self.stop_training = True
            else:
                print('no')
        else: 
            self.best_epoch_score = self.epoch_score
            self.attempt = 0

def dump_model(model_dir, tfidf, label_encoder_in, model_args, model):
    model_dict = {}
    #
    model_dict['model_dir'] = model_dir
    makedir(model_dir,overwrite=True)
    
    dump_pickle(tfidf_in,os.path.join(model_dir,'tfidf'))
    dump_pickle(label_encoder_in,os.path.join(model_dir, ' label_encoder'))
    
    torch.save(toch.nn.DataParallel(model).state_dict(), os.path.join(model_dir,'model_state_dict') )
    
    model_dict['args'] = model_args
    dump_pickle(model_dict,os.path.join(model_dir,'model_dict'))
    return model_dict

def load_model(model_dir):
    model_dict = read_pickle(os.path.join(model_dir,'model_dict'))
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model_out.load_state_dict(new_state_dict)
    return tfidf,le,model_out