import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']  
folder_path = './data'                   
prediction_data_path = './new_data/raw_data-nflx.csv'  

num_epochs = 5000                        
batch_size = 100                         
hidden_size = 30                         
output_size = 1                          
learning_rate = 0.0001                   
train_ratio = 0.7                        
val_ratio = 0.2                          


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TennyDataset(TensorDataset):
    
    def __init__(self, folder_path, label_position, device='cpu', scaler=None, fit_scaler=False):
        super(TennyDataset, self).__init__()
        self.folder_path = folder_path

        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        self.label_position = label_position
        self.device = device
        self.scaler = scaler

        self.train_files, self.val_files, self.test_files = self.split_data(file_names, train_ratio, val_ratio)
        data_df = self.read_and_clean_data(self.train_files)
        self.features, self.labels = self.prepare_features_labels(data_df)

        
        self.features, self.labels = self.tensors_to_device(self.features, self.labels, device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device)

    def tensors_to_device(self, features, labels, device):
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return features_tensor.to(device), labels_tensor.to(device)

    def split_data(self, file_names, train_ratio, val_ratio):
        total_files = len(file_names)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)

        train_files = file_names[:train_size]
        val_files = file_names[train_size:train_size + val_size]
        test_files = file_names[train_size + val_size:]

        return train_files, val_files, test_files

    def clean_data(self, df):
        
        df_cleaned = df.copy()  

        
        df_cleaned.replace(NON_NUMERIC_PLACEHOLDERS, pd.NA, inplace=True)

        
        df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')

        
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype == 'float64' or df_cleaned[column].dtype == 'int64':
                df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)

        return df_cleaned

    def read_and_clean_data(self, files):
        data = pd.DataFrame()
        for file in files:
            file_path = os.path.join(self.folder_path, file)
            temp_df = pd.read_csv(file_path, index_col=0)
            temp_df = temp_df.transpose()  
            temp_df = self.clean_data(temp_df)  

            
            data = pd.concat([data, temp_df], ignore_index=True)

        data = pd.DataFrame(data)  
        data.fillna(data.mean(), inplace=True)
        return data

    def prepare_features_labels(self, data_df):
        
        label_idx = self.label_position - 1
        labels = data_df.iloc[:, label_idx]  

        
        features = data_df.drop(data_df.columns[label_idx], axis=1)

        
        return features.values, labels.values

    def process_and_tensor_conversion(self, files, scaler=None, fit_scaler=False):
        data_df = self.read_and_clean_data(files)

        if fit_scaler:
            features = data_df.iloc[:, 1:].values  
            scaler.fit(features)

        features_scaled = scaler.transform(data_df.iloc[:, 1:].values)
        labels = data_df.iloc[:, 0].values.reshape(-1, 1)  

        
        features_tensor, labels_tensor = self.tensors_to_device(features_scaled, labels, self.device)

        
        super(TennyDataset, self).__init__(features_tensor, labels_tensor)


class TennyPredictionDataset(TennyDataset):
    def __init__(self, file_path, label_position, device='cpu', scaler=None, fit_scaler=False):
        super(TennyDataset, self).__init__()
        self.file_path = file_path
        self.folder_path = ''
        self.label_position = label_position
        self.device = device
        self.scaler = scaler

        data_df = self.read_and_clean_data([file_path])
        self.features, self.labels = self.prepare_features_labels(data_df)

        
        self.features, self.labels = self.tensors_to_device(self.features, self.labels, device)



class Tenny(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Tenny, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  

    
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x



tenny_dataset = TennyDataset(folder_path, label_position=1, device=device)


scaler = StandardScaler()


train_dataset = TennyDataset(folder_path, label_position=1, device=device, scaler=scaler, fit_scaler=True)
val_dataset = TennyDataset(folder_path, label_position=1, device=device, scaler=scaler)
test_dataset = TennyDataset(folder_path, label_position=1, device=device, scaler=scaler)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


input_size = train_dataset.features.shape[1]  
model = Tenny(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
model = model.to(device)  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
best_val_loss = float('inf')  


patience = 10
no_improve = 0


for epoch in range(num_epochs):
    
    model.train()  
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  

        
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch + 1}")
            break

        
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

    
    model.eval()  
    val_loss = 0  
    with torch.no_grad():  
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            val_loss += criterion(outputs, labels).item()  
    val_loss /= len(val_loader)  

    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

    
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        
    else:
        
        no_improve += 1
        if no_improve == patience:
            print("No improvement in validation loss for {} epochs, stopping training.".format(patience))
            break
    model.train()  


model.eval()  
test_loss = 0  
with torch.no_grad():  
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)  
        test_loss += criterion(outputs, labels).item()  
test_loss /= len(test_loader)  
print(f"Average Test Loss: {test_loss:.4f}")


prediction_dataset = TennyPredictionDataset(file_path=prediction_data_path, label_position=1, scaler=scaler, device=device)


new_features_tensor = prediction_dataset.features


model.eval()  
with torch.no_grad():  
    predictions = model(new_features_tensor)
    predictions_np = predictions.cpu().numpy()  

print(predictions_np)
print(f"Number of predictions: {len(predictions_np)}")
