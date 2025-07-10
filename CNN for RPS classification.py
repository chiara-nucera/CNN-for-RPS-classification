#CNN for Rock-Paper-Scissors classification

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
import torch.nn as nn
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import functools
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#Visualisation of a batch of images
#Each image is transformed to be visualized correctly
def show_images(loader, class_names, n=8): #Loader is an iterator that returns a bach of (image, label)
    #n = 8 standard number
    images, labels = next(iter(loader)) #iter creates an iterator on Loader
    images = images[:n] #images now contains a batch of images
    labels = labels[:n] #labels contains a batch of labels 
    plt.figure(figsize=(15,2)) #Setting the dimension for the plotting space
    for i in range(n): #Iteration on each image to visualise 
        img = images[i].permute(1, 2, 0).numpy() #Conversion in an array numpy for the visualization
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1) #Denormalizing (for a correct visualization)
        #np.clip -> to ba sure that the values are in the valid interval
        plt.subplot(1, n, i+1) 
        plt.imshow(img) #Visualizing the image
        plt.title(class_names[labels[i].item()]) #Name of the class above the image
        plt.axis('off')
    plt.show()

#IMPLEMENTING THE CNNs
class FirstCNN(nn.Module):
    def __init__(self):
        super(FirstCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #Moves over the image and extracts features by calculating a dot product
        #The convolutional layer  applies some filters (kernel) to the image to find the patterns
        #The result of each filter is a feature map (new image) that shows where the patter has been found in the original image
        #Output of the convolution: [16, 128, 128] -> 16 feature maps made of 128x128 pixels each

        #3 input channels -> defined by the shape of the data
        #16 -> number of feature maps to produce (standard number)
        #3 -> 3x3 size of the kernel (standard number)
        #=> We have 3 kernels moving over the image and producing 16 feature maps as a result
        #The padding is required to preserve the spatial dimensions of the input image after convolution operations on a feature map
        #Without the padding operation we would risk to loose information at the border
        
        self.pool = nn.MaxPool2d(2, 2)
        #It takes 2x2 pixels (area of 4) and creates one pixel out of that => extracts the most important information

        self.fc1 = nn.Linear(16 * 50 * 50, 3) 
        #Fully connected layer (linear)
        #3 as the three classes Rock, Paper, Scissors

    def forward(self, x):
            #Applying the first convolution layer to the input (finding the patterns) and reducing the dimension of the feature maps      
            x = self.pool(torch.relu(self.conv1(x))) 
            #Flattening each example in a vector
            x = x.view(-1, 16 * 50 * 50)
            #Applying the linear fully connected layer to transform the vector in a vector of only three elements (Rock, Paper, Scissors)
            x = self.fc1(x)
            return x

class SecondCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(SecondCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     #conv1 conv2 conv3 = convolutional layers
        self.bn1 = nn.BatchNorm2d(32)   #bn1 bn2 bn3 = Batch normalization layers

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   
        self.bn3 = nn.BatchNorm2d(128)  
        
        self.pool = nn.MaxPool2d(2, 2)  ##Max pooling layers to reduce spatial dimensions: 100x100 -> 50x50 -> 25x25 -> 12x12 
        self.dropout = nn.Dropout(dropout)  #Rate of 0.3, to reduce overfitting
        self.fc1 = nn.Linear(128 * 12 * 12, 64) #fc1 fc2 = fully connected layers 
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   
        x = self.dropout(x) 
        x = x.view(-1, 128 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    #Second and third convolution layer => more filters, therefore the NN can now recognize more complex patterns
    #Each pooling layer reduces the dimension of the feature maps (128 -> 64 -> 32 -> 16) 
    #=> Summarize the most important information (the NN becomes faster)
    #Batch norm: normalizes the values of the feature maps after the convolution
    #This helps to stabilize the training (to avoid problems caused by too small or too large values)
    #Dropout: randomly shuts down some connections between the neurons to reduce overfitting 
    # => avoid the NN to memorize the training set => helps to generalize
    #Fully connected layers = receive the information extracted from the other layers and combine them to obtain the final prediction
    #Adding more fully connected layers, the NN can learn more complex combinations of extracted features to understand which class the image belongs to

class ThirdCNN(nn.Module):
    def __init__(self):
        super(ThirdCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) #conv1 conv2 conv3 conv4 = convolutional layers
        self.bn1 = nn.BatchNorm2d(32)   #bn1 bn2 bn3 bn4 = Batch normalization layers

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2) #100x100 -> 50x50 -> 25x25 -> 12x12 -> 6x6

        self.dropout_conv = nn.Dropout2d(0.25)  #Dropout layer

        #Fully connected layers with additional dropout layers (fc1 fc2 + dropout_fc1 dropout_fc2)
        self.fc1 = nn.Linear(256 * 6 * 6, 128)  #6x6 after 4 pooling
        self.dropout_fc1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, 64)
        self.dropout_fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # [32, 64, 64]
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # [64, 32, 32]
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # [128, 16, 16]
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))   # [256, 8, 8]
        x = self.dropout_conv(x)
        x = x.view(-1, 256 * 6 * 6) #Flatten

        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        return x


#DEFINING TRAINING AND VALIDATION FUNCTIONS
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cpu'):
    model = model.to(device)
    #train_loaders = bacthes from training data, val_loader = batches from validation data
    #creterion = loss function
    #num_epochs = number of training iterations over the full dataset
    train_losses, val_losses = [], []   #Lists to store losses for each epoch
    train_accuracies, val_accuracies = [], []   #Lists to store accuracies for each epoch

    for epoch in range(num_epochs):
        #Training
        model.train()   #Sets the model to training mode
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()   #Reset gradient to avoid accumulation
            outputs = model(images) #Forward pass to compute prediction
            loss = criterion(outputs, labels)   #Compute loss
            loss.backward()     #Backpropagation to compute gradients
            optimizer.step()    #Updating model weights using optimizer

            #Accumulating total loss weighted by batch size
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            #Calculating the number of correct predictions
            correct += (predicted == labels).sum().item() 
            total += labels.size(0) #Total of sample processed
        #Computing average training loss and accuracy for the epoch
        train_losses.append(running_loss / total)
        train_accuracies.append(correct / total)

        #Validation (per epoch)
        model.eval()    #Evaluation mode
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():   #Disabling gradient computation (for efficiency)
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        #Computing and storing validation loss and accuracy (no weight updates).
        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_losses[-1]:.4f}, Train acc: {train_accuracies[-1]:.4f} | "
              f"Validation loss: {val_losses[-1]:.4f}, Validation acc: {val_accuracies[-1]:.4f}")
    
        if scheduler is not None:
            #To use ReduceLROnPlateau (with loss validation)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses[-1])
            else:
                scheduler.step()


    #Returning epoch summary with the losses and the accuracies
    return train_losses, val_losses, train_accuracies, val_accuracies


#EVALUATION OF THE MODEL
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()    #Setting the model to evaluation mode ()
    all_preds, all_labels = [], []  #Lists to store predictions and true labels
    with torch.no_grad():   #Disabling gradient computation (for efficiency)
        for images, labels in test_loader:  
            images = images.to(device)         #Moving images to the device (CPU)
            outputs = model(images)            #Forward pass: get raw model outputs (logits)
            _, preds = torch.max(outputs, 1)   #Getting predicted class by taking index of max logit
            all_preds.extend(preds.cpu().numpy())     #Collecting predictions on CPU as numpy arrays
            all_labels.extend(labels.numpy())         #Collecting true labels as numpy arrays
    
    #Calculating evaluation metrics (using sklearn functions)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1

#Plotting accuracies and losses from training and validation
def plot_curves(train_accuracies, val_accuracies, train_losses, val_losses, model_name):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title(f'Accuracy - {model_name}')
    plt.subplot(1,2,2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title(f'Loss - {model_name}')
    plt.show()


#Showing some images that the model misclassified
def show_misclassified(model, loader, class_names, device='cpu', n=8):
    model.eval()    #Setting the model to evaluation mode ()
    images_shown = 0    #Initializing a counter for misclassified images shown
    plt.figure(figsize=(15, 2))     #Space to display the images
    with torch.no_grad():   #Disabling gradient calculations (for efficiency)
        for images, labels in loader:   #Iterating over batched of images and their true labels
            images, labels = images.to(device), labels.to(device)   #Moving data to the device (cpu)
            outputs = model(images)    #To make the prediction of the label
            _, preds = torch.max(outputs, 1)
            for i in range(images.size(0)):     #Iterate over the images in the batch
                if preds[i] != labels[i]:   #If the prediction is wrong (misclassified)
                    img = images[i].cpu().permute(1, 2, 0).numpy()  #Convert tensor image to numpy for plotting
                    #Denormalize image for correct colors
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                    
                    #Plot of the misclassified images
                    plt.subplot(1, n, images_shown+1)
                    plt.imshow(img)
                    #Showing the true label and the predicted label above the image
                    plt.title(f"True: {class_names[labels[i].item()]}\nPred: {class_names[preds[i].item()]}")
                    plt.axis('off')
                    #Checking if there are enough images shown
                    images_shown += 1
                    if images_shown >= n:
                        plt.show()
                        return
    plt.show()
#"""

#HYPERPARAMETER TUNING
#Automatic choice of best values for the external parameters (learning rate, batch size...)
#The code tries a moltitude of different combinations and find the best one basing on the performances
#
def objective(trial, train_dataset, val_dataset):
    #Hyperparameters to try during the optimization
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)  #Learning rate in log scale
    batch_size = trial.suggest_categorical('batch_size', [16, 24, 32, 48])
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)    #Dropout rate between 0.2 and 0.5
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    #Creating dataloaders with the suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #Setting GPU as the device if available (otherwise -> cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Inizialization of the model with the suggested dropout rate (SecondCNN takes dropout as a parameter in input)
    model = SecondCNN(dropout=dropout).to(device)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    patience = 4    #Early stopping patience
    best_val_acc = 0
    epochs_no_improve = 0

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        #Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_losses.append(running_loss / total)
        train_accuracies.append(correct / total)

        #Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_losses.append(val_loss / val_total)
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        #Early stopping (stop if no improvement for 'patience' epochs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        #Reporting intermediate results to Optuna
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()  #Stop unpromising trials early

        if epochs_no_improve >= patience:
            break   #Stop training early due to no improvement

    #Saving training history in trial attributes for later analysis
    trial.set_user_attr("train_accuracies", train_accuracies)
    trial.set_user_attr("val_accuracies", val_accuracies)
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_losses", val_losses)

    return best_val_acc     #Returns the best validation accuracy found

def plot_best_trials(study, top_n=3):
    #Sorting completed trials by validation accuracy (descending)
    sorted_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)
    for i, trial in enumerate(sorted_trials[:top_n]):
        train_accuracies = trial.user_attrs.get("train_accuracies", [])
        val_accuracies = trial.user_attrs.get("val_accuracies", [])
        train_losses = trial.user_attrs.get("train_losses", [])
        val_losses = trial.user_attrs.get("val_losses", [])
        plot_curves(train_accuracies, val_accuracies, train_losses, val_losses, model_name=f"Optuna Trial {trial.number} (val_acc={trial.value:.4f})")

def plot_confusion_matrix(model, loader, class_names, device='cpu', title="Confusion Matrix"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 15

    #Loading and splitting dataset
    dataset = datasets.ImageFolder(root='C:/Users/chiar/Desktop/ML Project/archive', transform=None)
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    #Transformation to apply (differentiated between train and validation/test)
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(), 
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Custom Dataset wrapper to apply transform
    class DatasetWithTransform(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = DatasetWithTransform(train_dataset, train_transform)
    val_dataset = DatasetWithTransform(val_dataset, val_test_transform)
    test_dataset = DatasetWithTransform(test_dataset, val_test_transform)

    #Uploading the custom images
    custom_dataset = datasets.ImageFolder(root="C:/Users/chiar/Desktop/custom_test", transform=val_test_transform)
    custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    #Using ImageFolder, for each image in each folder a pair (image, label) is created
    print(dataset.class_to_idx) #Check class mapping: shows the mapping from folder names to integer labels
    #{'paper': 0, 'rock': 1, 'scissors': 2}
    print(f"Total images: {len(dataset)}") #Total number of images
    class_names = list(dataset.class_to_idx.keys()) #List of images

    #EXPLORATORY DATA ANALYSIS
    #Checking for class imbalance: counting the number of images for each of the three folders (rock, paper, scissors) 
    labels = [label for _, label in dataset] #For each pair (image, label) only the label is considered and counted
    class_counts = Counter(labels)
    print("Images per class:", class_counts)

    #Checking if there are black images (to be eventually deleted)
    to_tensor = transforms.ToTensor()
    black = [] 
    for idx, (img, label) in enumerate(dataset):
        img_tensor = to_tensor(img)
        #img is a tensor of shape [3, 128, 128]
        if torch.sum(img_tensor) == 0: #if the sum of the pixels is zero, the image is completely black
            black.append(idx) #To save the index of the black image to eventually visualize and delete it later
    if black:
        print(f"{len(black)} black images with index: {black}")
    else:
        print("No black images")
    #In this case, the dataset does not contain black images


    #Creating dataLoaders for initial training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #FIRSTCNN feature map visualization
    print("-Printing a batch of images-")
    show_images(train_loader, class_names, n=8)

    model = FirstCNN() 
    #Set the model to evaluation mode
    model.eval()

    #Get a single image from the test set
    #Test_loader might be empty if test_size is too small or if it's already iterated through
    #Ensure test_loader has data or reset it if needed
    try:
        images_test, _ = next(iter(test_loader))
        image_to_visualize = images_test[0].unsqueeze(0).to(device)
    except StopIteration:
        print("Test loader is empty.")
        return  #Exit main if no image is available

    #Pass the image through the first convolutional layer
    with torch.no_grad(): #Disable gradient calculations 
        #model.conv1 is directly accessed as it's a module
        feature_maps = model.conv1(image_to_visualize)
    plt.figure(figsize=(15, 5))
    for i in range(min(8, feature_maps.shape[1])):
        plt.subplot(2, 4, i + 1)
        plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='gray')   #Feature maps are typically visualized in grayscale [1][3]
        plt.title(f'Feature map {i + 1}')
        plt.axis('off') #Hide axes
    plt.tight_layout() #Adjust layout to prevent overlapping titles/labels
    plt.show() #Display the plot

    #Training and evaluatation of the CNNs
    def train_and_eval(model_class, name):
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        #Adam = algorithm to optimize the weights with learning rate 0.001

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #Also tried
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)    #Also tried

        criterion = nn.CrossEntropyLoss()
        print(f"Training {name}")
        train_losses, val_losses, train_acc, val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
        )
        print(f"Evaluating {name} on test set:")
        evaluate_model(model, test_loader, device)
        plot_curves(train_acc, val_acc, train_losses, val_losses, name)
        return model
        #train_losses = loss on the training set for each epoch
        #val_losses = loss on the validation set for each epoch
        #train_acc = accuracy on the training set for each epoch
        #val_acc = accuracy on the validation set for each epoch

    #Inizializing the models
    model1 = train_and_eval(FirstCNN, "FirstCNN")
    model2 = train_and_eval(SecondCNN, "SecondCNN")
    model3 = train_and_eval(ThirdCNN, "ThirdCNN")

    show_misclassified(model1, test_loader, class_names, device=device, n=8)

    #Hyperparameter tuning with Optuna
    print("Optuna study started")
    study = optuna.create_study(direction="maximize")
    objective_func = functools.partial(objective, train_dataset=train_dataset, val_dataset=val_dataset)
    study.optimize(objective_func, n_trials=20, timeout=3600)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    #Extraction of the best hyperparameters
    best_bs = trial.params['batch_size']
    best_lr = trial.params['learning_rate']
    best_dropout = trial.params.get('dropout', 0.3)
    best_optimizer_name = trial.params['optimizer']

    #Creating new DataLoader with the best batch size
    train_loader_best = DataLoader(train_dataset, batch_size=best_bs, shuffle=True)
    #During the training part, the model receives a batch at a time, and after seeing all the images an epoch is complete
    #When a new epoch starts, the images are mixed and the process starts again
    #The order of the images is randomized => good for training the model because we avoid the model to learn the order of the images
    val_loader_best = DataLoader(val_dataset, batch_size=best_bs, shuffle=False)
    test_loader_best = DataLoader(test_dataset, batch_size=best_bs, shuffle=False)  #The order of the images is not randomized => good for testing

    #TRAINING SecondCNN WITH THE BEST HYPERPARAMETERS)
    #Initialization of the best model
    model_best = SecondCNN(dropout=best_dropout).to(device)

    #Selecting the optimizer
    if best_optimizer_name == 'Adam':
        optimizer_best = optim.Adam(model_best.parameters(), lr=best_lr)
    elif best_optimizer_name == 'SGD':
        optimizer_best = optim.SGD(model_best.parameters(), lr=best_lr, momentum=0.9)
    else:
        optimizer_best = optim.RMSprop(model_best.parameters(), lr=best_lr)

    criterion = nn.CrossEntropyLoss() #Loss function for classification

    #Training
    print("Training SecondCNN with best hyperparameters")
    train_losses_best, val_losses_best, train_acc_best, val_acc_best = train_model(
        model_best, train_loader_best, val_loader_best, criterion, optimizer_best, scheduler=None, num_epochs=10, device=device
    )
    plot_curves(train_acc_best, val_acc_best, train_losses_best, val_losses_best, "SecondCNN (Best Hyperparameters)")

    #Final evaluation
    print("Evaluating SecondCNN (best) on test set:")
    evaluate_model(model_best, test_loader_best, device)
    print("Confusion Matrix for SecondCNN on test set:")
    plot_confusion_matrix(model_best, test_loader_best, class_names, device=device, title="SecondCNN - Test Set")

    print("- SecondCNN with custom images -")
    evaluate_model(model_best, custom_loader, device=device)
    show_misclassified(model_best, custom_loader, class_names, device=device, n=6)
    print("Confusion Matrix for SecondCNN on custom images:")
    plot_confusion_matrix(model_best, custom_loader, class_names, device=device, title="SecondCNN - Custom Images")

    # Visualize best trials
    plot_best_trials(study, top_n=3)

#To ensure main() is called only when the script is executed directly
if __name__ == '__main__':
    main()

