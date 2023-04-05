import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
import torchvision
from torch.utils.tensorboard import SummaryWriter


class SupervisedMLFramework:

    def __init__(self, model_name, model, train_dataset, test_dataset) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #Move model to GPU if available
        self.model = model.to(self.device)

        self.writer = SummaryWriter()
    
    def __del__(self):
        self.writer.close()
        
    def test(self, loss_function, batch_size=32):
        
        test_dataloader = DataLoader(self.test_dataset, batch_size)

        testing_loss, correct = 0, 0
        for batch, (X, y) in enumerate(test_dataloader):

            print(X.shape)

            X = X.to(self.device)
            y = y.to(self.device)

            prediction = self.model(X)
            loss = loss_function(prediction, y)

            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

            testing_loss += loss.item()

        print(f"% correct: {correct / len(self.test_dataset)}")
        return testing_loss / len(self.test_dataloader)

    def train(self, epochs, loss_function, optimizer, scheduler, save_dir, batch_size=32):

        train_dataloader = DataLoader(self.train_dataset, batch_size)

        for epoch in range(epochs):
            
            print(f"\n {'-'*10} Epoch {epoch} {'-'*10} ")
            total_batch_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):

                X = X.to(self.device)
                y = y.to(self.device)

                prediction = self.model(X)
                loss = loss_function(prediction, y)

                #Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    avg_loss_over_batch, current = loss.item(), (batch + 1) * len(X)
                    total_batch_loss += avg_loss_over_batch
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.train_dataset):>5d}]")

            epoch_validation_loss = self.test(loss_function)

            avg_epoch_loss = total_batch_loss / len(train_dataloader)

            scheduler.step()

            self.writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
            self.writer.add_scalar('Loss/validation', epoch_validation_loss, epoch)

            if epoch % 20 == 0 and epoch != 0:
                print(f"\n {'-'*10} Saving Checkpoint {epoch} {'-'*10} ")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_validation_loss
                }, save_dir + self.model_name + ".pt")
     
    def tune(self,  lr, epochs, loss_function, optim, batch_size=32, k=5):

        optimizer = optim(self.model.parameters(), lr=lr)

        #Do k fold cross validation on train portion.
        indices = np.random.permutation(range(len(self.train_dataset)))
        fold_size = int(len(indices) / k)
        k_fold_indices = list(range(len(indices)))

        split_validation_losses = []
        for split in range(1, k + 1):
            #If length of data isn't perfectly divisible by k, give the last fold whatever's left over (a max of k-1 samples) 
            validation_indices = k_fold_indices[(split -1) *fold_size: split *fold_size] if split != k else k_fold_indices[(split -1) *fold_size:]
                        
            train_indices = list(set(k_fold_indices)  - set(validation_indices))

            train_sampler = SequentialSampler(indices[train_indices])
            validation_sampler = SequentialSampler(indices[validation_indices])

            train_dataloader = DataLoader(self.train_dataset, batch_size, sampler=train_sampler)
            validation_dataloader = DataLoader(self.train_dataset, batch_size, sampler=validation_sampler)
            
            total_validation_loss = 0
            for epoch in range(epochs):
                
                print(f"\n {'-'*10} Epoch {epoch} {'-'*10} ")

                total_train_loss = 0
                for batch, (X, y) in enumerate(train_dataloader):
                    
                    X = X.to(self.device)
                    y = y.to(self.device)

                    prediction = self.model(X)
                    loss = loss_function(prediction, y)

                    #Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if batch % 100 == 0:
                        loss, current = loss.item(), (batch + 1) * len(X)
                        total_train_loss += loss
                        print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_sampler):>5d}]")
                
                average_epoch_train_loss = total_train_loss / len(train_dataloader)

                self.writer.add_scalar('Loss/train', average_epoch_train_loss, epoch)

                print(f"\n {'-'*10} Validating (epoch {epoch}) {'-'*10} ")

                epoch_validation_loss = 0
                num_correct = 0
                for batch, (X, y) in enumerate(validation_dataloader):
                    with torch.no_grad():

                        X = X.to(self.device)
                        y = y.to(self.device)

                        prediction = self.model(X)
                        loss = loss_function(prediction, y)

                        num_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

                        if batch % 100 == 0:
                            loss, current = loss.item(), (batch + 1) * len(X)
                            epoch_validation_loss += loss

                avg_epoch_validation_loss = epoch_validation_loss / len(validation_dataloader)
                total_validation_loss += avg_epoch_validation_loss
                print(f"Average batch validation loss for epoch {epoch}: {avg_epoch_validation_loss}")
                print(f"Percentage correct in validation pass: {num_correct / len(validation_indices)}")


            print(f"Average validation loss over all epochs: {total_validation_loss / epochs}")
            split_validation_losses.append(total_validation_loss)

        avg_validation_loss_all_splits = sum(split_validation_losses) / len(split_validation_losses)
        print(f"Average validation loss over all splits: {avg_validation_loss_all_splits}")

    def predict(self, sample):

        sample = sample.to(self.device)
        with torch.no_grad():
            return self.model(sample)
