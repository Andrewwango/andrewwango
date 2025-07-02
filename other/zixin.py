import torch
import deepinv as dinv

device = "cpu"

def pol2car(x):
    pass

def A_operator(x):
    return ...

class ConvolutionPhysics(dinv.physics.Physics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def A(self, x):
        return A_operator(x)
    
    def A_adjoint(self, y):
        return ...
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, physics):
        self.physics = physics
    
    def __len__(self):
        return ...
    
    def __getitem__(self, i):
        x = ... # |_
        y = self.physics.A(x)
        return x, y


physics = ConvolutionPhysics()

dataset = CustomDataset()
train_dataset = ...
test_dataset = ...

loss = dinv.loss.MCLoss()

class CNN(torch.nn.Module):
    def forward(self, y):
        ...

class CustomModel(CNN):
    def forward(self, y, *args, **kwargs):
        y_car = pol2car(y)
        super().forward(y_car)

model = CustomModel().to(device)

optimizer = torch.optim.Adam(model.parameters())

trainer = dinv.Trainer(
    model=model,
    losses=loss,
    physics=physics,
    train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=...),
    eval_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=...),
    epochs=20,
    optimizer=optimizer,
    device=device,
    wandb=True # You can use Weights and Biases (https://wandb.ai/) to easily track your training progress and evaluation
)

trainer.train()
trainer.test(torch.utils.data.DataLoader(test_dataset, batch_size=...))