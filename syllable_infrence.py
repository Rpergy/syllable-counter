import csv
import torch
import torch.nn as nn
import random

## HYPERPARAMETERS ##
lr = 0.05
lr_decay = 0.2
epochs = 1500

model_architecture = nn.Sequential(
                     nn.Linear(28, 50),
                     nn.ReLU(),
                     nn.Linear(50, 50),
                     nn.ReLU(),
                     nn.Linear(50, 50),
                     nn.ReLU(),
                     nn.Linear(50, 30),
                     nn.ReLU(),
                     nn.Linear(30, 20),
                     nn.ReLU(),
                     nn.Linear(20, 10),
                     nn.ReLU(),
                     nn.Linear(10, 1)
                    )

words = []
syllables = []

with open("phoneticDictionary.csv", "r") as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        words.append(lines[1])
        syllables.append(lines[3])

vocabulary = {
    'a': 0.0,
    'b': 1.0,
    'c': 2.0,
    'd': 3.0,
    'e': 4.0,
    'f': 5.0,
    'g': 6.0,
    'h': 7.0,
    'i': 8.0,
    'j': 9.0,
    'k': 10.0,
    'l': 11.0,
    'm': 12.0,
    'n': 13.0,
    'o': 14.0,
    'p': 15.0,
    'q': 16.0,
    'r': 17.0,
    's': 18.0,
    't': 19.0,
    'u': 20.0,
    'v': 21.0,
    'w': 22.0,
    'x': 23.0,
    'y': 24.0,
    'z': 25.0,
    '.': 26.0
}

value = [i for i in vocabulary if vocabulary[i]==3.0]
value[0]

def encode(input):
    output = []
    for letter in input:
        output.append(vocabulary[letter])
        
    return output

def decode(input):
    output = ""
    for val in input:
        output += [i for i in vocabulary if vocabulary[i]==val][0]
    
    return output

features = []
labels = []

for word in words:
    features.append(encode(word))

longest_len = 0
longest_index = 0

for word in range(len(features)):
    if len(features[word]) > longest_len:
        longest_len = len(features[word])
        longest_index = word

for i in range(len(features)):
    difference = longest_len - len(features[i])
    features[i].extend([26.0 for i in range(difference)])

labels = [float(i) for i in syllables]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = model_architecture
    
    def forward(self, x):
        return self.net(x)

m = Model()
optimizer = torch.optim.Adam(m.parameters(), lr)

print("Training Model...")

initial_lr = lr

for epoch in range(epochs):
    x = torch.tensor(features[random.randint(0, len(features))], dtype=torch.float32)
    y = torch.tensor(labels[random.randint(0, len(labels))], dtype=torch.float32).view(1)

    optimizer.zero_grad()
    output = m(x)


    mse_loss = nn.MSELoss()
    loss = mse_loss(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Model Loss in Epoch {epoch}: {loss.item()} Learning Rate: {lr}")
    
    lr = (1 / (1 + lr_decay * epoch)) * initial_lr
    optimizer.param_groups[0]['lr'] = lr

while True:
    wordInput = input("(Type ':exit' to exit) Enter a word: ")

    if wordInput == ":exit": break

    word = encode(wordInput)

    difference = longest_len - len(word)
    word.extend([26.0 for i in range(difference)])

    with torch.no_grad():
        output = m(torch.tensor(word))
        print(f"{wordInput} has {round(output.item())} syllables (Acutal Inference: {output.item()} syllables)\n")
