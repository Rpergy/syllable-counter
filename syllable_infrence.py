import csv
import torch
import torch.nn as nn

## HYPERPARAMETERS ##
lr = 0.05
lr_decay = 0.02
epochs = 200
batch_size = 7000

model_architecture = nn.Sequential(
                     nn.Linear(28, 30),
                     nn.Linear(30, 50),
                     nn.ReLU(),
                     nn.Linear(50, 80),
                     nn.ReLU(),
                     nn.Linear(80, 80),
                     nn.ReLU(),
                     nn.Linear(80, 50),
                     nn.ReLU(),
                     nn.Linear(50, 50),
                     nn.ReLU(),
                     nn.Linear(50, 50),
                     nn.ReLU(),
                     nn.Linear(50, 12),
                    )
# ---------------------

words = []
syllables = []

with open("phoneticDictionary.csv", "r") as file: # read data from file
    csvFile = csv.reader(file)

    for lines in csvFile:
        words.append(lines[1])
        syllables.append(lines[3])

vocabulary = { # list of vocabulary used in words
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

def encode(input): # encode features into list of integers
    output = []
    for letter in input:
        output.append(vocabulary[letter])
        
    return output

def decode(input): # decode features into a string
    output = ""
    for val in input:
        output += [i for i in vocabulary if vocabulary[i]==val][0]
    
    return output

features = []
labels = []

for word in words: # add encoded word to features list
    features.append(encode(word))

longest_len = 0
longest_index = 0

for word in range(len(features)): # find length/index of longest word
    if len(features[word]) > longest_len:
        longest_len = len(features[word])
        longest_index = word

for i in range(len(features)): # add a list of periods to the end of each word in order to maintain constant word length
    difference = longest_len - len(features[i])
    features[i].extend([26.0 for i in range(difference)])

labels = [nn.functional.one_hot(torch.tensor(int(label), dtype=torch.int64), 12).view(12).tolist() for label in syllables] # converts labels to one-hot vectors for inference 

class Model(nn.Module): # define network 
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
    nums = torch.randint(0, len(features), (1, batch_size)).view(batch_size)
    x = torch.tensor([features[i] for i in nums], dtype=torch.float32)
    y = torch.tensor([labels[i] for i in nums], dtype=torch.float32)

    optimizer.zero_grad()
    output = m(x)

    mse_loss = nn.MSELoss()
    loss = mse_loss(output, y)
    loss.backward()
    optimizer.step()

    print(f"Loss in Epoch {epoch}: {loss.item():0.4f} Learning Rate: {lr}")
    
    lr = (1 / (1 + lr_decay * epoch)) * initial_lr
    optimizer.param_groups[0]['lr'] = lr # decrease learning rate by the decay amount

while True:
    with torch.no_grad():
        wordInput = input("(Type ':exit' to exit) Enter a word: ")

        if wordInput == ":exit": break

        word = encode(wordInput)

        difference = longest_len - len(word)
        word.extend([26.0 for i in range(difference)])

        output = m(torch.tensor(word))

        highest_val = 0
        highest_index = 0

        for i in range(len(output)): # finds highest percentage 
            if output[i] > highest_val:
                highest_val = output[i]
                highest_index = i

        print(output)
        print(f"{highest_index} syllables, {highest_val.item() * 100:0.2f}% sure")
