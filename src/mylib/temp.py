from tqdm.notebook import tqdm
from sklearn import metrics
import torch

class Perceptron(torch.nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self, input_dim=784, num_layers=0, 
                 hidden_dim=64, output_dim=10, p=0.0):
        super(Perceptron, self).__init__()
        
        self.layers = torch.nn.Sequential()
        
        prev_size = input_dim
        for i in range(num_layers):
            self.layers.add_module('layer{}'.format(i), 
                                  torch.nn.Linear(prev_size, hidden_dim))
            self.layers.add_module('relu{}'.format(i), torch.nn.ReLU())
            self.layers.add_module('dropout{}'.format(i), torch.nn.Dropout(p=p))
            prev_size = hidden_dim
        
        self.layers.add_module('classifier', 
                               torch.nn.Linear(prev_size, output_dim))        
        
    def forward(self, input):
        return self.layers(input)

def train_teacher(teacher, train_data, test_data, phi=lambda x: x):
    
    #teacher = Teacher
    optimizer = torch.optim.Adam(teacher.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    epochs = 10

    for i in tqdm(range(epochs)):
        train_generator = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        teacher.train()
        for x, y in tqdm(train_generator, leave=False):
            optimizer.zero_grad()
            #x = x.to(device)
            x = x.view([-1, 784]).to(teacher.device)
            y = y.to(teacher.device)
            predict = teacher(phi(x))
            loss = loss_function(predict, y)
            loss.backward()
            optimizer.step()

        test_generator = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
        teacher.eval()
        for x, y in tqdm(test_generator, leave=False):
            #x = x.to(device)
            x = x.view([-1, 784]).to(teacher.device)
            y = y.to(teacher.device)
            predict = teacher(phi(x))
            loss = loss_function(predict, y) 
            
def distillation_train(student, train_data, test_data, teacher=None, T=1, phi=lambda x: x):   
    
    list_of_train_acc = []
    list_of_test_acc = []
    list_of_train_losses = []
    list_of_test_losses = []

    epochs = 20
    attempts = 3
    
    for attempt in tqdm(range(attempts)):
        #student = Student
        optimizer = torch.optim.Adam(student.parameters())
        loss_function = torch.nn.CrossEntropyLoss()
        
        train_acc = []
        test_acc = []
        train_losses = []
        test_losses = []
        
        for epoch in tqdm(range(epochs), leave=False):
            train_generator = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
            train_true = 0
            train_loss = 0
            for x, y in tqdm(train_generator, leave=False):
                optimizer.zero_grad()
                x = x.view([-1, 784]).to(student.device)
                y = y.to(student.device)
                student_output = student(x)
                
                if (teacher == None):
                    loss = loss_function(student_output, y)
                else:
                    teacher_output = teacher(phi(x))
                    loss = loss_function(student_output, y)\
                    - (torch.softmax(teacher_output/T, axis=1) *\
                       torch.log(torch.softmax(student_output/T, axis=1))).sum(axis=1).mean()

                loss.backward()
                optimizer.step()
                train_true += metrics.accuracy_score(y.cpu(), torch.argmax(student_output, axis=1).cpu())
                train_loss += loss.cpu().item()
                
            test_generator = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
            test_true = 0
            test_loss = 0
            for x, y in tqdm(test_generator, leave=False):
                x = x.view([-1, 784]).to(student.device)
                y = y.to(student.device)
                output = student(x)
                
                loss = loss_function(output, y)
                    
                test_true += metrics.accuracy_score(y.cpu(), torch.argmax(output, axis=1).cpu())
                test_loss += loss.cpu().item()
        
            train_acc.append(train_true*100/len(train_data))
            test_acc.append(test_true*100/len(test_data))
            train_losses.append(train_loss*100/len(train_data))
            test_losses.append(test_loss*100/len(test_data))
            
        list_of_train_acc.append(train_acc)
        list_of_test_acc.append(test_acc)
        list_of_train_losses.append(train_losses)
        list_of_test_losses.append(test_losses)
        
    return list_of_train_acc, list_of_test_acc, list_of_train_losses, list_of_test_losses