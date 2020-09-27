# Imports here

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import torch
import json
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets , models , transforms
import torchvision.models as models
from PIL import Image
import util
import json

#cat_names_to_categories
with open('cat_to_name.json' , 'r') as f:
    cat_to_name = json.load(f)


arch = {"densenet121" : 1024,
         "vgg16":25088,
         "alexnet" : 9216
        }


def build_network(structure = 'densenet121' , hidden_layer = 512 , lr = 0.01 , device = 'gpu'):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and device = 'gpu' else "cpu")
    
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)       
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True) 
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} invalid model".format(structure))
        
    
    #freez parameters    
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                          ('fc1' , nn.Linear(arch[structure] , hidden_layer)),
                          ('relu' , nn.ReLU()),
                          ('fc2' , nn.Linear(hidden_layer , 102)),
                          ('output' , nn.LogSoftmax(dim = 1))
                          ]))
        
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters() , lr)
        
        model.to(device)
        
        return model , optimizer , criterion
model,optimizer,criterion = build_network('densenet121' , 512 , 0.01 , device = 'gpu')
    
#time_for_training_our_network
def go_training(trainloaders , validloaders , model , optimizer , criterion , epochs = 20 , device = 'gpu'):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and device = 'gpu' else "cpu")
    
    steps = 0
    print_every = 5

    for e in range(epochs):
        running_loss = 0
        for inputs , labels in trainloaders:
            steps += 1
            inputs , labels = inputs.to(device) , labels.to(device)
            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps , labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs , labels in validloaders:
                        inputs , labels = inputs.to(device) , labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps , labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p , top_class = ps.topk(1 , dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloaders):.3f}.. "
                      f"Accuracy: {accuracy/len(validloaders):.3f}")

                running_loss = 0
                model.train()

#calc_accuracy_for_testing_set
def valid_on_testset(testloaders):
    
    test_loss = 0
    accuracy = 0
    model.to(device)

    with torch.no_grad():
        for inputs, labels in testloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)
                    
            test_loss += batch_loss.item()
                    
            # Calc_accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print('Accuracy on the test set images for the whole network: %d %%' % (100 * accuracy / len(testloaders)))
    
valid_on_testset(testloaders)

# TODO: Save the checkpoint 
def saving_checkpoint(train_datasets , structure , hidden_layer):
    model.class_to_idx = train_datasets.class_to_idx

    torch.save({'structure' :structure,
                'hidden_layer':hidden_layer,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(path = 'checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    hidden_layer = checkpoint['hidden_layer']
    model,_,_ = set_traind_model(structure , hidden_layer)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    
load_model('checkpoint.pth')  
print(model)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
     # TODO: Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image)
   
    trans_img = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    img_tensor = trans_img(img_pil)
    
    return img_tensor
    
img = (test_dir + '/1/' + 'image_06754.jpg')
img = process_image(img)
print(img.shape)

#show_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path , model , topk = 5 , device = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() and device = 'gpu' else "cpu")

    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()
    
    with torch.no_grad():
        img = img.to(device)
        output = model.forward(img)
    
    prop = F.softmax(output.data , dim = 1)
    
    return prop.topk(topk)

img = (test_dir + '/10/' + 'image_07104.jpg')
ps, tp_classes = predict(img , model , device = 'gpu')
print(ps)
print(tp_classes)



def display_topk():
    fig, ax = plt.subplots()

    index = 1
    path = test_dir + '/1/image_06743.jpg'
    ps = predict(path, model)
    image = process_image(path)

    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    ax1.title(cat_to_name[str(index)])


    a = np.array(ps[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(ps[1][0])]

    fig,ax2 = plt.subplots(figsize=(5,5))


    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(b)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_pos, a)

    plt.show()



display_topk()