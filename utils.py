import time
import torch

def calculate_accuracy(logits, labels,device='cuda'):
    correct = 0
    total = 0
    # Get predictions (logits to class indices)
     #_, predicted = torch.max(outputs, dim=1)  # Choose the class with the highest score
    predictions = torch.argmax(logits, dim=1)        
    total += labels.size(0)  # Total number of samples
    correct += (predictions == labels)  # Number of correct predictions
    
    # Calculate accuracy
    accuracy = correct / total * 100  # In percentage
    return accuracy

def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' % (hour, minute, second))