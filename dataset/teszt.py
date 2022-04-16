import torch
import pandas as pd
'''
print(torch.cuda.is_available())

print(torch.cuda.current_device())

print(torch.cuda.get_device_name(0))

print(torch.cuda.memory_allocated())

print(torch.cuda.memory_cached())   #ez elvileg kell, de még nem tudom, h miért
'''


#var1 = torch.FloatTensor([1.0, 2.0,3.0]).cuda()
#print(var1)
#print(var1.device)

df = pd.read_json('dataset/GIT_zizi.json')
