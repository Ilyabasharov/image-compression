import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from IPython.display import clear_output
from collections import defaultdict

class Collector:
    
    def __init__(
        self,
        root_graphics: str,
        root_desc: str,
        phases,
    ) -> None:
        
        os.makedirs(root_graphics, exist_ok=True)
        os.makedirs(root_desc, exist_ok=True)
        
        self.tables = defaultdict(dict)
        
        for phase in phases:
            self.tables[phase]['table'] = PrettyTable()
            self.tables[phase]['path'] = os.path.join(root_desc, '%s.txt' % phase)
        
        self.path_graphics = os.path.join(root_graphics, 'metrics.jpg')
        
        self.i = 0
        
    def step(
        self,
        data: dict,
    ) -> None:
        
        self.i += 1
        clear_output(True)
        
        n_losses = len(list(data.values())[0].keys())
        fig, axs = plt.subplots(ncols=n_losses, figsize=(20, 4))
        
        for phase in data:
            
            if len(self.tables[phase]['table'].field_names) == 0:
                self.tables[phase]['table'].field_names = ['epoch'] + list(data[phase].keys())
                
            self.tables[phase]['table'].add_row(
                [self.i] + [data[phase][loss_type][-1] for loss_type in data[phase]]
            )
            
            for i, loss_type in enumerate(data[phase]):
                axs[i].plot(data[phase][loss_type], label=f'{phase}/{loss_type}')
            
            with open(self.tables[phase]['path'], 'w') as file:
                file.write(self.tables[phase]['table'].get_string())
                
        for i in range(n_losses):
            axs[i].legend()
            axs[i].set_xlabel('epoch')
            axs[i].set_ylabel('loss')
            axs[i].grid()
            
        plt.savefig(self.path_graphics, format='jpg')
        plt.show()