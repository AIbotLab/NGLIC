from math import ceil, floor
import random
from typing import Any
import numpy as np
import dplacer_horizon as dp
import Die as Die
import Terminal as Terminal
class Legalization:
    def __init__(self) -> None:
        self.num_terminals = 0
        self.terminals = None
        self.terminals_x = None
        self.terminals_y = None
        self.terminal_size_w = 0
        self.terminal_size_h = 0
        self.terminal_spacing = 0
        self.terminals_weight = None
        self.half_terminal_spacing = 0
        self.xl = 0
        self.xh = 0
        self.yl = 0
        self.yh = 0
        self.num_bins_x = None
        self.num_bins_y = None
        self.bin_size_x = None
        self.bin_size_y = None 
        self.pin_offset_x = None # 1D array, pin offset x to its node
        self.pin_offset_y = None # 1D array, pin offset y to its node
        self.net2pin_map = None
        self.node2pin_map = None
        self.pin2node_map = None
        self.pin2net_map = None
        self.terminal2net_map =None
        self.net_weights = None
        self.dplacer = None
        pass
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    @property
    def width(self):
        """
        @return width of layout
        """
        return self.xh-self.xl

    @property
    def height(self):
        """
        @return height of layout
        """
        return self.yh-self.yl

    @property
    def area(self):
        """
        @return area of layout
        """
        return self.width*self.height


    def read(self,params):
        self.die = params['die']
        die_size = die.getDieSize()
        self.terminals = params['terminals']
        self.terminals_list = self.terminals.getTerminals()
        self.num_terminals = len(self.terminals_list)
        self.xl = die_size[0]
        self.xh = die_size[2]
        self.yl = die_size[1]
        self.yh = die_size[3]
        self.terminals_x = [terminal.getPlaceCoordinate()[0] for terminal in self.terminals_list]
        self.terminals_y = [terminal.getPlaceCoordinate()[1] for terminal in self.terminals_list]
        self.terminal_size_w = self.terminals.getTerminalSize()[0]
        self.terminal_size_h = self.terminals.getTerminalSize()[1]
        self.terminal_spacing = self.terminals.getTerminalSpacing()
        self.dplacer = dp.Dplacer()
        pass

    def scale_by_spacing(self):
        half_terminal_spacing = ceil(self.terminal_spacing/2)
        self.terminal_size_w += self.terminal_spacing
        self.terminal_size_h += self.terminal_spacing
        self.xl+=(self.terminal_spacing-half_terminal_spacing)
        self.xh-=half_terminal_spacing
        self.yl+=(self.terminal_spacing-half_terminal_spacing)
        self.yh-=half_terminal_spacing
        self.terminals_x = [max(terminal_x - half_terminal_spacing,self.xl) for terminal_x in self.terminals_x]
        self.terminals_y = [max(terminal_y - half_terminal_spacing,self.yl) for terminal_y in self.terminals_y]
        self.terminals_x = [min(terminal_x,self.xh - self.terminal_size_w) for terminal_x in self.terminals_x]
        self.terminals_y = [min(terminal_y,self.yh - self.terminal_size_h) for terminal_y in self.terminals_y]
        self.bin_size_x = min(4*self.terminal_size_w,floor(self.width/2))
        self.bin_size_y = min(4*self.terminal_size_h,floor(self.height/2))
          
    def net_hpwl(self, x, y, net_id):
        """
        @brief compute HPWL of a net
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of a net
        """
        
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes]+self.pin_offset_x[pins]) - np.amin(x[nodes]+self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes]+self.pin_offset_y[pins]) - np.amin(y[nodes]+self.pin_offset_y[pins])

        return (hpwl_x+hpwl_y)*self.net_weights[net_id]

    def legalize(self,expandX,expandY,initialX,initialY):
        self.dplacer.read({'die':self.die,'terminals':self.terminals,'expandX':expandX,'expandY':expandY,'initialX':initialX,'initialY':initialY})
        self.dplacer(True)

if __name__ == '__main__':
    params = {}
    die = Die.Die()
    die.setDieSize(0,0,1000,1000)
    terminals = Terminal.Terminals()
    terminals.setTerminalSize(6,6)
    terminals.setTerminalSpacing(11)
    xy = [[random.randint(10,300),random.randint(10,300)] for _ in range(100)]
    l= len(xy)
    for i in range(l):
        terminal = Terminal.Terminal(i)
        terminal.setPlaceCoordinate(xy[i][0],xy[i][1])
        terminals.addTerminal(terminal)
    params['die'] = die
    params['terminals'] = terminals  
    leaglization = Legalization()
    leaglization.read(params=params)
    leaglization.scale_by_spacing()
    leaglization.legalize(expandX=1,expandY=1,initialX=1,initialY=1)
    cost = 0
    max_mv = 0
    for i in range(l):
        xy_ = terminals.getTerminals()[i].getPlaceCoordinate()
        mv = (abs(xy_[0]+6-xy[i][0])+abs(xy_[1]+6-xy[i][1]))
        cost+=mv
        if mv >max_mv:
            max_mv = mv
    print(cost,max_mv)
    pass
