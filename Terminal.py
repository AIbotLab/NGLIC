import re


class Terminal:
    
    def __init__(self,terminal_name:str) -> None:
        self.terminal_name=terminal_name
        self.place_coordinate_x=0
        self.place_coordinate_y=0
        self.net_name=None
        self.offset=[]

    def getNetName(self)->str:
        return self.net_name
    
    def setNetName(self,net_name:str)->None:
        self.net_name=net_name

    def getOffset(self)-> list:
        return self.offset

    def setOffset(self,offset_x:float,offset_y:float)-> None:
        self.offset=[offset_x,offset_y]


    def getTerminalName(self)->str:
        return self.terminal_name
    
    def getPlaceCoordinate(self)->tuple:
        return (self.place_coordinate_x,self.place_coordinate_y)
    
    def setPlaceCoordinate(self,place_coordinate_x:float,place_coordinate_y:float)->None:
        self.place_coordinate_x=place_coordinate_x
        self.place_coordinate_y=place_coordinate_y

class Terminals:
    def __init__(self) -> None:
        self.terminal_size_x=0
        self.terminal_size_y=0
        self.terminal_spacing=0
        self.terminals=list()
        self.termianl_num=0
        self.terminal_name_to_index=dict()
    # 20220527添加
    def getTerminalNum(self)->int:
        return self.termianl_num

    def addTerminal(self,terminal:Terminal)->None:
        result=self.terminal_name_to_index.get(terminal.getTerminalName(),None)
        if result==None:
            self.terminal_name_to_index[terminal.getTerminalName()]=self.termianl_num
            self.terminals.append(terminal)
            self.termianl_num+=1
        else:
            print("存在重复 cells ！",flush=True)
            exit(0)

    def getTerminals(self)->list:
        return self.terminals
    
    def getTerminal(self,terminal_name:str)->Terminal:
        result=self.terminal_name_to_index.get(terminal_name,None)
        if result==None:
            print("terminals 中不存在 "+terminal_name,flush=True)
            exit(0)
        else:
            return self.terminals[result]

    def getTerminalSpacing(self)->int:
        return self.terminal_spacing

    def setTerminalSpacing(self,terminal_spacing:int)->int:
        self.terminal_spacing=terminal_spacing

    def getTerminalSize(self)->tuple:
        return (self.terminal_size_x,self.terminal_size_y)
    
    def setTerminalSize(self,terminal_size_x:int,terminal_size_y:int)->None:
        self.terminal_size_x=terminal_size_x
        self.terminal_size_y=terminal_size_y
