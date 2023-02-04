class Die:
    def __init__(self) -> None:
        # DieSize <lowerLeftX> <lowerLeftY> <upperRightX> <upperRightY>
        self.die_size=list()
        self.top_die_max_util=0.0
        self.bottom_die_max_util=0.0
        # TopDieRows <startX> <startY> <rowLength> <rowHeight> <repeatCount>
        # BottomDieRows <startX> <startY> <rowLength> <rowHeight> <repeatCount>
        self.top_die_rows=list()
        self.bottom_die_rows=list()
        self.top_die_tech_name=''
        self.bottom_die_tech_name=''


    def getBottomDieTechName(self)->str:
        return self.bottom_die_tech_name
    
    def setBottomDieTechName(self,bottom_die_tech_name:str)->None:
        self.bottom_die_tech_name=bottom_die_tech_name

    def getTopDieTechName(self)->str:
        return self.top_die_tech_name
    
    def setTopDieTechName(self,top_die_tech_name:str)->None:
        self.top_die_tech_name=top_die_tech_name

    def getBottomDieRows(self)->list:
        return self.bottom_die_rows

    def setBottomDieRows(self,start_x:int,start_y,row_length:int,row_height:int,repeat_count:int)->None:
        self.bottom_die_rows.extend([start_x,start_y,row_length,row_height,repeat_count])

    def getTopDieRows(self)->list:
        return self.top_die_rows

    def setTopDieRows(self,start_x:int,start_y,row_length:int,row_height:int,repeat_count:int)->None:
        self.top_die_rows.extend([start_x,start_y,row_length,row_height,repeat_count])

    def getDieSize(self)->list:
        return self.die_size
    
    def setDieSize(self,lower_left_x:int,lower_left_y:int,upper_right_x:int,upper_right_y:int)->None:
        self.die_size.extend([lower_left_x,lower_left_y,upper_right_x,upper_right_y])
    
    def getTopDieMaxUtil(self)->float:
        return self.top_die_max_util
    
    def setTopDieMaxUtil(self,top_die_max_util:float)->None:
        self.top_die_max_util=top_die_max_util
    
    def getBottomDieMaxUtil(self)->float:
        return self.bottom_die_max_util
    
    def setBottomDieMaxUtil(self,bottom_die_max_util:float)->None:
        self.bottom_die_max_util=bottom_die_max_util
    
