from math import ceil, floor
import bisect

class Region:  
    def __init__(self,lx,ly,hx,hy):
        self.lx = lx
        self.ly = ly
        self.hx = hx
        self.hy = hy
    @property
    def area(self):
        return (self.hx - self.lx) * (self.hy - self.ly)
    @property
    def width(self):
        return self.hx - self.lx
    @property
    def height(self):
        return self.hy - self.ly
    def expand(self,rangeX,rangeY,xl,yl,xh,yh):
        if self.lx - rangeX < xl:
            self.lx = xl
        else:
            self.lx = self.lx - rangeX
        if self.hx + rangeX > xh:
            self.hx = xh
        else:
            self.hx = self.hx + rangeX
        if self.ly - rangeY < yl:
            self.ly = yl
        else:
            self.ly = self.ly - rangeY
        if self.hy + rangeY > yh:
            self.hy = yh
        else:
            self.hy = self.hy + rangeY
        pass

class Cell:
    def __init__(self,x,y,w,h,id):
        self._ox = x
        self._oy = y
        self.xl = 0
        self.yl = 0
        self.xh = 0
        self.yh = 0
        self._tx = 0
        self._ty = 0
        self._x = -1
        self._y = -1
        self._hx = -1
        self._hy = -1
        # index in framework structure
        self.id = -1
        # index in DP structure
        self.i = id
        # in unit of sites
        self.w = w
        self.h = h
        # 0 if no overlap
        self.overlap = 0
        # true if placed
        self.placed = False
        ##########
    @property
    def area(self):
        return self.w*self.h
    @property
    def width(self):
        return self.w
    @property
    def height(self):
        return self.h
    def ox(self):
        return self._ox
    def oy(self):
        return self._oy
    def lx(self):
        return self._x
    def ly(self):
        return self._y
    def hx(self):
        return self._hx
    def hy(self):
        return self._hy
    def setLx(self, x):
        self._x = x
        self._hx = x+self.w
    def setLy(self, y):
        self._y = y
        self._hy = y+self.h
    def setOx(self,x):
        self._ox = x
    def setOy(self,y):
        self._oy = y
    def getTx(self):
        return self._tx
    def getTy(self):
        return self._ty
    def tx(self,x):
        self._tx = x
    def ty(self,y):
        self._ty = y
    def oxLong(self):
        return round(self._ox)
    def oyLong(self):
        return round(self._oy)
    
    def getOrigionRegion(self,rangeX,rangeY,xl,yl,xh,yh):
        region = Region(self.oxLong(), self.oyLong(),self.oxLong() + self.w,self.oyLong() + self.h)
        region.expand(rangeX, rangeY,xl,yl,xh,yh)
        return region

    def setOptimalRegion(self,xl,yl,xh,yh):
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh

class Segment:
    def __init__(self,i,x,y,w):
        self.i = i
        self.x = x
        self.y = y
        self.w = w
        self.cells = list()
        self.id2index_map = dict()
        self.ncells = 0
        pass

    def getCellsAt(self,lx,hx):
        if self.ncells == 0:
            return []
        result = []
        indexs = [cell._x for cell in self.cells]
        s=bisect.bisect_left(indexs,lx)
        e=bisect.bisect_left(indexs,hx)
        for i in range(s,e):
            result.append(self.cells[i])
        '''for cell in self.cells:
            if hx>cell.lx() and cell.hx() >lx:
                result.append(cell)'''
        return result

    def getCellAt(self,x, y):
        if self.ncells == 0:
            return 0
    def getCellsAt_(self,lx,hx):
        if self.ncells == 0:
            return []
        l = 0
        r = self.ncells-1
        while(l < r):
            mid = floor((l+r+1)/2)
            if self.cells[mid]._x < hx:
                l = mid
            else:
                r = mid -1
        if self.cells[l]._x >= hx:
            return []
        else: 
            end = l
            l = 0
            r = self.ncells-1
            while(l < r):
                mid = floor((l+r)/2)
                if self.cells[mid]._x <= lx:
                    l = mid +1 
                else:
                    r = mid
            if self.cells[r]._x > lx:
                return self.cells[r:end+1]
            else:
                return []

    def getCellAt_(self,x):
        if self.ncells == 0:
            return 0
        l = 0
        r = self.ncells-1
        while(l < r):
            mid = floor((l+r)/2)
            if self.cells[mid]._x <= x:
                l = mid +1 
            else:
                r = mid
        print(len(self.cells),r)
        if self.cells[r]._x > x:
            return r
        else:
            return self.ncells    

    def addCell(self,cell):

        indexs = [cell._x for cell in self.cells]
        self.cells.insert(bisect.bisect_left(indexs,cell._x),cell)
        self.ncells+=1
        
        pass

class LocalRegion:
    def __init__(self):
        self.localCells = list()
        self.id2index_map = dict()
        self._ncells = 0
        self.localSegments_h = list()
        self.lx = 0
        self.ly = 0
        self.hx = 0
        self.hy = 0
        pass

    def nCells(self):
        return self._ncells
    
    def addLocalCell(self,cell):
        localCell = LocalCell(cell,self._ncells)
        lx = localCell.lx
        ly = localCell.ly
        hx = localCell.hx
        hy = localCell.hy
        if lx < self.lx:
            lx = self.lx
            localCell.fixed = True
        if ly < self.ly:
            ly = self.ly
            localCell.fixed = True
        if hx > self.hx:
            hx = self.hx
            localCell.fixed = True
        if hy > self.hy:
            hy = self.hy
            localCell.fixed = True
        self.localCells.append(localCell)
        self.id2index_map[localCell.cell.i] = self._ncells
        self._ncells+=1
        return localCell.w*localCell.h
    
    def buildLocalSegments(self):
        line_set = set()
        cell2line = dict()
        line2seg = dict()
        localSegments_h = list()
        line_set.add(self.ly)
        line_set.add(self.hy)
        for localCell in self.localCells:
            line_set.add(localCell.ly)
            line_set.add(localCell.hy)
            cell2line[localCell.i] = localCell.ly
        line_list = sorted(line_set)
        seg_w = self.hx-self.lx
        for i in range(len(line_list)-1):
            seg = LocalSegment(line_l=line_list[i],line_u = line_list[i+1],w = seg_w)
            seg.x = self.lx
            localSegments_h.append(seg)
            line2seg[line_list[i]] = i
        for localCell in self.localCells:
            start = line2seg[cell2line[localCell.i]]
            self.localCells[localCell.i].h_start_r = line2seg[localCell.ly]
            for i in range(start,len(localSegments_h)):
                localSegment = localSegments_h[i]
                if localCell.hy > localSegment.line_l and localCell.ly < localSegment.line_u:
                    localSegment.addLocalCell(localCell.i)
                    self.localCells[localCell.i].addNsegs('h')
                else:
                    break
            self.localCells[localCell.i].h_end_r = self.localCells[localCell.i].h_start_r + self.localCells[localCell.i].h_nsegs-1
        self.localSegments_h = localSegments_h
        self.nsegs_h = len(self.localSegments_h)
        pass

    def placeL(self,pos):
        localSegments = self.localSegments_h
        localCells = self.localCells
        nsegs = self.nsegs_h
        fronts = [0]*nsegs
        ends = [0]*nsegs
        LBounds = [0]*nsegs
        for r in range(nsegs):
            fronts[r] = 0
            ends[r] = localSegments[r].ncells
            LBounds[r] = self.lx
        start_flag = dict()
        for r in range(nsegs):
            for id in localSegments[r].localCells:
                if start_flag.get(id) is None:
                    start_flag[id] = r
        finished = False     
        while not finished:
            finished = True
            for r in range(nsegs):
                while (fronts[r] != ends[r]):
                    finished = False
                    localCell = localCells[localSegments[r].localCells[fronts[r]]]
                    if start_flag[localCell.i] != r: 
                        break
                    blocked = False
                    for s in range(1,localCell.h_nsegs):
                        if localSegments[r + s].localCells[fronts[r + s]] != localCell.i:
                            blocked = True
                            break 
                    if blocked:
                        break
                    elif localCell.fixed:
                        pos[localCell.i] = localCell.lx
                    else:
                        lb = LBounds[r]
                        for s in range(1,localCell.h_nsegs):
                            lb = max(lb, LBounds[r + s])
                        pos[localCell.i] = lb
                    for s in range(localCell.h_nsegs):
                        if (fronts[r + s] + 1 != ends[r + s]):
                            cell = localCells[localSegments[r + s].localCells[fronts[r + s]+ 1]].cell
                            if (cell):
                                if (localCell.fixed):
                                    LBounds[r + s] = pos[localCell.i] + localCell.w
                                else:
                                    LBounds[r + s] = pos[localCell.i] + localCell.w       
                        fronts[r + s]+=1
                    pass
                    
    def placeR(self,pos):
        localSegments = self.localSegments_h
        localCells = self.localCells
        nsegs = self.nsegs_h
        fronts = [0]*nsegs
        ends = [0]*nsegs
        RBounds = [0]*nsegs
        for r in range(nsegs):
            fronts[r] = localSegments[r].ncells - 1
            ends[r] = -1
            RBounds[r] = self.hx
        start_flag = dict()
        for r in range(nsegs):
            for id in localSegments[r].localCells:
                if start_flag.get(id) is None:
                    start_flag[id] = r
        finished = False
        while not finished:
            finished = True
            for r in range(nsegs):
                while (fronts[r] != ends[r]):
                    finished = False
                    localCell = localCells[localSegments[r].localCells[fronts[r]]]
                    if start_flag[localCell.i] != r: 
                        break
                    blocked = False
                    for s in range(1,localCell.h_nsegs):
                        if localSegments[r + s].localCells[fronts[r + s]] != localCell.i:
                            blocked = True
                            break 
                    if blocked:
                        break
                    elif localCell.fixed:
                        pos[localCell.i] = localCell.lx
                    else:
                        rb = RBounds[r] - localCell.w
                        for s in range(1,localCell.h_nsegs):
                            rb = min(rb, RBounds[r + s] - localCell.w)
                        pos[localCell.i] = rb
                    
                    for s in range(localCell.h_nsegs):
                        RBounds[r + s] = pos[localCell.i]
                        if (fronts[r + s]):
                            cell = localCells[localSegments[r + s].localCells[fronts[r + s] - 1]].cell
                            if (cell):
                                if (localCell.fixed):
                                    RBounds[r + s] = pos[localCell.i]
                                else:
                                    RBounds[r + s] = pos[localCell.i]      
                        fronts[r + s]-=1
                    pass            

    def estimateL(self,posL,intervals,insertionPoint,targetX,targetCell):
        nsegs = len(self.localSegments_h)
        for r in range(nsegs):
            if (insertionPoint[r] < 0) :
                continue
            lCell = intervals[r][insertionPoint[r]].L.cell
            if (lCell) :
                lW = lCell.w
                if (posL[lCell.i] < 0 and lCell.lx > targetX - lW):
                    rb = targetX - lW
                    posL[lCell.i] = rb
                    
        finished = False
        while not finished:
            finished = True
            for segment in self.localSegments_h:
                for i in range(segment.ncells-2,-1,-1):
                    if (posL[segment.localCells[i + 1]] < 0):
                        continue
                    cellL = self.localCells[segment.localCells[i]]
                    cellR = self.localCells[segment.localCells[i + 1]]
                    if (cellL.fixed or cellR.fixed):
                        continue
                    lW = cellL.w
                    if posL[cellL.i] < 0:
                        if (cellL.lx > posL[cellR.i] - lW):
                            finished = False
                            rb = posL[cellR.i] - lW
                            posL[cellL.i] = rb
                    else :
                        if (posL[cellL.i] > posL[cellR.i] - lW):
                            finished = False
                            rb = posL[cellR.i] - lW
                            posL[cellL.i] = rb
        pass

    def estimateR(self,posR,intervals,insertionPoint,targetX,targetCell):
        nsegs = len(self.localSegments_h)
        for r in range(nsegs):
            if (insertionPoint[r] < 0) :
                continue
            rCell = intervals[r][insertionPoint[r]].R.cell
            if (rCell is not None) :
                rW = targetCell.w
                if (posR[rCell.i] < 0 and rCell.lx < targetX + rW):
                    lb = targetX + rW
                    posR[rCell.i] = lb
        finished = False
        while (not finished):
            finished = True
            for segment in self.localSegments_h:
                for i in range(1,segment.ncells):
                    if (posR[segment.localCells[i - 1]] < 0):
                        continue
                    cellR = self.localCells[segment.localCells[i]]
                    cellL = self.localCells[segment.localCells[i - 1]]
                    if cellR.fixed or cellL.fixed:
                        continue
                    
                    lW = cellL.w
                    if posR[cellR.i] < 0:
                        if cellR.lx < posR[cellL.i] + lW:
                            finished = False
                            lb = posR[cellL.i] + lW
                            posR[cellR.i] = lb

                    else:
                        if posR[cellR.i] < posR[cellL.i] + lW:
                            finished = False
                            lb = posR[cellL.i] + lW
                            posR[cellR.i] = lb

    def estimateH(self,posL, posR, intervals_h, insertionPoint, targetX, targetCell):
        self.estimateL(posL, intervals_h, insertionPoint, targetX, targetCell)
        self.estimateR(posR, intervals_h, insertionPoint, targetX, targetCell)

    def estimateL_(self,posL,intervals,targetX,targetCell):
        for interval in intervals:
            lCell = interval.L.cell
            if (lCell) :
                lW = lCell.w
                if (posL[lCell.i] < 0 and lCell.lx > targetX - lW):
                    rb = targetX - lW
                    posL[lCell.i] = rb
                    
        finished = False
        while not finished:
            finished = True
            for segment in self.localSegments_h:
                for i in range(segment.ncells-2,-1,-1):
                    if (posL[segment.localCells[i + 1]] < 0):
                        continue
                    cellL = self.localCells[segment.localCells[i]]
                    cellR = self.localCells[segment.localCells[i + 1]]
                    if (cellL.fixed or cellR.fixed):
                        continue
                    lW = cellL.w
                    if posL[cellL.i] < 0:
                        if (cellL.lx > posL[cellR.i] - lW):
                            finished = False
                            rb = posL[cellR.i] - lW
                            posL[cellL.i] = rb
                    else :
                        if (posL[cellL.i] > posL[cellR.i] - lW):
                            finished = False
                            rb = posL[cellR.i] - lW
                            posL[cellL.i] = rb
                    if posL[cellL.i] != -1 and posL[cellL.i] < self.lx:
                        print('error!!')
                        return False
            return True
        pass

    def estimateR_(self,posR,intervals,targetX,targetCell):
        for interval in intervals:
            rCell = interval.R.cell
            if (rCell is not None) :
                rW = targetCell.w
                if (posR[rCell.i] < 0 and rCell.lx < targetX + rW):
                    lb = targetX + rW
                    posR[rCell.i] = lb
        finished = False
        while (not finished):
            finished = True
            for segment in self.localSegments_h:
                for i in range(1,segment.ncells):
                    if (posR[segment.localCells[i - 1]] < 0):
                        continue
                    cellR = self.localCells[segment.localCells[i]]
                    cellL = self.localCells[segment.localCells[i - 1]]
                    if cellR.fixed or cellL.fixed:
                        continue
                    
                    lW = cellL.w
                    if posR[cellR.i] < 0:
                        if cellR.lx < posR[cellL.i] + lW:
                            finished = False
                            lb = posR[cellL.i] + lW
                            posR[cellR.i] = lb

                    else:
                        if posR[cellR.i] < posR[cellL.i] + lW:
                            finished = False
                            lb = posR[cellL.i] + lW
                            posR[cellR.i] = lb
                    if posR[cellR.i] != -1 and posR[cellR.i] + cellR.w > self.hx:
                        print('error!!')
                        return False
            return True

    def estimateH_(self,posL, posR, intervals_h, targetX, targetCell):
        if not self.estimateL_(posL, intervals_h, targetX, targetCell) or not self.estimateR_(posR, intervals_h, targetX, targetCell):
            print('error!!')
            return False
        return True


    def placeSpreadL(self,targetCell, posL):
        finished = False
        nsegs_h = self.nsegs_h
        localSegments = self.localSegments_h
        localCells = self.localCells
        while (not finished):
            finished = True
            for r in range(nsegs_h):
                segment = localSegments[r]
                nCells = segment.ncells
                for i in range(nCells-2,-1,-1):
                    if (segment.localCells[i] == targetCell):
                        continue
                    cellL = localCells[segment.localCells[i]]
                    cellR = localCells[segment.localCells[i + 1]]
                    if (cellL.fixed or cellR.fixed):
                        continue
                    lW = cellL.w
                    if (posL[cellL.i] + lW > posL[cellR.i]):
                        finished = False
                        rb = posL[cellR.i] - lW
                        posL[cellL.i] = rb
                        if not cellL.fixed and posL[cellL.i] < self.lx:
                            print('spreadLerror',posL[cellL.i])
                            return False
        return True
    
    def placeSpreadR(self,targetCell, posR):
        finished = False
        nsegs_h = self.nsegs_h
        localSegments = self.localSegments_h
        localCells = self.localCells
        while (not finished):
            finished = True
            for r in range(nsegs_h):
                segment = localSegments[r]
                nCells = segment.ncells
                for i in range(1,nCells):
                    if (segment.localCells[i] == targetCell):
                        continue
                    cellR = localCells[segment.localCells[i]]
                    cellL = localCells[segment.localCells[i - 1]]
                    if (cellR.fixed or cellL.fixed):
                        continue
                    
                    lW = cellL.w
                    if (posR[cellR.i] < posR[cellL.i] + lW):
                        finished = False
                        lb = posR[cellL.i] + lW
                        posR[cellR.i] = lb
                        if not cellR.fixed and posR[cellR.i] + cellR.w > self.hx:
                            print('spreadRerror',posR[cellR.i])
                            return False
        return True

class LocalSegment:
    def __init__(self,line_l,line_u,w):
        self.localCells = list()
        self.orient = None
        self.pos = 0
        self.line_l = line_l
        self.line_u = line_u
        self.ncells = 0
        self.h = line_u-line_l
        self.x = 0
        self.y = line_l
        self.w = w
        pass
    
    def nCells(self):
        return self.ncells

    def addLocalCell(self,index):
        self.localCells.append(index)
        self.ncells+=1

class LocalCell:
    def __init__(self,cell,i):
        self.cell = cell
        self._ox = cell.ox()
        self._oy = cell.oy()
        self.lx = cell.lx()
        self.ly = cell.ly()
        self.hx = cell.hx()
        self.hy = cell.hy()
        self.fixed = False
        self.w = cell.w
        self.h = cell.h
        self.i = i
        self.h_nsegs = 0
        self.v_nsegs = 0
        self.h_start_r = 0
        self.h_end_r = 0
        pass
    

    def ox(self):
        return self._ox
    def oy(self):
        return self._oy
    def setOx(self,x):
        self._ox = x
    def setOy(self,y):
        self._oy = y

    def setFixed(self,lx,ly,hx,hy):
        if lx!=self.lx or ly!=self.ly or hx!=self.hx or hy!=self.hy:
            self.fixed = True
            self.lx = lx
            self.ly = ly
            self.hx = hx
            self.hy = hy
            self.w = self.hx - self.lx
            self.h = self.hy - self.ly
        pass

    def addNsegs(self,orient):
        if orient=='v':
            self.v_nsegs+=1
        elif orient=='h':
            self.h_nsegs+=1

class LegalMoveIntervalEndpoint:
    def __init__(self,side):
        self.side = side
        self.pos = 0
        self.cell = None
        self.interval = None
        pass

class LegalMoveInterval:
    def __init__(self,r):
        self.pos = -1
        self.i = 0
        self.L = LegalMoveIntervalEndpoint('L')
        self.R = LegalMoveIntervalEndpoint('R')
        self.T = LegalMoveIntervalEndpoint('T')
        self.B = LegalMoveIntervalEndpoint('B')
        self.orient = -1
        self.width = 0
        self._optPointL = 0
        self._optPointR = 0
        self._optPointT = 0
        self._optPointB = 0
        self.r = r
        pass

    def optPointL(self):
        return self._optPointL
    def optPointR(self):
        return self._optPointR
    def optPointT(self):
        return self._optPointT
    def optPointB(self):
        return self._optPointB
    def setOptPointL(self,optPointL):
        self._optPointL = optPointL
    def setOptPointR(self,optPointR):
        self._optPointR = optPointR
    def setOptPointT(self,optPointT):
        self._optPointT = optPointT
    def setOptPointB(self,optPointB):
        self._optPointB = optPointB

class DisplacementSubcurve:
    def __init__(self):
        pass

class DisplacementCurve:
    def __init__(self):
        pass
    
class CriticalPoint:
    def __init__(self,xx,l,u):
        self.x = xx
        self.lSlope = l
        self.rSlope = u
        self.bSlope = l
        self.tSlope = u
        pass 

class CellList:
    def __init__(self):
        pass
    
class Move:
    def __init__(self,cell,tx,ty):
        self.cell = cell
        self.x = tx
        self.y = ty
        self.legals = list()
        self.id2index_map = None
        pass

    def setCellMoveMap(self,id2index_map):
        self.id2index_map = id2index_map

class CellMove:
    def __init__(self,cell,x,y):
        self.cell = cell
        self.x = x
        self.y = y

class Bin:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cells = list()
        self.total_area = w*h
        self.cell_area = 0 
        self.ncells = 0
        self.segments = list()
        pass
    
    def addCell(self,cell):
        self.cells.append(cell)
        self.ncells+=1
        self.cell_area+=cell.w*cell.h
        pass
    
    @property
    def density(self):
        return self.cell_area/(self.total_area+0.000001)
    
    def buildSegments(self,row_height):
        self.row_length = self.w
        self.row_height = row_height
        self.repeat_count = ceil(self.h/self.row_height)
        for i in range(self.repeat_count):
            self.segments.append(Segment(i,self.x,self.y+i*self.row_height,self.row_length))
            
class Bin:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cells = list()
        self.total_area = w*h
        self.cell_area = 0 
        self.ncells = 0
        self.segments = list()
        self._overlap = -1
        self.overlapCells = set()
        self.no_overlapCells = set(self.cells)
        pass
    
    def addCell(self,cell):
        self.cells.append(cell)
        self.ncells+=1
        self.cell_area+=cell.w*cell.h
        pass
    
    @property
    def density(self):
        return self.cell_area/(self.total_area+0.000001)
    

    def overlap(self):
        if self._overlap == -1:
            total_area = 0
            for i in range(self.ncells-1):
                ci = self.cells[i]
                for j in range(i+1,self.ncells):
                    cj = self.cells[j]
                    ol = (max(min(ci.oy()+ci.h, cj.oy()+cj.h)-max(ci.oy(), cj.oy()), 0.0)*max(min(ci.ox()+ci.w, cj.ox()+cj.w)-max(ci.ox(), cj.ox()), 0.0))
                    if ol != 0:
                        self._overlap += ol
                        total_area += ci.area
                        #print(ci.i,cj.i,'ol,tl',ol,ci.area)
            self._overlap = 0 if total_area == 0 else self._overlap/total_area
        return self._overlap

    def buildSegments(self,row_height):
        self.row_length = self.w
        self.row_height = row_height
        self.repeat_count = ceil(self.h/self.row_height)
        for i in range(self.repeat_count):
            self.segments.append(Segment(i,self.x,self.y+i*row_height,self.w))
    
    def reset(self):
        self.no_overlapCells = set(self.cells)
        self.overlapCells = set()
        self._overlap = -1
        pass
    
    def removeCell(self,cell):
        start = floor((cell.ly()-self.y)/self.row_height)
        end = floor((cell.ly()-self.y+cell.h-1)/self.row_height)
        end = min(end,len(self.segments)-1)
        start = max(0,start)
        for i in range(start,end+1):
            self.segments[i].cells.remove(cell)
        pass
    def setInitial(self):
        for cell in self.cells:
            cell.isAbacus = True