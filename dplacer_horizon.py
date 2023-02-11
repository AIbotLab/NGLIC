'''
Author: error: git config user.name and git config user.email & please set dead value or install git
Date: 2022-06-29 22:13:49
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-09 13:35:23
FilePath: /generateself.bookshelf/dp/dplacer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from collections import defaultdict
import functools
from math import ceil, floor
import pickle
import random
from statistics import mean
import sys, os
import time
sys.path.append('/home/pc/aibot/wjchen/file/ICCAD/GP/back/linkhdfs')
from place3d.dp_copy.prof import do_cprofile
from place3d.Legalization import Legalization
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Die import Die
from Terminal import Terminal, Terminals
import numpy as np
from place3d.dp_copy.dp_data_horizon import Bin, Cell, CellMove, CriticalPoint, LegalMoveInterval, LocalRegion, Move,Segment
from place3d.dp_copy.dplacer import Dplacer as Dplacer_o
from place3d.DataBase import DataBase
import place3d.dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import place3d.dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize
from torch.autograd import Variable
import torch
import heapq


class Dplacer:
    def __init__(self):
        self.cells = list()
        self.id2cell_map = dict()
        self.ncells = 0
        self.segments = list()
        self.localSeg_sum = 0
        self.cnt = 0
        self.h_sum = 0
        self.h_sum_cnt = 0
        self.max_h = 0
        self.max_seg = 0
        self.over_cnt = [0]*7
        self.under_cnt = 0
        self.time1= 0
        self.time2=0
        self.bins = list()
        self.cnt1 = 0
        self.cnt2 = 0
        self.dp_cnt = 0
        self.seg2cells = dict()
        pass
    
    @do_cprofile("./result.out")
    def __call__(self,flag):
        cnt=0
        self.buildSegments()
        while True:
            if self.getUnplaced() or self.overlaped():# or self.overlaped()
                if self.dp_cnt%5==0:
                    self.LocalRegionW += self.terminal_size_w
                    self.LocalRegionH += self.terminal_size_h
                if flag:
                    #self.tetris(self.dp_cnt)
                    self.abacus(self.dp_cnt)
                import time
                start = time.process_time()
                print("开始dplacer")
                self.legalize(flag)
                self.dp_cnt+=1
                print("dplacer 次数:",self.dp_cnt)
                end = time.process_time()
                print('Running time: %s Seconds'%(end-start))
                sum = 0
                for seg in self.segments:
                    sum+=seg.ncells
                print('@@@average',sum/len(self.segments))
            else:
                break
        pass

    def reset(self):
        #self.placed_cell.clear()
        for i in range(self.ncells):
            self.cells[i].placed = False
        pass

    def result(self):
        result = []
        for cell in self.cells:
            result.append((cell.lx()+self.half_terminal_spacing,cell.ly()+self.half_terminal_spacing))
        return result

    def addCell(self,x,y,w,h):
        dpCell = Cell(x,y,w,h,self.ncells)
        self.cells.append(dpCell)
        self.id2cell_map[self.ncells] = dpCell
        self.ncells+=1
        return dpCell
    
    def removeCell(self,cell):
        start = floor((cell.ly()-self.yl)/self.row_height)
        end = floor((cell.ly()-self.yl+cell.h-1)/self.row_height)
        for i in range(start,end+1):
            self.segments[i].cells.remove(cell)

    def overlaped(self):
        segs = self.segments
        cells = self.cells
        for cell in cells:
            if cell.ly() <self.yl or cell.hy()>self.yh or cell.lx()<self.xl or cell.hx()>self.xh:
                print('outside:',cell.ly(),self.yl,cell.hy(),self.yh,cell.lx(),self.xl,cell.hx(),self.xh)
                return True
            neibor_cell_set = set()
            start = floor((cell.ly()-self.yl)/self.row_height)
            end = floor((cell.ly()-self.yl + cell.h -1)/self.row_height)
            start = max(0,start-1)
            end = min(self.repeat_count-1,end+1)
            for i in range(start,end+1):
                #for c in segs[i].getCellsAt(cell.lx(),cell.lx()+cell.w):
                result,s,e = segs[i].getCellsAt(cell.lx()-cell.w+1,cell.hx())
                for c in result:
                    if c.i not in neibor_cell_set:
                        neibor_cell_set.add(c.i)
                    else:continue
                    #if cell.i != c.i and max(min(c.ly()+c.h, cell.ly()+cell.h)-max(c.ly(), cell.ly()), 0.0) != 0:
                    if cell.i != c.i and c.hy()>cell.ly() and cell.hy()>c.ly():
                        print('ol',cell.i,c.i)
                        print(cell.lx(),cell.ly(),c.lx(),c.ly())
                        print(cell.hx(),cell.hy(),c.hx(),c.hy())
                        return True
        return False

    def getUnplaced(self): 
        for cell in self.cells:
            if not cell.placed:
                return True
        print('time:',self.dp_cnt)
        return False
    
    def read(self,params): 
        die = params['die']
        die_size = die.getDieSize()
        self.terminals = params['terminals']
        self.terminals_list = self.terminals.getTerminals()
        self.num_terminals = len(self.terminals_list)
        self.xl = die_size[0]
        self.xh = die_size[2]
        self.yl = die_size[1]
        self.yh = die_size[3]
        self.terminal_size_w = self.terminals.getTerminalSize()[0]
        self.terminal_size_h = self.terminals.getTerminalSize()[1]
        self.terminal_spacing = self.terminals.getTerminalSpacing()
        self.half_terminal_spacing = ceil(self.terminal_spacing/2)
        self.terminal_size_w += self.terminal_spacing
        self.terminal_size_h += self.terminal_spacing
        self.xl+=(self.terminal_spacing-self.half_terminal_spacing)
        self.xh-=self.half_terminal_spacing
        self.yl+=(self.terminal_spacing-self.half_terminal_spacing)
        self.yh-=self.half_terminal_spacing
        self.row_length = self.xh-self.xl
        self.row_height = self.terminal_size_h
        self.repeat_count = ceil((self.yh-self.yl)/self.row_height)
        self.ori_x = [terminal.place_coordinate_x - self.half_terminal_spacing for terminal in self.terminals_list]
        self.ori_y = [terminal.place_coordinate_y - self.half_terminal_spacing for terminal in self.terminals_list]
        self.ori_x_v = [self.yh-y-self.terminal_size_h for y in self.ori_y]
        self.ori_y_v = [x for x in self.ori_x]
        self.terminal_x = [min(self.xh - self.terminal_size_w,max(terminal.place_coordinate_x - self.half_terminal_spacing,self.xl)) for terminal in self.terminals_list]
        self.terminal_y = [min(self.yh - self.terminal_size_h,max(terminal.place_coordinate_y - self.half_terminal_spacing,self.yl)) for terminal in self.terminals_list]
        for x,y in zip(self.terminal_x,self.terminal_y):
            self.addCell(x,y,self.terminal_size_w,self.terminal_size_h)
        self.LocalRegionW = int(params['initialX']*self.terminal_size_w)
        self.LocalRegionH = int(params['initialY']*self.terminal_size_h)
        self.expandW = self.terminal_size_w*params['expandX']
        self.expandH = self.terminal_size_h*params['expandY']
        print(self.expandW,self.expandH)
        self.num_bin_x = 32
        self.num_bin_y = 32

    def buildSegments(self):
        self.segments = None
        self.segments = list()
        print('repeat_count',self.repeat_count,self.row_height)
        for i in range(self.repeat_count):
            self.segments.append(Segment(i,self.xl,self.yl+i*self.row_height,self.row_length))

    def buildBins(self):
        coreW = (self.xh - self.xl)
        coreH = (self.yh - self.yl)
        W = min(coreW,max(floor(coreW/self.num_bin_x),4*self.terminal_size_w))
        H = min(coreH,max(floor(coreH/self.num_bin_y),4*self.terminal_size_h))
        self.num_bin_x = ceil(coreW/W)
        self.num_bin_y = ceil(coreH/H)
        
        print(W,H,self.num_bin_x,self.num_bin_y)
        for y in range(self.num_bin_y):
            bins = list()
            for x in range(self.num_bin_x):
                bins.append(Bin(self.xl+x*W,self.yl+y*H,W,H))
            self.bins.append(bins)
        for bin in self.bins[-1]:
            bin.h = self.yh-bin.y
        for bins in self.bins:
            bins[-1].w = self.xh-bin.x
        for cell in self.cells:
            xi = floor((cell.ox()+cell.w/2-self.xl)/W)
            yi = floor((cell.oy()+cell.h/2-self.yl)/H)
            self.bins[yi][xi].addCell(cell)
        for bins in self.bins:
            for bin in bins:
                #deal with overflowed
                print('d:',bin.density,'o:',bin.overlap())
                if bin.overlap() > 0.25:
                    bin.setInitial()
                elif bin.density > 0.7:
                    bin.setInitial()
        
    def getOptimalX(self,critiPtsInput,rangeBoundL,rangeBoundR):
        critiPts_num = len(critiPtsInput)
        critiPts = [None for _ in range(critiPts_num)]
        for i in range(critiPts_num):
            critiPts[i] = critiPtsInput[i]
        critiPts.sort(key=lambda p:p.x)
        filled = 0
        for i in range(1,critiPts_num):
            if (critiPts[i].x == critiPts[filled].x):
                critiPts[filled].lSlope += critiPts[i].lSlope
                critiPts[filled].rSlope += critiPts[i].rSlope
            else :
                if (critiPts[filled].lSlope or critiPts[filled].rSlope):
                    filled+=1
                critiPts[filled] = critiPts[i]
        #critiPts.resize(filled + 1)
        critiPts = critiPts[:filled+2]
        critiPts_num = len(critiPts)
        slopes = [0 for _ in range(critiPts_num+1)]
        for i in range(critiPts_num):
            slopes[i + 1] = slopes[i] + critiPts[i].rSlope
        acc = 0
        for i in range(critiPts_num,0,-1):
            acc += critiPts[i - 1].lSlope
            slopes[i - 1] += acc
        values = [None for _ in range(critiPts_num)]
        values[0] = 0
        iMin = 0
        minV = 0
        for i in range(1,critiPts_num):
            values[i] = values[i - 1] + slopes[i] * (critiPts[i].x - critiPts[i - 1].x)
            if (values[i] < minV):
                minV = values[i]
                iMin = i
        iL = iMin
        iR = iMin
        placeXL = critiPts[iL].x
        placeXR = critiPts[iR].x
        if (iMin < critiPts_num - 1 and values[iMin + 1] == values[iMin]):
            iL = iMin + 1
            placeXL = (critiPts[iMin].x + critiPts[iMin + 1].x) / 2
            placeXR = (critiPts[iMin].x + critiPts[iMin + 1].x) / 2 + 1
        placeXC = round(placeXL)
        if (placeXR < rangeBoundL):
            return rangeBoundL
        elif (placeXL > rangeBoundR):
            return rangeBoundR
        else: return placeXC


    
    def legalize(self,flag): 
        def shiftCell(cell,targetX,targetY):
            #cell.setLx(targetX)
            #cell.setLy(targetY)
            self.removeCell(cell)
            moveCell(cell,targetX,targetY)
            pass

        def moveCell(cell,targetX,targetY):
            cell.setLx(targetX)
            cell.setLy(targetY)
            cell.placed = True
            start = floor((cell.ly()-self.yl)/self.row_height)
            end = floor((cell.ly()-self.yl+cell.h-1)/self.row_height)
            cell.seg.clear()
            for i in range(start,end+1):
                self.segments[i].addCell(cell)
            pass

        def isLegalMoveMLL(move, region):
            def comp_up(a, b):
                if a.pos != b.pos:
                    return a.pos - b.pos
                if a.side != b.side:
                    return -1 if a.side == 'L' else 1
                return a.interval.r - b.interval.r
            lx = region.lx
            ly = region.ly
            hx = region.hx
            hy = region.hy
            targetCell = move.cell
            targetX = move.x
            targetY = move.y
            targetCellH = targetCell.h
            targetCellW = targetCell.w
            localRegion = LocalRegion()
            density = defineLocalRegion(targetCell,localRegion,region)
            if density:
                return False
            localSegments_h = localRegion.localSegments_h
            localCells = localRegion.localCells
            nsegs_h = localRegion.nsegs_h
            begin_row = -1
            end_row = -1
            for r in range(nsegs_h):
                if localSegments_h[r].y + localSegments_h[r].h >= targetY:
                    begin_row = r
                    break
            for r in range(begin_row,nsegs_h):
                if localSegments_h[r].y + localSegments_h[r].h >= targetY + targetCellH:
                    end_row = r
                    break
            if begin_row == -1 or end_row == -1:return False
            sum_seg_h = {-1:0}
            cur_h = 0
            for cur_r in range(nsegs_h):
                cur_h += localSegments_h[cur_r].h
                sum_seg_h[cur_r] = cur_h
            insert_points = defaultdict(lambda:[[],[]])
            for curseg in range(begin_row,end_row+1):
                insert_points[curseg][0].append(begin_row)
                insert_points[curseg][1].append(end_row)
            begin = begin_row - 1
            end = end_row - 1
            while begin >= -1:
                while sum_seg_h[end] - sum_seg_h[begin] >= targetCellH:
                        for curseg in range(begin+1,end+1):
                            insert_points[curseg][0].append(begin+1)
                            insert_points[curseg][1].append(end)
                        end-=1
                begin-=1
            begin = begin_row
            end = end_row
            while end < nsegs_h:
                while sum_seg_h[end] - sum_seg_h[begin] >= targetCellH:
                        for curseg in range(begin+1,end+1):
                            insert_points[curseg][0].append(begin+1)
                            insert_points[curseg][1].append(end)
                        begin+=1
                end+=1
            '''up_btmRow_map = dict()
            up_topRow_map = dict()
            low_btmRow_map = dict()
            low_topRow_map = dict()
            begin = begin_row - 1
            end = end_row - 1
            while begin >= -1:
                while sum_seg_h[end] - sum_seg_h[begin] >= targetCellH:
                        low_btmRow_map[end] = begin+1
                        low_topRow_map[end] = end
                        end-=1
                begin-=1
            begin = begin_row
            end = end_row
            while end < nsegs_h:
                while sum_seg_h[end] - sum_seg_h[begin] >= targetCellH:
                        up_btmRow_map[begin+1] = begin+1
                        up_topRow_map[begin+1] = end
                        begin+=1
                end+=1
            insert_points = dict()
            for row in range(nsegs_h):
                btmRows = []
                topRows = []
                if low_btmRow_map.get(row) is not None:
                    btmRow = low_btmRow_map[row]
                    topRow = low_topRow_map[row]
                    while True:
                        btmRows.append(btmRow)
                        topRows.append(topRow)
                        if low_btmRow_map.get(topRow+1) is not None and low_btmRow_map[topRow+1] <= row and low_topRow_map[topRow+1] >= row:
                            btmRow = low_btmRow_map[topRow+1]
                            topRow = low_topRow_map[topRow+1]
                        else:break
                if row >= begin_row and end_row >= row:
                    btmRows.append(begin_row)
                    topRows.append(end_row)
                if up_btmRow_map.get(row) is not None:
                    btmRow = up_btmRow_map[row]
                    topRow = up_topRow_map[row]
                    while True:
                        btmRows.append(btmRow)
                        topRows.append(topRow)
                        if up_btmRow_map.get(btmRow-1) is not None and up_btmRow_map[btmRow-1]<= row and up_topRow_map[btmRow-1] >= row:
                            topRow = up_topRow_map[btmRow-1]
                            btmRow = up_btmRow_map[btmRow-1] 
                        else:break
                insert_points[row]=[btmRows,topRows]
            for r in range(nsegs_h):
                if len(insert_points[r][0]) !=len(insert_points1[r][0]):
                    print(r,insert_points[r],insert_points1[r])
                    
                pass
            #print('a:',insert_points,'b:',insert_points1)'''
            intervals_h = [[LegalMoveInterval(r) for _ in range(localSegments_h[r].nCells()+1)] for r in range(nsegs_h)]
            pos = [[localCell.lx,localCell.ly] for localCell in localRegion.localCells]
            pos_h = [i[0] for i in pos]
            localRegion.placeL(pos_h)
            for r in range(nsegs_h):
                segment = localSegments_h[r]
                intervals_h[r][0].L.pos = segment.x
                intervals_h[r][0].L.cell = None
                for i in range(segment.nCells()):
                    localCell = localRegion.localCells[segment.localCells[i]]
                    lb = pos_h[localCell.i] + localCell.w
                    intervals_h[r][i + 1].L.pos = lb
                    intervals_h[r][i + 1].L.cell = localCell
            localRegion.placeR(pos_h)
            for r in range(nsegs_h):
                segment = localSegments_h[r]
                for i in range(segment.nCells()):
                    localCell = localRegion.localCells[segment.localCells[i]]
                    rb = pos_h[localCell.i] - targetCell.width
                    intervals_h[r][i].R.pos = rb
                    intervals_h[r][i].R.cell = localCell
                intervals_h[r][segment.nCells()].R.pos = segment.x + segment.w - targetCell.width
                intervals_h[r][segment.nCells()].R.cell = None
            for r in range(nsegs_h):
                nIntervals = len(intervals_h[r])
                for i in range(nIntervals-1,-1,-1):
                    if intervals_h[r][i].L.pos > intervals_h[r][i].R.pos:
                        intervals_h[r].remove(intervals_h[r][i])
                segment = localSegments_h[r]
                nIntervals = len(intervals_h[r])
                for i in range(nIntervals):
                    intervals_h[r][i].i = i
                    lCell = intervals_h[r][i].L.cell
                    rCell = intervals_h[r][i].R.cell
                    if lCell:
                        intervals_h[r][i].setOptPointL(lCell.ox()+lCell.w)
                    else:
                        intervals_h[r][i].setOptPointL(segment.x)
                    if rCell:
                        intervals_h[r][i].setOptPointR(rCell.ox() - targetCellW)
                    else:
                        intervals_h[r][i].setOptPointR(segment.x + segment.w - targetCell.w)
                    intervals_h[r][i].L.interval = intervals_h[r][i]
                    intervals_h[r][i].R.interval = intervals_h[r][i]
            intervalSort = []
            for r in range(nsegs_h):
                for i in range(len(intervals_h[r])):
                    intervalSort.append(intervals_h[r][i].L)
                    intervalSort.append(intervals_h[r][i].R)
            intervalSort.sort(key=functools.cmp_to_key(comp_up))
            inQueueBgn = [0 for _ in range(nsegs_h)]
            inQueueEnd = [0 for _ in range(nsegs_h)]
            validBegin = [[0 for _ in range(nsegs_h)] for _ in range(nsegs_h)]
            posL = [-1 for _ in range(localRegion.nCells()+1)]
            posR = [-1 for _ in range(localRegion.nCells()+1)]
            bestX = sys.maxsize
            bestY = sys.maxsize
            bestCost = sys.maxsize
            bestIntervals = None
            cost_map = dict()
            
            for endpoint in intervalSort:
                row = endpoint.interval.r
                if endpoint.side == 'R':
                    for r in range(nsegs_h):
                        if validBegin[r][row] == inQueueBgn[row]:
                            validBegin[r][row]+=1
                    inQueueBgn[row]+=1
                    continue

                cell = endpoint.interval.L.cell
                if (cell and cell.h_nsegs > 1):
                    for r in range(cell.h_start_r,cell.h_start_r + cell.h_nsegs): 
                        if (r == row) :
                            continue
                        if (inQueueEnd[r] > 0 and intervals_h[r][inQueueEnd[r] - 1].L.cell == cell) :
                            validBegin[row][r] = inQueueEnd[r] - 1
                            continue
                        validBegin[row][r] = inQueueEnd[r]
                    
                inQueueEnd[row]+=1
                if insert_points.get(row) is not None:
                    btmRows = insert_points[row][0]
                    topRows = insert_points[row][1]
                else: continue
                for btmRow,topRow in zip(btmRows,topRows):
                    indexes = [-1 for _ in range(nsegs_h)]
                    insert_btm_y = localSegments_h[btmRow].y
                    insert_top_y = localSegments_h[topRow].y+localSegments_h[topRow].h
                    if insert_top_y <= targetY+targetCellH:
                        placeY = insert_top_y - targetCellH
                    elif insert_btm_y >= targetY:
                        placeY = insert_btm_y
                    else:
                        placeY = targetY
                    cost_y = abs(placeY - targetY)
                    nTotal = 1
                    r = btmRow
                    while r<=topRow and nTotal==1:
                        if (r == row) :
                            indexes[r] = inQueueEnd[r] - 1
                        else :
                            indexes[r] = validBegin[row][r]
                            nTotal *= (inQueueEnd[r] - validBegin[row][r])
                        r+=1
                    if (nTotal!=1):
                        continue
                    index = 0
                    while index != nTotal:
                        for r in range(btmRow,topRow):
                            if (indexes[r] >= inQueueEnd[r]) :
                                if (r == row) :
                                    indexes[r] = inQueueEnd[r] - 1
                                else :
                                    indexes[r] = validBegin[row][r]

                                indexes[r + 1]+=1
                            else:
                                break
                        key_str = ''
                        temp_intervals = []
                        llist = list()
                        rlist = list()
                        lmap = defaultdict(lambda : 0)
                        rmap = defaultdict(lambda : 0)
                        temp_intervals.append(intervals_h[btmRow][indexes[btmRow]])
                        lCell = intervals_h[btmRow][indexes[btmRow]].L.cell
                        rCell = intervals_h[btmRow][indexes[btmRow]].R.cell
                        if lCell is not None:
                            lmap[lCell.i] += btmRow-lCell.h_start_r+1
                        if rCell is not None:
                            rmap[rCell.i] += btmRow-rCell.h_start_r+1
                        for r in range(btmRow+1,topRow):
                            temp_intervals.append(intervals_h[r][indexes[r]])
                            lCell = intervals_h[r][indexes[r]].L.cell
                            rCell = intervals_h[r][indexes[r]].R.cell
                            if lCell is not None:
                                lmap[lCell.i] += 1
                            if rCell is not None:
                                rmap[rCell.i] += 1
                        if btmRow != topRow:
                            temp_intervals.append(intervals_h[topRow][indexes[topRow]])
                            lCell = intervals_h[topRow][indexes[topRow]].L.cell
                            rCell = intervals_h[topRow][indexes[topRow]].R.cell
                            if lCell is not None:
                                lmap[lCell.i] += lCell.h_end_r - topRow+1
                            if rCell is not None:
                                rmap[rCell.i] += rCell.h_end_r - topRow+1
                        for k,v in lmap.items():
                            if localCells[k].h_nsegs == v:
                                llist.append(k)
                        for k,v in rmap.items():
                            if localCells[k].h_nsegs == v:
                                rlist.append(k)
                        key_str = '.'.join([str(ks) for ks in llist])+'-'+'.'.join([str(ks) for ks in rlist])
                        self.cnt1+=1
                        if cost_map.get(key_str) is not None:
                            _,best_cost_y,_ = cost_map[key_str]
                            if best_cost_y > cost_y:
                                cost_map[key_str] = (placeY,cost_y,temp_intervals)
                        else:cost_map[key_str] = (placeY,cost_y,temp_intervals)
                        index+=1
                        indexes[btmRow]+=1  
            self.cnt2+=len(cost_map)
            for placeY,cost_y,intervals in cost_map.values():
                    rangeBoundL = lx
                    rangeBoundR = hx
                    for intervalY in intervals:
                        rangeBoundL = max(intervalY.L.pos, rangeBoundL)
                        rangeBoundR = min(intervalY.R.pos, rangeBoundR)
                    if (rangeBoundL > rangeBoundR):
                        continue
                    woDistL = -sys.maxsize - 1
                    woDistR = sys.maxsize
                    for intervalY in intervals:
                        woDistL = max(intervalY.optPointL() + 0.5, woDistL)
                        woDistR = min(intervalY.optPointR() + 0.5, woDistR)
                    
                    woDistL = max(woDistL, targetX)
                    woDistR = min(woDistR, targetX)
                    woDistL = min(woDistL, rangeBoundR)
                    woDistR = max(woDistR, rangeBoundL)
                    placeX = targetX
                    if (woDistL != targetX or targetX != woDistR) :
                        critiPts = []
                        posL = [-1 for _ in range(localRegion.nCells())]
                        if not localRegion.estimateL_(posL, intervals, woDistR, targetCell):
                            return False
                        for i in range(localRegion.nCells()):
                            if (posL[i] >= 0) :
                                shift = woDistR - posL[i]
                                curX = localRegion.localCells[i].lx + shift
                                gpX = localRegion.localCells[i].cell.ox() + shift
                                if (curX <= gpX) :
                                    if (curX >= rangeBoundL) :
                                        critiPts.append(CriticalPoint(curX, -1, 0))
                                else :
                                    if (gpX >= rangeBoundL) :
                                        critiPts.append(CriticalPoint(gpX, -2, 0))
                                    if (curX >= rangeBoundL) :
                                        critiPts.append(CriticalPoint(curX, 1, 0))
                        posR = [-1 for _ in range(localRegion.nCells())]
                        if not localRegion.estimateR_(posR, intervals, woDistL, targetCell):
                            return False
                        for i in range(localRegion.nCells()):
                            if posR[i] >= 0:
                                shift = woDistL - posR[i]
                                curX = localRegion.localCells[i].lx + shift
                                gpX = localRegion.localCells[i].cell.ox() + shift
                                if curX >= gpX:
                                    if curX <= rangeBoundR:
                                        critiPts.append(CriticalPoint(curX, 0, 1))
                                else :
                                    if curX <= rangeBoundR:
                                        critiPts.append(CriticalPoint(curX, 0, -1))
                                    if gpX <= rangeBoundR:
                                        critiPts.append(CriticalPoint(gpX, 0, 2))
                        critiPts.append(CriticalPoint(targetX, -1, 1))
                        placeX = self.getOptimalX(critiPts,rangeBoundL, rangeBoundR)
                    placeX = max(rangeBoundL, min(rangeBoundR, placeX))
                    targetCostX = 0
                    targetCostY = 0
                    targetCostX = abs(targetX - placeX)
                    targetCostY = cost_y
                    localCostX = 0
                    localCostY = 0
                    posL = [-1 for _ in range(localRegion.nCells())]
                    posR = [-1 for _ in range(localRegion.nCells())]
                    if not localRegion.estimateH_(posL, posR, intervals, placeX, targetCell):
                        return False
                    for i in range(localRegion.nCells()):
                        cell = localRegion.localCells[i]
                        if (posL[i] >= 0):
                            localCostX += (abs(posL[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                            
                        if (posR[i] >= 0):
                            localCostX += (abs(posR[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                    costx = targetCostX + localCostX
                    costy = targetCostY + localCostY
                    cost = costx + costy
                    if (cost < bestCost):
                        bestX = placeX
                        bestY = placeY
                        bestCost = cost
                        bestIntervals = intervals
            if (bestCost == sys.maxsize):
                return False
            curTargetX = targetCell.lx()
            curTargetY = targetCell.ly()
            
            if bestX<self.xl or bestX+targetCellW>self.xh or bestY<self.yl or bestY+targetCellH>self.yh:
                return False
            
            targetCell.setLx(bestX)
            targetCell.setLy(bestY)
            localRegion.addLocalCell(targetCell)
            newCellIndex = localRegion.nCells()-1
            for interval in bestIntervals:
                rCell = interval.R.cell
                r = interval.r
                if (rCell is None) :
                    localSeg = localSegments_h[r]
                    localSeg.addLocalCell(newCellIndex)
                else:
                    cellRIndex = rCell.i
                    localSeg = localSegments_h[r]
                    localSeg.addLocalCell(newCellIndex)
                    for i in range(localSeg.nCells() - 2,-1,-1):
                        localSeg.localCells[i + 1] = localSeg.localCells[i]
                        if (localSeg.localCells[i] == cellRIndex):
                            localSeg.localCells[i] = newCellIndex
                            break
            pos_h = [localRegion.localCells[i].lx for i in range(localRegion.nCells())]
            if not localRegion.placeSpreadL(newCellIndex, pos_h):
                return False
            if not localRegion.placeSpreadR(newCellIndex, pos_h): 
                return False

            move.x = bestX
            move.y = bestY
            nCells = localRegion.nCells() - 1
            
            for i in range(nCells):
                cell = localRegion.localCells[i]
                if (cell.fixed):
                    move.no_moves.append(cell.cell)
                    continue
                if (cell.lx != cell.cell.lx() or cell.lx != pos_h[cell.i]):
                    move.legals.append((CellMove(cell.cell, pos_h[cell.i], cell.ly)))
                else:
                    move.no_moves.append(cell.cell)
            self.localSeg_sum+=nsegs_h
            self.cnt+=1
            return True
            
        def isLegalMoveMLL_(move, region):
            def comp_up(a, b):
                if a.pos != b.pos:
                    return a.pos - b.pos
                if a.side != b.side:
                    return -1 if a.side == 'L' else 1
                return a.interval.r - b.interval.r
            def comp_low(a, b):
                if a.pos != b.pos:
                    return a.pos - b.pos
                if a.side != b.side:
                    return -1 if a.side == 'L' else 1
                return b.interval.r - a.interval.r
            lx = region.lx
            ly = region.ly
            hx = region.hx
            hy = region.hy
            targetCell = move.cell
            targetX = move.x
            targetY = move.y
            targetCellH = targetCell.h
            targetCellW = targetCell.w
            localRegion = LocalRegion()
            density = defineLocalRegion(targetCell,localRegion,region)
            if density:
                return False
            localSegments_h = localRegion.localSegments_h
            nsegs_h = localRegion.nsegs_h
            begin_row = -1
            end_row = -1
            for r in range(nsegs_h):
                if localSegments_h[r].y + localSegments_h[r].h >= targetY:
                    begin_row = r
                    break
            for r in range(begin_row,nsegs_h):
                if localSegments_h[r].y + localSegments_h[r].h >= targetY + targetCellH:
                    end_row = r
                    break
            if begin_row == -1 or end_row == -1:return False
            up_btmRow_map = dict()
            up_topRow_map = dict()
            low_btmRow_map = dict()
            low_topRow_map = dict()
            sum_seg_h = {-1:0}
            cur_h = 0
            for cur_r in range(nsegs_h):
                cur_h += localSegments_h[cur_r].h
                sum_seg_h[cur_r] = cur_h
            begin = begin_row - 1
            end = end_row - 1
            while begin >= -1:
                while sum_seg_h[end] - sum_seg_h[begin] >= targetCellH:
                        low_btmRow_map[end] = begin+1
                        low_topRow_map[end] = end
                        end-=1
                begin-=1
            for r in range(end+1):
                low_btmRow_map[r] = 0
                low_topRow_map[r] = end+1
            begin = begin_row
            end = end_row
            while end < nsegs_h:
                while sum_seg_h[end] - sum_seg_h[begin] >= targetCellH:
                        up_btmRow_map[begin+1] = begin+1
                        up_topRow_map[begin+1] = end
                        begin+=1
                end+=1
            insert_points = dict()
            for row in range(nsegs_h):
                btmRows = []
                topRows = []
                if low_btmRow_map.get(row) is not None:
                    btmRow = low_btmRow_map[row]
                    topRow = low_topRow_map[row]
                    while True:
                        btmRows.append(btmRow)
                        topRows.append(topRow)
                        if low_btmRow_map.get(topRow+1) is not None and low_btmRow_map[topRow+1] <= row and low_topRow_map[topRow+1] >= row:
                            btmRow = low_btmRow_map[topRow+1]
                            topRow = low_topRow_map[topRow+1]
                        else:break
                if row >= begin_row and end_row >= row:
                    btmRows.append(begin_row)
                    topRows.append(end_row)
                if up_btmRow_map.get(row) is not None:
                    btmRow = up_btmRow_map[row]
                    topRow = up_topRow_map[row]
                    while True:
                        btmRows.append(btmRow)
                        topRows.append(topRow)
                        if up_btmRow_map.get(btmRow-1) is not None and up_btmRow_map[btmRow-1]<= row and up_topRow_map[btmRow-1] >= row:
                            topRow = up_topRow_map[btmRow-1]
                            btmRow = up_btmRow_map[btmRow-1] 
                        else:break
                insert_points[row]=[btmRows,topRows]
            intervals_h = [[LegalMoveInterval(r) for _ in range(localSegments_h[r].nCells()+1)] for r in range(nsegs_h)]
            pos = [[localCell.lx,localCell.ly] for localCell in localRegion.localCells]
            pos_h = [i[0] for i in pos]
            localRegion.placeL(pos_h)
            for r in range(nsegs_h):
                segment = localSegments_h[r]
                intervals_h[r][0].L.pos = segment.x
                intervals_h[r][0].L.cell = None
                for i in range(segment.nCells()):
                    localCell = localRegion.localCells[segment.localCells[i]]
                    lb = pos_h[localCell.i] + localCell.w
                    intervals_h[r][i + 1].L.pos = lb
                    intervals_h[r][i + 1].L.cell = localCell
            localRegion.placeR(pos_h)
            for r in range(nsegs_h):
                segment = localSegments_h[r]
                for i in range(segment.nCells()):
                    localCell = localRegion.localCells[segment.localCells[i]]
                    rb = pos_h[localCell.i] - targetCell.width
                    intervals_h[r][i].R.pos = rb
                    intervals_h[r][i].R.cell = localCell
                intervals_h[r][segment.nCells()].R.pos = segment.x + segment.w - targetCell.width
                intervals_h[r][segment.nCells()].R.cell = None
            for r in range(nsegs_h):
                nIntervals = len(intervals_h[r])
                for i in range(nIntervals-1,-1,-1):
                    if intervals_h[r][i].L.pos > intervals_h[r][i].R.pos:
                        intervals_h[r].remove(intervals_h[r][i])
                segment = localSegments_h[r]
                nIntervals = len(intervals_h[r])
                for i in range(nIntervals):
                    intervals_h[r][i].i = i
                    lCell = intervals_h[r][i].L.cell
                    rCell = intervals_h[r][i].R.cell
                    if lCell:
                        intervals_h[r][i].setOptPointL(lCell.ox()+lCell.w)
                    else:
                        intervals_h[r][i].setOptPointL(segment.x)
                    if rCell:
                        intervals_h[r][i].setOptPointR(rCell.ox() - targetCellW)
                    else:
                        intervals_h[r][i].setOptPointR(segment.x + segment.w - targetCell.w)
                    intervals_h[r][i].L.interval = intervals_h[r][i]
                    intervals_h[r][i].R.interval = intervals_h[r][i]
            intervalSort = []
            for r in range(nsegs_h):
                for i in range(len(intervals_h[r])):
                    intervalSort.append(intervals_h[r][i].L)
                    intervalSort.append(intervals_h[r][i].R)
            intervalSort.sort(key=functools.cmp_to_key(comp_up))
            inQueueBgn = [0 for _ in range(nsegs_h)]
            inQueueEnd = [0 for _ in range(nsegs_h)]
            validBegin = [[0 for _ in range(nsegs_h)] for _ in range(nsegs_h)]
            nlocalcells = localRegion.nCells()
            posL = [-1 for _ in range(nlocalcells+1)]
            posR = [-1 for _ in range(nlocalcells+1)]
            bestX = sys.maxsize
            bestY = sys.maxsize
            bestCost = sys.maxsize
            cost_map = dict()
            bestIntervals = [0 for _ in range(nsegs_h)]
            for endpoint in intervalSort:
                row = endpoint.interval.r
                if endpoint.side == 'R':
                    for r in range(nsegs_h):
                        if validBegin[r][row] == inQueueBgn[row]:
                            validBegin[r][row]+=1
                    inQueueBgn[row]+=1
                    continue

                cell = endpoint.interval.L.cell
                if (cell and cell.h_nsegs > 1):
                    for r in range(cell.h_start_r,cell.h_start_r + cell.h_nsegs): 
                        if (r == row) :
                            continue
                        if (inQueueEnd[r] > 0 and intervals_h[r][inQueueEnd[r] - 1].L.cell == cell) :
                            validBegin[row][r] = inQueueEnd[r] - 1
                            continue
                        validBegin[row][r] = inQueueEnd[r]
                    
                inQueueEnd[row]+=1
                if insert_points.get(row) is not None:
                    btmRows = insert_points[row][0]
                    topRows = insert_points[row][1]
                else: continue
                for btmRow,topRow in zip(btmRows,topRows):
                    indexes = [-1 for _ in range(nsegs_h)]
                    insert_btm_y = localSegments_h[btmRow].y
                    insert_top_y = localSegments_h[topRow].y+localSegments_h[topRow].h
                    if insert_top_y <= targetY+targetCellH:
                        placeY = insert_top_y - targetCellH
                    elif insert_btm_y >= targetY:
                        placeY = insert_btm_y
                    else:
                        placeY = targetY
                    nTotal = 1
                    r = btmRow
                    while r<=topRow and nTotal==1:
                        if (r == row) :
                            indexes[r] = inQueueEnd[r] - 1
                        else :
                            indexes[r] = validBegin[row][r]
                            nTotal *= (inQueueEnd[r] - validBegin[row][r])
                        r+=1
                    if (nTotal!=1):
                        continue
                    index = 0
                    while index != nTotal:
                        for r in range(btmRow,topRow):
                            if (indexes[r] >= inQueueEnd[r]) :
                                if (r == row) :
                                    indexes[r] = inQueueEnd[r] - 1
                                else :
                                    indexes[r] = validBegin[row][r]
                                
                                indexes[r + 1]+=1
                            else:
                                break
                        rangeBoundL = lx
                        rangeBoundR = hx
                        for r in range(btmRow,topRow+1):
                            rangeBoundL = max(intervals_h[r][indexes[r]].L.pos, rangeBoundL)
                            rangeBoundR = min(intervals_h[r][indexes[r]].R.pos, rangeBoundR)
                        if (rangeBoundL > rangeBoundR):
                            continue
                        woDistL = -sys.maxsize - 1
                        woDistR = sys.maxsize
                        for r in range(btmRow,topRow+1):
                            woDistL = max(intervals_h[r][indexes[r]].optPointL() + 0.5, woDistL)
                            woDistR = min(intervals_h[r][indexes[r]].optPointR() + 0.5, woDistR)
                        
                        woDistL = max(woDistL, targetX)
                        woDistR = min(woDistR, targetX)
                        woDistL = min(woDistL, rangeBoundR)
                        woDistR = max(woDistR, rangeBoundL)
                        placeX = targetX
                        if (woDistL != targetX or targetX != woDistR) :
                            critiPts = []
                            posL = [-1 for _ in range(nlocalcells)]
                            localRegion.estimateL(posL, intervals_h, indexes, woDistR, targetCell)
                            for i in range(nlocalcells):
                                if (posL[i] >= 0) :
                                    shift = woDistR - posL[i]
                                    curX = localRegion.localCells[i].lx + shift
                                    gpX = localRegion.localCells[i].cell.ox() + shift
                                    if (curX <= gpX) :
                                        if (curX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(curX, -1, 0))
                                    else :
                                        if (gpX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(gpX, -2, 0))
                                        if (curX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(curX, 1, 0))
                            posR = [-1 for _ in range(nlocalcells)]
                            localRegion.estimateR(posR, intervals_h, indexes, woDistL, targetCell)
                            for i in range(nlocalcells):
                                if posR[i] >= 0:
                                    shift = woDistL - posR[i]
                                    curX = localRegion.localCells[i].lx + shift
                                    gpX = localRegion.localCells[i].cell.ox() + shift
                                    if curX >= gpX:
                                        if curX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(curX, 0, 1))
                                    else :
                                        if curX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(curX, 0, -1))
                                        if gpX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(gpX, 0, 2))
                            critiPts.append(CriticalPoint(targetX, -1, 1))
                            placeX = self.getOptimalX(critiPts,rangeBoundL, rangeBoundR)
                        placeX = max(rangeBoundL, min(rangeBoundR, placeX))
                        targetCostX = 0
                        targetCostY = 0
                        targetCostX = abs(targetX - placeX)
                        targetCostY = abs(targetY - placeY)
                        localCostX = 0
                        localCostY = 0
                        posL = [-1 for _ in range(nlocalcells)]
                        posR = [-1 for _ in range(nlocalcells)]
                        localRegion.estimateH(posL, posR, intervals_h, indexes, placeX, targetCell)
                        for i in range(nlocalcells):
                            cell = localRegion.localCells[i]
                            if (posL[i] >= 0):
                                localCostX += (abs(posL[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                                
                            if (posR[i] >= 0):
                                localCostX += (abs(posR[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                        costx = targetCostX + localCostX
                        costy = targetCostY + localCostY
                        cost = costx + costy
                        if (cost < bestCost):
                            bestX = placeX
                            bestY = placeY
                            bestCost = cost
                            for r in range(nsegs_h):
                                bestIntervals[r] = indexes[r]
                        
                        key_str = ''
                        lset = set()
                        rset = set()
                        for rr in range(btmRow,topRow+1):
                            LCell = intervals_h[rr][indexes[rr]].L.cell
                            RCell = intervals_h[rr][indexes[rr]].R.cell
                            if LCell is not None:
                                lset.add(LCell.i)
                            if RCell is not None:
                                rset.add(RCell.i)       
                        key_str = '.'.join([str(ks) for ks in sorted(lset)])+'-'+'.'.join([str(ks) for ks in sorted(rset)])
                        if cost_map.get(key_str) is not None:
                            if costx not in cost_map[key_str]:
                                print(key_str,costx, cost_map[key_str])
                                cost_map[key_str].add(costx)
                        else:cost_map[key_str] = set([costx])
                        index+=1
                        indexes[btmRow]+=1
            '''insert_points_low = dict()
            insert_points_up = dict()
            for row in range(nsegs_h):
                btmRows = []
                topRows = []
                if low_btmRow_map.get(row) is not None:
                    btmRow = low_btmRow_map[row]
                    topRow = low_topRow_map[row]
                    while True:
                        btmRows.append(btmRow)
                        topRows.append(topRow)
                        if low_btmRow_map.get(topRow+1) is not None and low_btmRow_map[topRow+1] <= row and low_topRow_map[topRow+1] >= row:
                            btmRow = low_btmRow_map[topRow+1]
                            topRow = low_topRow_map[topRow+1]
                        else:break
                if row >= begin_row and end_row >= row:
                    btmRows.append(begin_row)
                    topRows.append(end_row)
                btmRows.reverse()
                topRows.reverse()
                insert_points_low[row]=[btmRows,topRows]
            for row in range(nsegs_h):
                btmRows = []
                topRows = []
                if up_btmRow_map.get(row) is not None:
                    btmRow = up_btmRow_map[row]
                    topRow = up_topRow_map[row]
                    while True:
                        btmRows.append(btmRow)
                        topRows.append(topRow)
                        if up_btmRow_map.get(btmRow-1) is not None and up_btmRow_map[btmRow-1]<= row and up_topRow_map[btmRow-1] >= row:
                            topRow = up_topRow_map[btmRow-1]
                            btmRow = up_btmRow_map[btmRow-1] 
                        else:break
                btmRows.reverse()
                topRows.reverse()
                insert_points_up[row]=[btmRows,topRows]
            sum_h = 0
            sum_h_cnt = 0
            segkeys = set([k for k in insert_points_up.keys()]+[k for k in insert_points_low.keys()])
            for k in segkeys:
                temp_h = 0
                for btm,top in zip(insert_points_low[k][0],insert_points_low[k][1]):
                    temp_h+=top-btm+1
                for btm,top in zip(insert_points_up[k][0],insert_points_up[k][1]):
                        temp_h+=top-btm+1
                if k >=begin_row and k <= end_row:
                    temp_h+=end_row- begin_row+1
                sum_h+=temp_h
                sum_h_cnt+=1 if temp_h !=0 else 0
                if temp_h>=400:
                    self.over_cnt[6]+=1
                elif temp_h>=255:
                    self.over_cnt[5]+=1
                elif temp_h>=100:
                    self.over_cnt[4]+=1
                elif temp_h>=64:
                    self.over_cnt[3]+=1
                elif temp_h>=36:
                    self.over_cnt[2]+=1
                elif temp_h>=16:
                    self.over_cnt[1]+=1
                else:self.over_cnt[0]+=1
                if self.max_h<temp_h :self.max_h=temp_h
            self.h_sum+=sum_h
            self.h_sum_cnt+=sum_h_cnt
            intervals_h = [[LegalMoveInterval(r) for _ in range(localSegments_h[r].nCells()+1)] for r in range(nsegs_h)]
            pos = [[localCell.lx,localCell.ly] for localCell in localRegion.localCells]
            pos_h = [i[0] for i in pos]
            localRegion.placeL(pos_h)
            for r in range(nsegs_h):
                segment = localSegments_h[r]
                intervals_h[r][0].L.pos = segment.x
                intervals_h[r][0].L.cell = None
                for i in range(segment.nCells()):
                    localCell = localRegion.localCells[segment.localCells[i]]
                    lb = pos_h[localCell.i] + localCell.w
                    intervals_h[r][i + 1].L.pos = lb
                    intervals_h[r][i + 1].L.cell = localCell
            localRegion.placeR(pos_h)
            for r in range(nsegs_h):
                segment = localSegments_h[r]
                for i in range(segment.nCells()):
                    localCell = localRegion.localCells[segment.localCells[i]]
                    rb = pos_h[localCell.i] - targetCell.width
                    intervals_h[r][i].R.pos = rb
                    intervals_h[r][i].R.cell = localCell
                intervals_h[r][segment.nCells()].R.pos = segment.x + segment.w - targetCell.width
                intervals_h[r][segment.nCells()].R.cell = None
            for r in range(nsegs_h):
                nIntervals = len(intervals_h[r])
                for i in range(nIntervals-1,-1,-1):
                    if intervals_h[r][i].L.pos > intervals_h[r][i].R.pos:
                        intervals_h[r].remove(intervals_h[r][i])
                segment = localSegments_h[r]
                nIntervals = len(intervals_h[r])
                for i in range(nIntervals):
                    intervals_h[r][i].i = i
                    lCell = intervals_h[r][i].L.cell
                    rCell = intervals_h[r][i].R.cell
                    if lCell:
                        intervals_h[r][i].setOptPointL(lCell.ox()+lCell.w)
                    else:
                        intervals_h[r][i].setOptPointL(segment.x)
                    if rCell:
                        intervals_h[r][i].setOptPointR(rCell.ox() - targetCellW)
                    else:
                        intervals_h[r][i].setOptPointR(segment.x + segment.w - targetCell.w)
                    intervals_h[r][i].L.interval = intervals_h[r][i]
                    intervals_h[r][i].R.interval = intervals_h[r][i]
            intervalSort_low = []
            for r in range(end_row+1):
                for i in range(len(intervals_h[r])):
                    intervalSort_low.append(intervals_h[r][i].L)
                    intervalSort_low.append(intervals_h[r][i].R)
            intervalSort_low.sort(key=functools.cmp_to_key(comp_low))
            intervalSort_up = []
            for r in range(begin_row,nsegs_h):
                for i in range(len(intervals_h[r])):
                    intervalSort_up.append(intervals_h[r][i].L)
                    intervalSort_up.append(intervals_h[r][i].R)
            intervalSort_up.sort(key=functools.cmp_to_key(comp_up))
            inQueueBgn = [0 for _ in range(nsegs_h)]
            inQueueEnd = [0 for _ in range(nsegs_h)]
            validBegin = [[0 for _ in range(nsegs_h)] for _ in range(nsegs_h)]
            posL = [-1 for _ in range(localRegion.nCells()+1)]
            posR = [-1 for _ in range(localRegion.nCells()+1)]
            bestX = sys.maxsize
            bestY = sys.maxsize
            bestCost = sys.maxsize
            bestIntervals = [0 for _ in range(nsegs_h)]
            for endpoint in intervalSort_low:
                row = endpoint.interval.r
                if endpoint.side == 'R':
                    for r in range(nsegs_h):
                        if validBegin[r][row] == inQueueBgn[row]:
                            validBegin[r][row]+=1
                    inQueueBgn[row]+=1
                    continue

                cell = endpoint.interval.L.cell
                if (cell and cell.h_nsegs > 1):
                    for r in range(cell.h_start_r,cell.h_start_r + cell.h_nsegs): 
                        if (r == row) :
                            continue
                        if (inQueueEnd[r] > 0 and intervals_h[r][inQueueEnd[r] - 1].L.cell == cell) :
                            validBegin[row][r] = inQueueEnd[r] - 1
                            continue
                        validBegin[row][r] = inQueueEnd[r]
                    
                inQueueEnd[row]+=1
                if insert_points_low.get(row) is not None:
                    btmRows = insert_points_low[row][0]
                    topRows = insert_points_low[row][1]
                else: continue
                for btmRow,topRow in zip(btmRows,topRows):
                    indexes = [-1 for _ in range(nsegs_h)]
                    insert_btm_y = localSegments_h[btmRow].y
                    insert_top_y = localSegments_h[topRow].y+localSegments_h[topRow].h
                    if insert_top_y <= targetY+targetCellH:
                        placeY = insert_top_y - targetCellH
                    elif insert_btm_y >= targetY:
                        placeY = insert_btm_y
                    else:
                        placeY = targetY
                    if abs(placeY - targetY) > bestCost:continue
                    nTotal = 1
                    r = btmRow
                    while r<=topRow and nTotal==1:
                        if (r == row) :
                            indexes[r] = inQueueEnd[r] - 1
                        else :
                            indexes[r] = validBegin[row][r]
                            nTotal *= (inQueueEnd[r] - validBegin[row][r])
                        r+=1
                    if (nTotal!=1):
                        continue
                    index = 0
                    while index != nTotal:
                        for r in range(btmRow,topRow):
                            if (indexes[r] >= inQueueEnd[r]) :
                                if (r == row) :
                                    indexes[r] = inQueueEnd[r] - 1
                                else :
                                    indexes[r] = validBegin[row][r]
                                
                                indexes[r + 1]+=1
                            else:
                                break
                        rangeBoundL = lx
                        rangeBoundR = hx
                        for r in range(btmRow,topRow+1):
                            rangeBoundL = max(intervals_h[r][indexes[r]].L.pos, rangeBoundL)
                            rangeBoundR = min(intervals_h[r][indexes[r]].R.pos, rangeBoundR)
                        if (rangeBoundL > rangeBoundR):
                            continue
                        woDistL = -sys.maxsize - 1
                        woDistR = sys.maxsize
                        for r in range(btmRow,topRow+1):
                            woDistL = max(intervals_h[r][indexes[r]].optPointL() + 0.5, woDistL)
                            woDistR = min(intervals_h[r][indexes[r]].optPointR() + 0.5, woDistR)
                        
                        woDistL = max(woDistL, targetX)
                        woDistR = min(woDistR, targetX)
                        woDistL = min(woDistL, rangeBoundR)
                        woDistR = max(woDistR, rangeBoundL)
                        placeX = targetX
                        if (woDistL != targetX or targetX != woDistR) :
                            critiPts = []
                            posL = [-1 for _ in range(localRegion.nCells())]
                            localRegion.estimateL(posL, intervals_h, indexes, woDistR, targetCell)
                            for i in range(localRegion.nCells()):
                                if (posL[i] >= 0) :
                                    shift = woDistR - posL[i]
                                    curX = localRegion.localCells[i].lx + shift
                                    gpX = localRegion.localCells[i].cell.ox() + shift
                                    if (curX <= gpX) :
                                        if (curX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(curX, -1, 0))
                                    else :
                                        if (gpX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(gpX, -2, 0))
                                        if (curX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(curX, 1, 0))
                            posR = [-1 for _ in range(localRegion.nCells())]
                            localRegion.estimateR(posR, intervals_h, indexes, woDistL, targetCell)
                            for i in range(localRegion.nCells()):
                                if posR[i] >= 0:
                                    shift = woDistL - posR[i]
                                    curX = localRegion.localCells[i].lx + shift
                                    gpX = localRegion.localCells[i].cell.ox() + shift
                                    if curX >= gpX:
                                        if curX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(curX, 0, 1))
                                    else :
                                        if curX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(curX, 0, -1))
                                        if gpX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(gpX, 0, 2))
                            critiPts.append(CriticalPoint(targetX, -1, 1))
                            placeX = self.getOptimalX(critiPts,rangeBoundL, rangeBoundR)
                        placeX = max(rangeBoundL, min(rangeBoundR, placeX))
                        targetCostX = 0
                        targetCostY = 0
                        targetCostX = abs(targetX - placeX)
                        targetCostY = abs(targetY - placeY)
                        localCostX = 0
                        localCostY = 0
                        posL = [-1 for _ in range(localRegion.nCells())]
                        posR = [-1 for _ in range(localRegion.nCells())]
                        localRegion.estimateH(posL, posR, intervals_h, indexes, placeX, targetCell)
                        for i in range(localRegion.nCells()):
                            cell = localRegion.localCells[i]
                            if (posL[i] >= 0):
                                localCostX += (abs(posL[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                                
                            if (posR[i] >= 0):
                                localCostX += (abs(posR[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                        costx = targetCostX + localCostX
                        costy = targetCostY + localCostY
                        cost = costx + costy
                        if (cost < bestCost):
                            bestX = placeX
                            bestY = placeY
                            bestCost = cost
                            for r in range(nsegs_h):
                                bestIntervals[r] = indexes[r]
                        index+=1
                        indexes[btmRow]+=1
            inQueueBgn = [0 for _ in range(nsegs_h)]
            inQueueEnd = [0 for _ in range(nsegs_h)]
            validBegin = [[0 for _ in range(nsegs_h)] for _ in range(nsegs_h)]
            posL = [-1 for _ in range(localRegion.nCells()+1)]
            posR = [-1 for _ in range(localRegion.nCells()+1)]
            for endpoint in intervalSort_up:
                row = endpoint.interval.r
                if endpoint.side == 'R':
                    for r in range(nsegs_h):
                        if validBegin[r][row] == inQueueBgn[row]:
                            validBegin[r][row]+=1
                    inQueueBgn[row]+=1
                    continue

                cell = endpoint.interval.L.cell
                if (cell and cell.h_nsegs > 1):
                    for r in range(cell.h_start_r,cell.h_start_r + cell.h_nsegs): 
                        if (r == row) :
                            continue
                        if (inQueueEnd[r] > 0 and intervals_h[r][inQueueEnd[r] - 1].L.cell == cell) :
                            validBegin[row][r] = inQueueEnd[r] - 1
                            continue
                        validBegin[row][r] = inQueueEnd[r]
                    
                inQueueEnd[row]+=1
                if insert_points_up.get(row) is not None:
                    btmRows = insert_points_up[row][0]
                    topRows = insert_points_up[row][1]
                else: continue
                for btmRow,topRow in zip(btmRows,topRows):
                    indexes = [-1 for _ in range(nsegs_h)]
                    insert_btm_y = localSegments_h[btmRow].y
                    insert_top_y = localSegments_h[topRow].y+localSegments_h[topRow].h
                    if insert_top_y <= targetY+targetCellH:
                        placeY = insert_top_y - targetCellH
                    elif insert_btm_y >= targetY:
                        placeY = insert_btm_y
                    else:
                        placeY = targetY
                    if abs(placeY - targetY) > bestCost:continue
                    nTotal = 1
                    r = btmRow
                    while r<=topRow and nTotal==1:
                        if (r == row) :
                            indexes[r] = inQueueEnd[r] - 1
                        else :
                            indexes[r] = validBegin[row][r]
                            nTotal *= (inQueueEnd[r] - validBegin[row][r])
                        r+=1
                    if (nTotal!=1):
                        continue
                    index = 0
                    while index != nTotal:
                        for r in range(btmRow,topRow):
                            if (indexes[r] >= inQueueEnd[r]) :
                                if (r == row) :
                                    indexes[r] = inQueueEnd[r] - 1
                                else :
                                    indexes[r] = validBegin[row][r]
                                
                                indexes[r + 1]+=1
                            else:
                                break
                        rangeBoundL = lx
                        rangeBoundR = hx
                        for r in range(btmRow,topRow+1):
                            rangeBoundL = max(intervals_h[r][indexes[r]].L.pos, rangeBoundL)
                            rangeBoundR = min(intervals_h[r][indexes[r]].R.pos, rangeBoundR)
                        if (rangeBoundL > rangeBoundR):
                            continue
                        woDistL = -sys.maxsize - 1
                        woDistR = sys.maxsize
                        for r in range(btmRow,topRow+1):
                            woDistL = max(intervals_h[r][indexes[r]].optPointL() + 0.5, woDistL)
                            woDistR = min(intervals_h[r][indexes[r]].optPointR() + 0.5, woDistR)
                        
                        woDistL = max(woDistL, targetX)
                        woDistR = min(woDistR, targetX)
                        woDistL = min(woDistL, rangeBoundR)
                        woDistR = max(woDistR, rangeBoundL)
                        placeX = targetX
                        if (woDistL != targetX or targetX != woDistR) :
                            critiPts = []
                            posL = [-1 for _ in range(localRegion.nCells())]
                            localRegion.estimateL(posL, intervals_h, indexes, woDistR, targetCell)
                            for i in range(localRegion.nCells()):
                                if (posL[i] >= 0) :
                                    shift = woDistR - posL[i]
                                    curX = localRegion.localCells[i].lx + shift
                                    gpX = localRegion.localCells[i].cell.ox() + shift
                                    if (curX <= gpX) :
                                        if (curX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(curX, -1, 0))
                                    else :
                                        if (gpX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(gpX, -2, 0))
                                        if (curX >= rangeBoundL) :
                                            critiPts.append(CriticalPoint(curX, 1, 0))
                            posR = [-1 for _ in range(localRegion.nCells())]
                            localRegion.estimateR(posR, intervals_h, indexes, woDistL, targetCell)
                            
                            for i in range(localRegion.nCells()):
                                if posR[i] >= 0:
                                    shift = woDistL - posR[i]
                                    curX = localRegion.localCells[i].lx + shift
                                    gpX = localRegion.localCells[i].cell.ox() + shift
                                    if curX >= gpX:
                                        if curX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(curX, 0, 1))
                                    else :
                                        if curX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(curX, 0, -1))
                                        if gpX <= rangeBoundR:
                                            critiPts.append(CriticalPoint(gpX, 0, 2))
                            critiPts.append(CriticalPoint(targetX, -1, 1))
                            placeX = self.getOptimalX(critiPts,rangeBoundL, rangeBoundR)
                        placeX = max(rangeBoundL, min(rangeBoundR, placeX))
                        targetCostX = 0
                        targetCostY = 0
                        targetCostX = abs(targetX - placeX)
                        targetCostY = abs(targetY - placeY)
                        localCostX = 0
                        localCostY = 0
                        posL = [-1 for _ in range(localRegion.nCells())]
                        posR = [-1 for _ in range(localRegion.nCells())]
                        localRegion.estimateH(posL, posR, intervals_h, indexes, placeX, targetCell)
                        for i in range(localRegion.nCells()):
                            cell = localRegion.localCells[i]
                            if (posL[i] >= 0):
                                localCostX += (abs(posL[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                                
                            if (posR[i] >= 0):
                                localCostX += (abs(posR[i] - cell.cell.ox()) - abs(cell.lx - cell.cell.ox()))
                        costx = targetCostX + localCostX
                        costy = targetCostY + localCostY
                        cost = costx + costy
                        if (cost < bestCost):
                            bestX = placeX
                            bestY = placeY
                            bestCost = cost
                            for r in range(nsegs_h):
                                bestIntervals[r] = indexes[r]
                        index+=1
                        indexes[btmRow]+=1'''
            if (bestCost == sys.maxsize):
                return False
            targetCell.setLx(bestX)
            targetCell.setLy(bestY)
            localRegion.addLocalCell(targetCell)
            #localRegion.addLocalCell(targetCell,bestX,bestY)
            nlocalcells = localRegion.nCells()
            newCellIndex = nlocalcells-1
            for r in range(nsegs_h):
                if (bestIntervals[r] < 0):
                    continue
                rCell = intervals_h[r][bestIntervals[r]].R.cell
                if (rCell is None) :
                    localSeg = localSegments_h[r]
                    localSeg.addLocalCell(newCellIndex)
                else:
                    cellRIndex = rCell.i
                    localSeg = localSegments_h[r]
                    localSeg.addLocalCell(newCellIndex)
                    for i in range(localSeg.nCells() - 2,-1,-1):
                        localSeg.localCells[i + 1] = localSeg.localCells[i]
                        if (localSeg.localCells[i] == cellRIndex):
                            localSeg.localCells[i] = newCellIndex
                            break
            pos_h = [localRegion.localCells[i].lx for i in range(nlocalcells)]
            localRegion.placeSpreadL(newCellIndex, pos_h)
            localRegion.placeSpreadR(newCellIndex, pos_h)
            move.x = bestX
            move.y = bestY
            nCells = nlocalcells - 1
            for i in range(nCells):
                cell = localRegion.localCells[i]
                if (cell.fixed):
                    continue
                if (cell.lx != cell.cell.lx() or cell.lx != pos_h[cell.i]):
                    move.legals.append((CellMove(cell.cell, pos_h[cell.i], cell.ly)))
            self.localSeg_sum+=nsegs_h
            self.cnt+=1
            if nsegs_h >self.max_seg:
                self.max_seg = nsegs_h
            return True
            
        def isLegalizeMove(move,region):
            isLegalizable = False
            alreadyLegal = True
            targetCell = move.cell
            cell_w = targetCell.width
            cell_h = targetCell.height
            targetX = move.x
            targetY = move.y
            cclx = targetCell.lx()
            ccly = targetCell.ly()
            placed = targetCell.placed
            if placed:
                self.removeCell(targetCell)
                targetCell.placed = False
            start = floor((targetY-self.yl)/self.row_height)
            end = floor((targetY-self.yl + cell_h -1)/self.row_height)
            start = max(0,start-1)
            end = min(self.repeat_count-1,end+1)
            segs = self.segments
            neibor_cell_set = set()
            for i in range(start,end+1):
                result,s,e = segs[i].getCellsAt(targetX-cell_w+1,targetX+cell_w)
                for c in result:
                    if c.i not in neibor_cell_set:
                        neibor_cell_set.add(c.i)
                    else:continue
                    if targetY+cell_h>c.ly() and c.hy()>targetY:
                        alreadyLegal = False
                        break
            if not alreadyLegal:
                isLegalizable = isLegalMoveMLL(move, region)
                pass
            else: isLegalizable = True
            if placed and not isLegalizable:
                #move.x = targetCell.lx()
                #move.y = targetCell.ly()
                #moveCell(targetCell, move.x, move.y)
                moveCell(targetCell, cclx, ccly)
            return isLegalizable

        def doLegalMove(move):
            targetCell = move.cell
            segment2Cell = dict()
            cell_map = dict()
            for cellmove in move.legals:
                #shiftCell(cellmove.cell, cellmove.x, cellmove.y)
                cell = cellmove.cell
                cell.setLx(cellmove.x)
                cell.setLy(cellmove.y)
                cell_map[cell.i] = 1
                for seg in cell.seg:
                    if(segment2Cell.get(seg) is None):
                        segment2Cell[seg] = [cell]
                    else:
                        segment2Cell[seg].append(cell)
            for k,_ in segment2Cell.items():
                v = self.seg2cells[k]
                if len(segment2Cell[k])!=0:
                    a = v[0]
                    b = 0
                    ans = []
                    while a<v[1] and b<len(segment2Cell[k]): 
                        if(cell_map.get(self.segments[k].cells[a].i) is not None):
                            a+=1
                            continue
                        if(self.segments[k].cells[a].lx()>segment2Cell[k][b].lx()):
                            ans.append(segment2Cell[k][b])
                            b+=1
                        else:
                            ans.append(self.segments[k].cells[a])
                            a+=1
                    while a<v[1]:
                        if(cell_map.get(self.segments[k].cells[a].i) is not None):
                            a+=1
                            continue
                        ans.append(self.segments[k].cells[a])
                        a+=1
                    while b<len(segment2Cell[k]): 
                        ans.append(segment2Cell[k][b])
                        b+=1
                    self.segments[k].cells[v[0]:v[1]] = ans           
            if (targetCell): 
                moveCell(targetCell, move.x, move.y)
            pass
        
        def defineLocalRegion(cell,localRegion,region):
            lx = region.lx
            ly = region.ly
            hx = region.hx
            hy = region.hy
            localRegion.lx = lx
            localRegion.ly = ly
            localRegion.hx = hx
            localRegion.hy = hy
            total_area = (hy-ly)*(hx-lx)
            neibor_cell_set = set()
            segs = self.segments
            start = floor((localRegion.ly-self.yl)/self.row_height)
            end = floor((localRegion.hy-self.yl-1)/self.row_height)
            area = 0
            localCells = []
            self.seg2cells.clear()
            for i in range(start,end+1):
                segCells = []
                #for c in segs[i].getCellsAt(localRegion.lx,localRegion.hx): 
                result,s,e = segs[i].getCellsAt(localRegion.lx-cell.w+1,localRegion.hx)
                for c in result:
                    if c.i not in neibor_cell_set:
                        neibor_cell_set.add(c.i)
                    else:continue
                    if hy>c.ly() and c.hy() >ly:
                        segCells.append(c)
                        #这里有问题
                self.seg2cells[i] = (s,e)
                localCells = list(heapq.merge(localCells,segCells,key=lambda x:x._x))
            for c in localCells:
                area+=localRegion.addLocalCell(c)
            if (area+cell.area)/(total_area+0.00001) > 1.0:return True
            else:
                localRegion.buildLocalSegments()
            return False
        if flag:
            last_move = self.abacus_move
        else:
            last_move = sys.maxsize
        best_move = sys.maxsize
        cur_move = 0
        stop = False
        self.best_result = []
        for cell in self.cells:
            self.best_result.append((cell.lx(),cell.ly()))
        for i in range(1):
            cells = self.cells
            op_cell2region = dict()
            ori_cell2region = {cell:cell.getOrigionRegion(self.LocalRegionW, self.LocalRegionH,self.xl,self.yl,self.xh,self.yh) for cell in cells}

            while len(ori_cell2region) != 0:
                for cell,region in ori_cell2region.items():
                    op_cell2region[cell] = region
                ori_cell2region.clear()
                for cell,region in op_cell2region.items():
                    cell.tx(cell.ox())
                    cell.ty(cell.oy())
                    move = Move(cell,cell.getTx(),cell.getTy())
                    if isLegalizeMove(move,region=region):
                        doLegalMove(move)
                        pass
                    else:
                        ori_cell2region[cell] = region
                op_cell2region.clear()   
                cur_move = self.move()
                imp = last_move - cur_move
                print(len(ori_cell2region),imp,cur_move)
                if flag and imp/last_move<0.005: 
                    for cell,region in ori_cell2region.items():
                        cell.placed = True
                    break
                last_move = cur_move
                for cell,region in ori_cell2region.items():
                    if len(ori_cell2region) == 2:
                        print(region.lx,region.ly,region.hx,region.hy)
                    region.expand(self.expandW,self.expandH,self.xl,self.yl,self.xh,self.yh)
                    op_cell2region[cell] = region        
            '''tyl = self.yl
            tyh = self.yh
            self.yl = self.xl
            self.yh = self.xh
            self.xl = 0
            self.xh = tyh-tyl
            self.repeat_count = ceil((self.yh-self.yl)/self.row_height)
            self.buildSegments()
            cells = self.cells
            for cell in cells:
                clx = cell.lx()
                cox = cell.ox()
                cell.setOx(tyh-cell.oy()-cell.h)
                cell.setOy(cox)
                if (tyh-cell.ly()-cell.h) <0:print('error')
                moveCell(cell,tyh-cell.ly()-cell.h,clx)
            op_cell2region = dict()
            ori_cell2region = {cell:cell.getOrigionRegion(self.LocalRegionW, self.LocalRegionH,self.xl,self.yl,self.xh,self.yh) for cell in cells}

            #last_move = sys.maxsize
            while len(ori_cell2region) != 0:
                for cell,region in ori_cell2region.items():
                    op_cell2region[cell] = region
                ori_cell2region.clear()
                for cell,region in op_cell2region.items():
                    cell.tx(cell.ox())
                    cell.ty(cell.oy())
                    move = Move(cell,cell.getTx(),cell.getTy())
                    if isLegalizeMove(move,region=region):
                        doLegalMove(move)
                        pass
                    else:
                        ori_cell2region[cell] = region
                op_cell2region.clear()   

                cur_move = self.move(v=True)
                imp = last_move - cur_move
                print(imp,cur_move)
                if flag and imp/last_move<0.005: 
                    for cell,region in ori_cell2region.items():
                        cell.placed = True
                    break
                last_move = cur_move
                for cell,region in ori_cell2region.items():
                    region.expand(self.expandW,self.expandH,self.xl,self.yl,self.xh,self.yh)
                    op_cell2region[cell] = region
            ttyl = self.yl
            ttyh = self.yh
            self.yl = tyl
            self.yh = tyh
            self.xl = ttyl
            self.xh = ttyh
            self.repeat_count = ceil((self.yh-self.yl)/self.row_height)
            self.buildSegments()
            for cell in cells:
                clx = cell.lx()
                cox = cell.ox()
                cell.setOx(cell.oy())
                cell.setOy(self.yh - cox - cell.h)
                moveCell(cell,cell.ly(),self.yh - clx - cell.h)'''
        print(best_move)
        self.result_1 = []
        for cell in self.cells:
            self.result_1.append((cell.lx()+self.half_terminal_spacing,cell.ly()+self.half_terminal_spacing))
        '''tyl = self.yl
        tyh = self.yh
        self.yl = self.xl
        self.yh = self.xh
        self.xl = 0
        self.xh = tyh-tyl
        self.repeat_count = ceil((self.yh-self.yl)/self.row_height)
        self.buildSegments()
        cells = self.cells
        for cell in cells:
            clx = cell.lx()
            cox = cell.ox()
            cell.setOx(tyh-cell.oy()-cell.h)
            cell.setOy(cox)
            if (tyh-cell.ly()-cell.h) <0:print('error')
            moveCell(cell,tyh-cell.ly()-cell.h,clx)
        op_cell2region = dict()
        ori_cell2region = {cell:cell.getOrigionRegion(self.LocalRegionW, self.LocalRegionH,self.xl,self.yl,self.xh,self.yh) for cell in cells}

        last_move = sys.maxsize
        while len(ori_cell2region) != 0:
            for cell,region in ori_cell2region.items():
                op_cell2region[cell] = region
            ori_cell2region.clear()
            for cell,region in op_cell2region.items():
                cell.tx(cell.ox())
                cell.ty(cell.oy())
                move = Move(cell,cell.getTx(),cell.getTy())
                if isLegalizeMove(move,region=region):
                    doLegalMove(move)
                    pass
                else:
                    ori_cell2region[cell] = region
            op_cell2region.clear()   
            print(len(ori_cell2region),flush = True)

            cur_move = self.move(v=True)
            imp = last_move - cur_move
            print(imp,last_move,cur_move)
            if flag and imp/last_move<0.005: 
                for cell,region in ori_cell2region.items():
                    cell.placed = True
                break 
            last_move = cur_move  
            for cell,region in ori_cell2region.items():
                region.expand(self.LocalRegionW/2,self.LocalRegionH/2,self.xl,self.yl,self.xh,self.yh)
                op_cell2region[cell] = region
        ttyl = self.yl
        ttyh = self.yh
        self.yl = tyl
        self.yh = tyh
        self.xl = ttyl
        self.xh = ttyh
        self.repeat_count = ceil((self.yh-self.yl)/self.row_height)
        self.buildSegments()
        for cell in cells:
            clx = cell.lx()
            cox = cell.ox()
            cell.setOx(cell.oy())
            cell.setOy(self.yh - cox - cell.h)
            moveCell(cell,cell.ly(),self.yh - clx - cell.h)
        #print('@@@,@@@vehicle',self.localSeg_sum/self.cnt,self.max_seg)
        #print('@@@,@@@vehicle',self.h_sum/self.h_sum_cnt,self.max_h,self.over_cnt)
        print(self.xl,self.xh,self.yl,self.yh)'''
        
        return

    def move(self,v = False):
        sum = 0
        if v:
            for i in range(len(self.cells)):
                sum+=(abs(self.cells[i].lx()-self.ori_x_v[i])+abs(self.cells[i].ly()-self.ori_y_v[i]))
        else:
            for i in range(len(self.cells)):
                sum+=(abs(self.cells[i].lx()-self.ori_x[i])+abs(self.cells[i].ly()-self.ori_y[i]))
        return sum
    
    def abacus(self,cnt):
        cells = self.cells
        dtype = np.float64
        if cnt == 0:
            node_x = np.array([cell.ox() for cell in cells]).astype(dtype)
            node_y = np.array([cell.oy() for cell in cells]).astype(dtype)
        else:
            self.buildSegments()
            node_x = np.array([cell.lx() for cell in cells]).astype(dtype)
            node_y = np.array([cell.ly() for cell in cells]).astype(dtype)
        node_size_x = np.array([self.terminal_size_w]*len(cells)).astype(dtype)
        node_size_y = np.array([self.terminal_size_h]*len(cells)).astype(dtype)
        num_terminals = 0 
        num_terminal_NIs = 0 
        num_filler_nodes = 0
        num_movable_nodes = len(node_x)-num_terminals-num_terminal_NIs-num_filler_nodes
        site_width = 1 
        row_height = self.terminal_size_h 
        num_bins_x = 2
        num_bins_y = 2
        flat_region_boxes = np.zeros(0, dtype=dtype)
        flat_region_boxes_start = np.array([0], dtype=np.int32)
        node2fence_region_map = np.zeros(0, dtype=np.int32)
        pos = Variable(torch.from_numpy(np.concatenate([node_x, node_y])))
        custom = greedy_legalize.GreedyLegalize(
        torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), torch.from_numpy(np.zeros_like(node_x)), 
        flat_region_boxes=torch.from_numpy(flat_region_boxes),          flat_region_boxes_start=torch.from_numpy(flat_region_boxes_start),node2fence_region_map=torch.from_numpy(node2fence_region_map), 
        xl=self.xl, yl=self.yl, xh=self.xh, yh=self.yh, 
        site_width=site_width, row_height=row_height, 
        num_bins_x=num_bins_x, num_bins_y=num_bins_y, 
        num_movable_nodes=num_movable_nodes, 
        num_terminal_NIs=num_terminal_NIs, 
        num_filler_nodes=num_filler_nodes)
        pos = Variable(torch.from_numpy(np.concatenate([node_x, node_y])))
        result = custom(pos, pos)
        a_custom = abacus_legalize.AbacusLegalize(
        torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), torch.from_numpy(np.zeros_like(node_x)), 
        flat_region_boxes=torch.from_numpy(flat_region_boxes),          flat_region_boxes_start=torch.from_numpy(flat_region_boxes_start),node2fence_region_map=torch.from_numpy(node2fence_region_map), 
        xl=self.xl, yl=self.yl, xh=self.xh, yh=self.yh,
        site_width=site_width, row_height=row_height, 
        num_bins_x=num_bins_x, num_bins_y=num_bins_y, 
        num_movable_nodes=num_movable_nodes, 
        num_terminal_NIs=num_terminal_NIs, 
        num_filler_nodes=num_filler_nodes)
        result = a_custom(pos,result)
        result = result.numpy()
        result = (result[:num_movable_nodes],result[num_movable_nodes:])
        self.abacus_move = 0
        for i,c in enumerate(cells):
            self.abacus_move+=(abs(result[0][i]-self.ori_x[i])+abs(result[1][i]-self.ori_y[i]))
            c.setLx(result[0][i])
            c.setLy(result[1][i])
            c.placed = True
            start = floor((c.ly()-self.yl)/self.row_height)
            end = floor((c.ly()-self.yl+c.h-1)/self.row_height)
            for i in range(start,end+1):
                self.segments[i].addCell(c)

    def tetris(self,cnt):
        cells = self.cells
        dtype = np.float64
        if cnt == 0:
            node_x = np.array([cell.ox() for cell in cells]).astype(dtype)
            node_y = np.array([cell.oy() for cell in cells]).astype(dtype)
        else:
            self.buildSegments()
            node_x = np.array([cell.lx() for cell in cells]).astype(dtype)
            node_y = np.array([cell.ly() for cell in cells]).astype(dtype)
        node_size_x = np.array([self.terminal_size_w]*len(cells)).astype(dtype)
        node_size_y = np.array([self.terminal_size_h]*len(cells)).astype(dtype)
        num_terminals = 0 
        num_terminal_NIs = 0 
        num_filler_nodes = 0
        num_movable_nodes = len(node_x)-num_terminals-num_terminal_NIs-num_filler_nodes
        site_width = 1 
        row_height = self.terminal_size_h 
        num_bins_x = 2
        num_bins_y = 2
        flat_region_boxes = np.zeros(0, dtype=dtype)
        flat_region_boxes_start = np.array([0], dtype=np.int32)
        node2fence_region_map = np.zeros(0, dtype=np.int32)
        pos = Variable(torch.from_numpy(np.concatenate([node_x, node_y])))
        custom = greedy_legalize.GreedyLegalize(
        torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), torch.from_numpy(np.zeros_like(node_x)), 
        flat_region_boxes=torch.from_numpy(flat_region_boxes),          flat_region_boxes_start=torch.from_numpy(flat_region_boxes_start),node2fence_region_map=torch.from_numpy(node2fence_region_map), 
        xl=self.xl, yl=self.yl, xh=self.xh, yh=self.yh, 
        site_width=site_width, row_height=row_height, 
        num_bins_x=num_bins_x, num_bins_y=num_bins_y, 
        num_movable_nodes=num_movable_nodes, 
        num_terminal_NIs=num_terminal_NIs, 
        num_filler_nodes=num_filler_nodes)
        pos = Variable(torch.from_numpy(np.concatenate([node_x, node_y])))
        result = custom(pos, pos)
        '''a_custom = abacus_legalize.AbacusLegalize(
        torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), torch.from_numpy(np.zeros_like(node_x)), 
        flat_region_boxes=torch.from_numpy(flat_region_boxes),          flat_region_boxes_start=torch.from_numpy(flat_region_boxes_start),node2fence_region_map=torch.from_numpy(node2fence_region_map), 
        xl=self.xl, yl=self.yl, xh=self.xh, yh=self.yh,
        site_width=site_width, row_height=row_height, 
        num_bins_x=num_bins_x, num_bins_y=num_bins_y, 
        num_movable_nodes=num_movable_nodes, 
        num_terminal_NIs=num_terminal_NIs, 
        num_filler_nodes=num_filler_nodes)
        #result = a_custom(pos,result)'''
        result = result.numpy()
        result = (result[:num_movable_nodes],result[num_movable_nodes:])
        self.abacus_move = 0
        for i,c in enumerate(cells):
            self.abacus_move+=(abs(result[0][i]-self.ori_x[i])+abs(result[1][i]-self.ori_y[i]))
            c.setLx(result[0][i])
            c.setLy(result[1][i])
            c.placed = True
            start = floor((c.ly()-self.yl)/self.row_height)
            end = floor((c.ly()-self.yl+c.h-1)/self.row_height)
            for i in range(start,end+1):
                self.segments[i].addCell(c)

if __name__ == "__main__":
    dplacer = Dplacer()
    die = Die()
    #die.setDieSize(0,0,180,200)
    #die.setTopDieRows(10,10,160,20,10)
    die.setDieSize(0,0,200,200)
    die.setTopDieRows(0,0,220,10,7)
    terminals = Terminals()
    terminals.setTerminalSize(20,20)
    terminals.setTerminalSpacing(20)
    xy = [[34,62],[66,45],[70,100],[46,135],[120,110],[125,58],[125,140]]#
    #xy = [[12,12],[12,12],[12,12],[12,10]]
    #xy = [[random.randint(20,360),random.randint(20,360)] for _ in range(70)]
    #xy = [[30,70],[85,55],[140,30],[100,85],[150,100]]#[65,88],[50,100],[115,85],
    l = len(xy)
    for i in range(l):
        terminal = Terminal(i)
        terminal.setPlaceCoordinate(xy[i][0],xy[i][1])
        terminals.addTerminal(terminal)
    database = DataBase()
    database.terminals = terminals
    database.die = die
    database.generateTerminalBookShelf_("temp_",top_flag=True,bottom_flag=False)
    output_hal = open("2.pkl", 'wb')
    str1 = pickle.dumps(database)
    output_hal.write(str1)
    output_hal.close()
    legalization = Legalization()
    legalization.read({'die':die,'terminals':terminals})
    legalization.scale_by_spacing()
    legalization.plot('init.png')
    dplacer.read({'die':die,'terminals':terminals})
    #dplacer.abacus()
    dplacer(False)
    result = dplacer.result()
    print(result)
    #print('@@@,@@@',dplacer.localSeg_sum/dplacer.cnt,dplacer.max_seg)
    #print('@@@,@@@',dplacer.h_sum/dplacer.h_sum_cnt,dplacer.max_h,dplacer.over_cnt)
    print(dplacer.time1,dplacer.time2)
    legalization.detailed_legalize()
    #legalization.rough_legalize()
    legalization.plot('abacus_.png')
    cost = 0
    max_mv = 0
    for i in range(len(xy)):
        xy_ = database.terminals.getTerminals()[i].getPlaceCoordinate()
        mv = (abs(xy_[0]-xy[i][0])+abs(xy_[1]-xy[i][1]))
        cost+=mv
        if mv >max_mv:
            max_mv = mv
    print("abacus_move",cost,max_mv)
    cost = 0
    max_mv = 0
    tl = terminals.getTerminals()
    index = 0
    for i in range(l):
        mv = (abs(result[i][0]-xy[i][0])+abs(result[i][1]-xy[i][1]))
        tl[i].setPlaceCoordinate(result[i][0],result[i][1])
        cost+= mv
        if mv >max_mv:
            max_mv = mv
            index = i
    print(cost,max_mv)
    #print(result[4][0],xy[4][0],result[4][1],xy[4][1])
    #print(result[6][0],xy[6][0],result[6][1],xy[6][1])
    leaglization = Legalization()
    leaglization.read(params={"die":die,"terminals":terminals})
    leaglization.scale_by_spacing()
    leaglization.plot("imp1.png")
    pass