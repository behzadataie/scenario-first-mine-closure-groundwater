#!/usr/bin/env python3
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import pandas as pd

HEADER=np.dtype([('itemp1','<i4'),('itemp2','<i4'),('icount','<i4')])
COO=np.dtype([('i','<i4'),('j','<i4'),('dtemp','<f8')])
REC=np.dtype([('j','<i4'),('dtemp','<f8')])

def read_pest_binary(path: str|Path) -> pd.DataFrame:
    path=Path(path)
    with path.open('rb') as f:
        itemp1,itemp2,icount=np.fromfile(f,HEADER,1)[0]
        itemp1,itemp2,icount=int(itemp1),int(itemp2),int(icount)
        if itemp1==0 and itemp2==icount:
            raise NotImplementedError('dense sequential format not needed here')
        ncol,nrow=abs(itemp1),abs(itemp2)
        if itemp1>=0:
            data=np.fromfile(f,COO,icount)
            x=np.zeros((nrow,ncol),dtype=float)
            x[data['i'],data['j']]=data['dtemp']
            col_names=[]
            row_names=[]
            for _ in range(ncol):
                col_names.append(struct.unpack('200s',f.read(200))[0].strip().decode(errors='replace').lower())
            for _ in range(nrow):
                row_names.append(struct.unpack('200s',f.read(200))[0].strip().decode(errors='replace').lower())
        else:
            data=np.fromfile(f,REC,icount)
            icols=((data['j']-1)//nrow)+1
            irows=data['j']-((icols-1)*nrow)
            x=np.zeros((nrow,ncol),dtype=float)
            x[irows-1,icols-1]=data['dtemp']
            col_names=[]
            row_names=[]
            for _ in range(ncol):
                col_names.append(struct.unpack('12s',f.read(12))[0].strip().decode(errors='replace').lower())
            for _ in range(nrow):
                row_names.append(struct.unpack('20s',f.read(20))[0].strip().decode(errors='replace').lower())
    return pd.DataFrame(x,index=row_names,columns=col_names)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('input',type=Path)
    ap.add_argument('output',type=Path)
    a=ap.parse_args()
    df=read_pest_binary(a.input)
    df.to_csv(a.output)
    print(a.input,df.shape,'->',a.output)
