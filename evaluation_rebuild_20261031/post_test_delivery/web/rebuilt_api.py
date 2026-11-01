"""Independent local API for the frozen seed42 deployment examples."""
from contextlib import asynccontextmanager
import io
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neu_protocol.core import Protocol, configure_runtime, load, sha

p = Protocol(ROOT)
configure_runtime(p)
import cv2
import httpx
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from ultralytics import YOLO
from neu_protocol.prediction import RuntimeDetector
import api_contracts as contract

torch.set_num_threads(2)
LOCK = threading.Lock()
POOL = {}
NAMES = dict(enumerate(p.config['names']))
LEGACY = load(Path(__file__).with_name('legacy_model.json'))
MODES = {
    'balanced': dict(label='GPU 平衡模式', families='AC', tta=True, device='cuda:0', conf=.35, nms=[.3]*6),
    'recall': dict(label='GPU 召回模式', families='AC', tta=True, device='cuda:0', conf=.35, nms=[.5,.3,.3,.3,.5,.3]),
    'cpu_balanced': dict(label='CPU 融合（无 TTA）', families='AC', tta=False, device='cpu', conf=.35, nms=[.3]*6),
    'cpu_single': dict(label='CPU C 单模型', families='C', tta=False, device='cpu', conf=.25, nms=[.7]*6),
    'legacy_onnx': dict(label='旧 mosaic000 ONNX', families='', tta=False, device='cpu', conf=.25, nms=[.7]*6),
}


def detector(mode):
    spec = MODES[mode]
    if spec['device'].startswith('cuda') and not torch.cuda.is_available():
        raise HTTPException(503, '此模式需要 CUDA GPU，请显式选择 CPU 模式。')
    if mode not in POOL:
        if mode == 'legacy_onnx':
            path = Path(LEGACY['path'])
            if not path.is_file() or sha(path) != LEGACY['sha256']:
                raise HTTPException(503, '旧模型缺失或哈希变化，回退入口拒绝加载。')
            POOL[mode] = YOLO(str(path), task='detect')
        else:
            paths = [ROOT / (f'exports/{family}_seed42.onnx' if spec['device']=='cpu'
                             else f'runs/{family}_seed42/weights/best.pt') for family in spec['families']]
            frozen = load(ROOT / 'freezes/final.json')
            for path in paths:
                if sha(path) != frozen['files'][str(path.relative_to(ROOT)).replace('\\','/')]:
                    raise HTTPException(503, '部署模型哈希与冻结记录不一致。')
            POOL[mode] = RuntimeDetector(paths, device=spec['device'], augment=spec['tta'], profile=spec['nms'])
    return POOL[mode]


@asynccontextmanager
async def lifespan(app):
    p.final_freeze()
    p.check_files(load(ROOT/'test/archive_manifest.json'))
    app.state.deepseek_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5,read=30,write=10,pool=5)) if contract.ENABLE_DEEPSEEK else None
    app.state.ready = False
    default = 'balanced' if torch.cuda.is_available() else 'cpu_single'
    with LOCK:
        model = detector(default)
        model.predict(Image.new('RGB',(200,200)),conf=MODES[default]['conf'],iou=.7,imgsz=640,device=MODES[default]['device'],verbose=False)
    app.state.default = default
    app.state.ready = True
    try:
        yield
    finally:
        if app.state.deepseek_client is not None:
            await app.state.deepseek_client.aclose()
        POOL.clear()


app = FastAPI(title='钢材缺陷检测：固定 seed42 升级接口', lifespan=lifespan)


@app.get('/health')
def health(request: Request):
    default = getattr(request.app.state,'default','balanced')
    return dict(status='ok' if getattr(request.app.state,'ready',False) else 'not_ready',
                model='NEU rebuilt seed42 / '+default,device=MODES[default]['device'],imgsz=640,
                class_names={str(k):v for k,v in NAMES.items()},seed=42,
                modes={k:dict(v,available=v['device']=='cpu' or torch.cuda.is_available()) for k,v in MODES.items()})


@app.post('/predict')
def predict(request:Request,file:UploadFile=File(...),mode:str=Query('balanced'),
            conf:float|None=Query(None,ge=.001,le=1),iou:float=Query(.7,gt=.01,lt=.99)):
    if mode not in MODES:
        raise HTTPException(422,'未知检测模式。')
    if not math.isclose(iou,.7,abs_tol=1e-9):
        raise HTTPException(422,'该升级入口固定首级 NMS=0.70，请使用模式选择切换第二级后处理。')
    data=file.file.read(contract.MAX_UPLOAD_BYTES+1)
    if len(data)>contract.MAX_UPLOAD_BYTES:
        raise HTTPException(413,'图片超过10 MiB。')
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width*source.height>25_000_000:
                raise HTTPException(413,'图片像素数过大。')
            image=ImageOps.exif_transpose(source).convert('RGB');image.load()
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError) as e:
        raise HTTPException(422,'无法解码上传图片。') from e
    spec=MODES[mode];threshold=spec['conf'] if conf is None else conf
    started=time.perf_counter()
    with LOCK:
        visible_before=os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            model=detector(mode)
            result=model.predict(image,imgsz=640,conf=threshold,iou=.7,max_det=300,device=spec['device'],verbose=False)[0]
            if spec['device'].startswith('cuda'):torch.cuda.synchronize()
        finally:
            # Legacy YOLO CPU selection writes CUDA_VISIBLE_DEVICES=-1.
            # Restore the process setting before another request can switch back.
            if visible_before is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=visible_before
    elapsed=1000*(time.perf_counter()-started)
    detections=[]
    for row in result.boxes.data.cpu().tolist():
        x1,y1,x2,y2,score,c=row;c=int(c)
        detections.append(contract.Detection(class_id=c,class_name=result.names[c],confidence=round(score,6),
            box_xyxy=contract.BoxXYXY(x1=round(min(max(x1,0),image.width),2),y1=round(min(max(y1,0),image.height),2),
                                     x2=round(min(max(x2,0),image.width),2),y2=round(min(max(y2,0),image.height),2))))
    response=contract.PredictResponse(model=f'{mode}_seed42' if mode!='legacy_onnx' else 'legacy_mosaic000',
          filename=(file.filename or '')[:255],image=contract.ImageInfo(width=image.width,height=image.height),
          settings=contract.PredictSettings(imgsz=640,conf=threshold,iou=.7,max_det=300,device=spec['device']),
          count=len(detections),processing_ms=round(elapsed,3),coordinate_system='xyxy pixel coordinates on the original uploaded image',detections=detections)
    return dict(**response.model_dump(),mode=mode,tta=spec['tta'],second_stage_nms=spec['nms'],
                frozen_workpoint=math.isclose(threshold,spec['conf'],abs_tol=1e-9),
                matches_gpu_tta=mode in ('balanced','recall'))


# The copied contract only broadens the device field to accept GPU responses.
# The existing DeepSeek behavior is reused, and sends data only on explicit /analyze.
app.add_api_route('/analyze',contract.analyze,methods=['POST'],response_model=contract.AnalyzeResponse)
