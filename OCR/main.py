from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
import cv2
import io
from fastapi import FastAPI, UploadFile, File, Form, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
import csv
#from fastapi.responses import StreamingResponse, HTMLResponse  #, JSONResponse
#from fastapi.staticfiles import StaticFiles
#from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import uvicorn
# import requests
from typing import List
import shutil

import base64
from pydantic import BaseModel
import glob
import os
import time
from datetime import datetime
#import pytz
#import re
#torch.cuda.is_available = lambda : False  # принудительно использовать только CPU для PyTorch
import sqlite3
import random

from bboxes import draw_boxes
from ocr_model import OCR_Model
from image_processing import black2white
from text_processing import GreedyDecoder
from data_matching import find_matching_rows
from image_processing import ocr_process_image

# Путь к папке для сохранения
save_path = './images'
default_mdl_path = './models/'  # Путь по умолчанию, где хранятся модели
data_path = './data'

bb_types = ['main', 'serial', 'rezim']

# Глобальные переменные
main_char_list = ['|', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0', '.', ' ']
serial_char_list = ['№', 'К', 'Г', 'В', '|', '_', 'O', 'K', 'A', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0', '.', '-', ' ']
rezim_char_list = ['|', '2', '1', '0', ' ']

app = FastAPI(title="Counters System API based on YOLOv8", version="0.3.2", debug=True)  # Инициализация FastAPI

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#local_tz = pytz.timezone("Europe/Moscow") # Пример часового пояса
#local_time = datetime.now(local_tz)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if obj != obj:  # проверка на NaN
                return None  # или другое значение по умолчанию
            elif obj == float('inf'):
                return 'Infinity'  # или другое значение по умолчанию
            elif obj == float('-inf'):
                return '-Infinity'  # или другое значение по умолчанию
        return super().default(obj)

class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(jsonable_encoder(content), ensure_ascii=False, cls=CustomJSONEncoder).encode('utf-8')


class ResponseModel(BaseModel):
    image: str
    results: dict

def get_latest_model(path):  # Функция для выбора самой последней модели
    list_of_files = glob.glob(f'{path}*.pt')
    if not list_of_files:
        return None  # Ни одного файла модели не найдено
    latest_model = max(list_of_files, key=os.path.getctime)  # Выбираем самый свежий файл
    print(f'♻️  Latest Model: {latest_model}')
    return latest_model


# Добавление статических файлов
#app.mount('/static', StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")

def generate_file_name(model_name, file_ext, images_dir='./images'):
    # Получение текущего времени
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # Выделение первых двух цифр из названия модели
    #model_digits = re.findall(r'\d+', model_name)[:2]
    #model_prefix = ''.join(model_digits)[:2]  # Берем только первые две цифры

    model_prefix = model_name[:2]

    # Определение порядкового номера файла
    existing_files = glob.glob(os.path.join(images_dir, '*.', file_ext))  # предполагаем, что изображения в формате .jpg
    next_file_number = len(existing_files) + 1

    # Генерация имени файла
    file_name = f"{current_time}_M{model_prefix}_{str(next_file_number).zfill(2)}.{file_ext}"

    return file_name


# Get client IP address
@app.get("/")
async def read_root(request: Request):
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_host = forwarded_for.split(",")[0]
    else:
        client_host = request.client.host
    return {"📡 Client IP: ": client_host}


@app.get('/info')
def read_root():
    return {'Project 2023': '📟 Counters, Москва, 2023 г.]'}


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...), request_type: str = Query("main", enum=["main", "serial", "rezim"])):
    # Сопоставляем тип запроса с соответствующим списком символов
    if request_type == "main":
        char_list = main_char_list
        model_name = 'OCR_main_3_12.pt'
    elif request_type == "serial":
        char_list = serial_char_list
        model_name = 'OCR_serials_18_12.pt'
    elif request_type == "rezim":
        char_list = rezim_char_list
        model_name = 'OCR_Reg_25_11.pt'

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Загрузка модели и таблицы из локальной папки data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(char_list) + 1  # +1 для символа пустого пробела
    ocr_model = OCR_Model(n_classes=n_classes)  # OCR_Model уже определена 
    ocr_model.load_state_dict(torch.load(f'data/{model_name}', map_location=device))
    ocr_model = ocr_model.to(device)

    # Обработка изображения и получение данных
    result = ocr_process_image(image, ocr_model, char_list)

    return result
