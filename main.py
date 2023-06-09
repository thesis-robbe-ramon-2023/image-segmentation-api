import base64
from fastapi import FastAPI, File, Form, UploadFile
from typing import Annotated
import cv2
import numpy as np
import shutil
import os

import imageio as iio

import matplotlib.pyplot as plt
from skimage import measure
from starlette.websockets import WebSocket
from fastapi.middleware.cors import CORSMiddleware

from geomeansegmentation.image_segmentation.drlse_segmentation import perform_segmentation, EdgeIndicator, PotentialFunction

import json


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with the appropriate origin URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parameter_cache = {}

async def _segment(websocket: WebSocket, img_name: str):
    img = cv2.imread(f"img/{img_name}")

    if parameter_cache[img_name]["edgeIndicator"] == 0:
        edgeIndicator = EdgeIndicator.SCALAR_DIFFERENCE
    elif parameter_cache[img_name]["edgeIndicator"] == 1:
        edgeIndicator = EdgeIndicator.GEODESIC_DISTANCE
    else:
        edgeIndicator = EdgeIndicator.EUCLIDEAN_DISTANCE

    seg = perform_segmentation(
        image=img,
        initial_contours_coordinates=[tuple(parameter_cache[img_name]["contour"])],
        iter_inner=parameter_cache[img_name]["inner_iterations"],
        iter_outer=parameter_cache[img_name]["outer_iterations"],
        lmbda=parameter_cache[img_name]["lmbda"],
        alfa=parameter_cache[img_name]["alpha"],
        epsilon=1.5,
        sigma=parameter_cache[img_name]["sigma"],
        potential_function=PotentialFunction.DOUBLE_WELL,
        edge_indicator=edgeIndicator,
        amount_of_points=100
    )

    n = 0
    while True:
        try:
            n = n + 1
            phi = next(seg)
            if n == 1:
                filename = f"cache/iteration-{n}.png"
                _construct_image(phi, img, filename)

                # send the file to the client 
                img_contour = iio.imread(f"./cache/iteration-{n}.png")
                img_contour = image_to_base64(img_contour)
                await websocket.send_bytes(img_contour)

            else:
                filename = f"cache/iteration-{n}.png"
                _construct_image(phi, img, filename)

                # send the file to the client 
                img_contour = iio.imread(f"./cache/iteration-{n}.png")
                img_contour = image_to_base64(img_contour)
                await websocket.send_bytes(img_contour)
        except StopIteration:
            break
    
    # delete cache
    folder = './cache'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return


@app.post("/segment/")
async def create_upload_file(file: UploadFile, parameters: Annotated[str, Form()], contour: Annotated[str, Form()]):
    filename = file.filename
    file_path = os.path.join("img", filename)
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(parameters)

    parameters = json.loads(parameters)
    contour = json.loads(contour)
    parameters["contour"] = contour
    parameter_cache[filename] = parameters

    return {"filename": file.filename}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        filename = str(data)
        file_path = os.path.join("img", filename)
        if os.path.exists(file_path):
            await _segment(websocket, filename)
        else:
            await websocket.send_text(f"File {file_path} does not exist")


@app.get("/cache/clear/")
async def clear_cache():
    dir_path = "img"

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return {"message": "Error deleting cache"}
    return {"message": "Cache cleared"}


def _construct_image(phi: np.ndarray, img: np.ndarray, name: str):
    fig = plt.figure(1, figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    fig.patch.set_alpha(0)  # Set the figure background to transparent
    contours = measure.find_contours(phi, 0)

    ax2 = fig.add_subplot(111)

    ax2.axis('off')

    ax2.imshow(img, interpolation='nearest')
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)
    fig.savefig(name)


def image_to_base64(img: np.ndarray) -> bytes:
    """ Given a numpy 2D array, returns a JPEG image in base64 format """
    img_buffer = cv2.imencode('.png', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')