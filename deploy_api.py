from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime
from src.face_judgement_align import IDphotos_create
from src.layoutCreate import generate_layout_photo, generate_layout_image
from hivisionai.hycv.vision import add_background
import base64
import numpy as np
import cv2
import ast

app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='.*(ehafo|yiqizuoti|yihafo)\.com.*',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# 将hex颜色值转换为BGR颜色值字符串
def hex2bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    # 3位hex颜色值转换为6位
    if len(hex_color) == 3:
        hex_color = "".join([c + c for c in hex_color])
    rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    return str(bgr_color)


# 将图像转换为 Base64 编码
def numpy_2_base64(img: np.ndarray):
    # retval, buffer = cv2.imencode('.png', img)
    # retval不需要，但是要接收，所以用_代替
    _, buffer = cv2.imencode('.png', img)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return base64_image


# 根据参数获取图片
async def getImg(input_image, input_image_base64, imgDecodeType=cv2.IMREAD_COLOR):
    img = None
    if input_image:
        # 优先使用file类型传入
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, imgDecodeType)
    elif input_image_base64:
        # 同时支持base64编码传入
        base64_bytes = input_image_base64.encode('utf-8')
        image_bytes = base64.b64decode(base64_bytes)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, imgDecodeType)

    return img


# 证件照智能制作接口（结合抠图+添加背景）
@app.post("/make_idphoto")
async def make_idphoto(
    input_image: UploadFile = None,
    input_image_base64: str = Form(None),
    size: str = Form(...),
    color: str = Form(None),
    color_hex: str = Form(None),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio=0.45,
    top_distance_max=0.12,
    top_distance_min=0.10,
):
    img = await getImg(input_image, input_image_base64)
    color = color if color else hex2bgr(color_hex)
    # 将字符串转为元组
    size = ast.literal_eval(size)
    (
        result_image_hd,
        result_image_standard,
        typography_arr,
        typography_rotate,
        _,
        _,
        _,
        _,
        status,
    ) = IDphotos_create(
        img,
        size=size,
        head_measure_ratio=head_measure_ratio,
        head_height_ratio=head_height_ratio,
        align=False,
        beauty=False,
        fd68=None,
        human_sess=sess,
        IS_DEBUG=False,
        top_distance_max=top_distance_max,
        top_distance_min=top_distance_min,
    )

    # 如果检测到人脸数量不等于1（照片无人脸 or 多人脸）
    if status == 0:
        result_message = {"status": False}

    # 如果检测到人脸数量等于1, 则获取标准照和高清照结果（png 4通道图像）
    else:
        img_result = {
            "status": True,
            "img_output_standard": numpy_2_base64(result_image_standard),
            "img_output_standard_hd": numpy_2_base64(result_image_hd),
        }

        # 将标准照添加背景
        img = await getImg(
            None, img_result["img_output_standard"], cv2.IMREAD_UNCHANGED
        )

        color = ast.literal_eval(color)

        result_message = {
            "status": True,
            "image": numpy_2_base64(add_background(img, bgr=color)),
        }
    return result_message


# 证件照智能抠图接口
@app.post("/idphoto")
async def idphoto_inference(
    input_image: UploadFile = None,
    input_image_base64: str = Form(None),
    size: str = Form(...),
    head_measure_ratio=0.2,
    head_height_ratio=0.45,
    top_distance_max=0.12,
    top_distance_min=0.10,
):
    img = await getImg(input_image, input_image_base64)
    # 将字符串转为元组
    size = ast.literal_eval(size)

    (
        result_image_hd,
        result_image_standard,
        typography_arr,
        typography_rotate,
        _,
        _,
        _,
        _,
        status,
    ) = IDphotos_create(
        img,
        size=size,
        head_measure_ratio=head_measure_ratio,
        head_height_ratio=head_height_ratio,
        align=False,
        beauty=False,
        fd68=None,
        human_sess=sess,
        IS_DEBUG=False,
        top_distance_max=top_distance_max,
        top_distance_min=top_distance_min,
    )

    # 如果检测到人脸数量不等于1（照片无人脸 or 多人脸）
    if status == 0:
        result_message = {"status": False}

    # 如果检测到人脸数量等于1, 则返回标准证和高清照结果（png 4通道图像）
    else:
        result_message = {
            "status": True,
            "img_output_standard": numpy_2_base64(result_image_standard),
            "img_output_standard_hd": numpy_2_base64(result_image_hd),
        }

    return result_message


# 透明图像添加纯色背景接口
@app.post("/add_background")
async def photo_add_background(
    input_image: UploadFile = None,
    input_image_base64: str = Form(...),
    color: str = Form(...),
):
    img = await getImg(input_image, input_image_base64, cv2.IMREAD_UNCHANGED)

    color = ast.literal_eval(color)

    # try:
    result_message = {
        "status": True,
        "image": numpy_2_base64(add_background(img, bgr=color)),
    }

    # except Exception as e:
    #     print(e)
    #     result_message = {
    #         "status": False,
    #         "error": e
    #     }

    return result_message


# 六寸排版照生成接口
@app.post("/generate_layout_photos")
async def generate_layout_photos(
    input_image: UploadFile = None,
    input_image_base64: str = Form(...),
    size: str = Form(...),
):
    try:
        img = await getImg(input_image, input_image_base64)

        size = ast.literal_eval(size)

        typography_arr, typography_rotate = generate_layout_photo(
            input_height=size[0], input_width=size[1]
        )

        result_layout_image = generate_layout_image(
            img, typography_arr, typography_rotate, height=size[0], width=size[1]
        )

        result_message = {
            "status": True,
            "image": numpy_2_base64(result_layout_image),
        }

    except Exception as e:
        result_message = {
            "status": False,
        }

    return result_message


if __name__ == "__main__":
    import uvicorn

    # 加载权重文件
    HY_HUMAN_MATTING_WEIGHTS_PATH = "./hivision_modnet.onnx"
    sess = onnxruntime.InferenceSession(HY_HUMAN_MATTING_WEIGHTS_PATH)

    # 在 8080 端口运行推理服务
    uvicorn.run(app, host="0.0.0.0", port=8080)
