import base64

import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
from io import BytesIO

header = st.beta_container()
desc = st.beta_container()
inference = st.beta_container()
json_ct = st.beta_container()

colors = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
          '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']

is_label_show = True

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5x, custom #yolov5s'

json_out = None


def json_create(img_name, width, height, class_names, class_idx, confidences, boxes, total_detections):

    detections_dict = {}

    # gathering data for the same class
    for idx, class_name in enumerate(class_names):

        if class_name not in list(detections_dict.keys()):
            class_confidences = []
            class_boxes = []
            class_ids = []
            class_confidences.append(confidences[idx])
            class_boxes.append(boxes[idx])
            class_ids.append(class_idx[idx])

            inner_dict = {}
            inner_dict['inner_class_confidences'] = class_confidences
            inner_dict['inner_class_boxes'] = class_boxes
            inner_dict['inner_class_ids'] = class_ids
            detections_dict[class_name] = inner_dict

        else:
            class_confidences = detections_dict.get(class_name).get('inner_class_confidences')
            class_boxes = detections_dict.get(class_name).get('inner_class_boxes')
            class_ids = detections_dict.get(class_name).get('inner_class_ids')

            class_confidences.append(confidences[idx])
            class_boxes.append(boxes[idx])
            class_ids.append(class_idx[idx])

            # do  not sure this part really necessary
            inner_dict = {}
            inner_dict['inner_class_confidences'] = class_confidences
            inner_dict['inner_class_boxes'] = class_boxes
            inner_dict['inner_class_ids'] = class_ids
            detections_dict[class_name] = inner_dict

    ## adding to json format
    data = {}

    data['info'] = []

    data['info'].append({
        'img_name': img_name,
        'width': width,
        'height': height,
        'total_detections' : total_detections
    })

    data['detections'] = []


    for class_name in list(detections_dict.keys()):
        grouped_detections_dict = detections_dict.get(class_name)

        print(class_name, '>>>>>>>>')
        print(grouped_detections_dict)

        grouped_confidences = grouped_detections_dict.get('inner_class_confidences')
        grouped_boxes = grouped_detections_dict.get('inner_class_boxes')
        class_idx = grouped_detections_dict.get('inner_class_ids')[0]
        num_of_detections = len(grouped_confidences)

        detection_list = []
        for i in range(len(grouped_confidences)):
            single_dict = {}
            single_dict['confidence'] = grouped_confidences[i]
            single_dict['coordinates'] = grouped_boxes[i]
            detection_list.append(single_dict)

        data['detections'].append({
            'brand_name' : class_name,
            'brand_id' : class_idx,
            'num_of_detections' : num_of_detections,
            'detections': detection_list

        })

    #with open('data.json', 'w') as outfile:
        #json.dump(data, outfile)
    return data


with header:
    #st.title('Logo Detection')
    #header.markdown("<img src='/logo.png' />", unsafe_allow_html=True)
    #header.image('logo.png', width=100)

    LOGO_IMAGE = "logo.png"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    header.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        </div>
        """,
        unsafe_allow_html=True
    )

    header.markdown("<h1 style='text-align: center; color: rgb(38, 39, 48);'>Logo Detection</h1>", unsafe_allow_html=True)

with desc:
    #st.subheader('The Description')
    #st.text('You will able to detect over 3000 logos in the uploaded images by using our model.')

    desc.markdown("<h4 style='text-align: left; color: rgb(38, 39, 48);'>Description</h4>",
                    unsafe_allow_html=True)
    desc.markdown("<p style='text-align: justify; color: rgb(38, 39, 48);'>Our logo detection model has trained over 3000 different brands. "
                  "You will able to use our model to logo in your uploaded images. You can upload image either from your device or by copy and pasting image url. </p>",
                  unsafe_allow_html=True)
    desc.markdown(
        "<p style='text-align: justify; color: rgb(38, 39, 48);'>Choose the confidence score that you would like to have in detection results."
        "The detection results having same or higher confidence scores compared to your chosen score will be shown below.</p>",
        unsafe_allow_html=True)

with inference:
    st.set_option('deprecation.showPyplotGlobalUse', False)

    sel_col , disp_col = st.beta_columns(2)

    upload_col, detect_col = st.beta_columns(2)

    is_having_image = False

    selection = sel_col.radio('Upload image', ('From Device', 'By Link'))

    confidence_score = disp_col.slider("Confidence Threshold", 0.00, 1.00, 0.5, 0.05)

    if selection == 'From Device':

        uploaded_file = sel_col.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg', 'webp'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_name = uploaded_file.name
            is_having_image = True

    elif selection == 'By Link':
        img_url = sel_col.text_area('Image URL', '''''')
        if len(img_url) != 0:
            img_name = img_url.split('/')[-1]
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content))
            is_having_image = True


    if is_having_image:
        upload_col.subheader("Uploaded  Image")
        st.text("")
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        upload_col.pyplot(use_column_width=True)
        #st.image(image, caption='Uploaded Image.', use_column_width=True)

        new_img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        height, width, channels = img.shape

        class_names = []
        confidences = []
        class_ids = []
        boxes = []

        result = model(new_img)

        pd = result.pandas().xyxy[0]
        print(pd)

        total_detections = 0

        for index, row in pd.iterrows():
            x_min, y_min, x_max, y_max, confidence, class_name, class_id = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'], row['name'], row['class']

            if confidence > confidence_score:

                class_names.append(class_name)
                confidences.append(confidence)
                class_ids.append(class_id)
                boxes.append([(int(x_min), int(y_min)), (int(x_max), int(y_max))])
                total_detections+=1

        for i in range(len(boxes)):
                point_1, point_2 = boxes[i]

                unique_classes =  list(dict.fromkeys(class_names))
                unique_color = colors[unique_classes.index(class_names[i])]

                color_rgb = tuple(int(unique_color[i:i+2], 16) for i in (0, 2, 4))

                line_thickness = round(0.002 * (new_img.shape[0] + new_img.shape[1]) / 2) + 1
                print(line_thickness)

                cv2.rectangle(new_img, point_1, point_2, color_rgb, line_thickness)
                if is_label_show:
                    font_thickness = max(line_thickness - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(class_names[i], 0, fontScale=font_thickness / 3, thickness=font_thickness)[0]
                    c2 = point_1[0] + t_size[0], point_1[1] - t_size[1] - 3
                    cv2.rectangle(new_img, point_1, c2, color_rgb, -1, cv2.LINE_AA)  # filled
                    cv2.putText(new_img, class_names[i], (point_1[0], point_1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        st.text("")
        detect_col.subheader("Object-Detected Image")
        st.text("")
        plt.figure(figsize=(15, 15))
        plt.imshow(new_img)
        detect_col.pyplot(use_column_width=True)

        json_out = json_create(img_name, width, height, class_names, class_ids, confidences, boxes, total_detections)

with json_ct:
    if json_out is not None:
        st.subheader('JSON Output')
        #json_ct.success(json_out)
        json_ct.json(json_out)
        #json_ct.write(json_out)





