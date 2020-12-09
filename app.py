from flask import Flask, render_template, url_for, request
import datetime
import os
from masterAlgorithm.homography import DamageDetectorWithImages


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ".\source_images"
app.config['UPLOAD_FOLDER2'] = r'.\masterAlgorithm\uploaded_dataset'

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        ct = str(datetime.datetime.now().timestamp()).replace(".", "")
        uploaded_image = request.files['damage_file']
        upl_file_name = os.path.splitext(uploaded_image.filename)[1]
        full_path = os.path.join(app.config['UPLOAD_FOLDER'],'source_img_'+ct+upl_file_name)
        uploaded_image.save( full_path )
        damageDetectorWithImages = DamageDetectorWithImages()
        detection_status, highlights_cnt, output_file_path = damageDetectorWithImages.compareAndHighlightDamages(full_path)
        
        print("Detected: ", end="")
        if detection_status == 1:
            print("Yes", end="")
            if highlights_cnt > 0:
                print(" - Number Of Damages detected: ", highlights_cnt," - Output_file_path: ",output_file_path, end="")
            else:
                print(" - No Damages detected")

        else:
            print("No")
        return render_template('index_output.html', outfilepath = output_file_path )

@app.route('/update_ds', methods=['POST','GET'])
def index2():
    if request.method == 'GET':
        return render_template('dataset_image_upload.html')
    else:
        ct = str(datetime.datetime.now().timestamp()).replace(".", "")
        uploaded_image = request.files['damage_file']
        upl_file_name = os.path.splitext(uploaded_image.filename)[1]
        full_path = os.path.join(app.config['UPLOAD_FOLDER2'],'source_img_'+ct+upl_file_name)
        uploaded_image.save( full_path )
        return render_template('dataset_image_upload_output.html', uploaded_path = full_path)

if __name__ == "__main__":
    app.run(debug = True)