from flask import Flask, render_template, request
import subprocess
import os
import shutil

app = Flask('MDL')

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files[]")
    minFreq = request.form.get('minFreq')

    # 清空目标文件夹
    target_folder = os.path.abspath("uploads")
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    for file in files:
        file_path = os.path.join(target_folder, file.filename)
        file.save(file_path)

    # 执行cmd命令，假设cmd命令需要文件夹路径作为参数
    cmd = f"Rscript MDL-R.R {target_folder} {minFreq}"  # 替换cmd命令和参数
    print(cmd)
    subprocess.call(cmd, shell=True)

    return "文件夹上传和命令执行完成"


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8080)
