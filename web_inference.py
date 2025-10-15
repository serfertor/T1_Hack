from queue import Queue

from flask import request,jsonify, send_file
from app import app
import hashlib
import os
import tempfile
import threading
import time
from classifier.inference import FAQInference
from classifier.inference import model_config, id_to_label_maps

def run_application():
    if __name__ == '__main__':
        threading.Thread(target=lambda: app.run(debug=False)).start()

class Task:
    def __init__(self, task_type, func, kwargs: dict[str, any]):
        self.task_id = hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()
        self.data = {"id": self.task_id, "type": task_type, "status": "process", "content": {}}
        self.func = func
        self.kwargs = kwargs
        self.status_code = 200

    def set_result(self, content, status_code):
        if content is None:
            self.data["content"] = {}
            self.data["status"] = "abort"
            self.status_code = status_code
        else:
            self.data["status"] = "ready"
            self.data["content"] = content
            self.status_code = status_code


class MainApplication:
    def __init__(self):
        self.queue = Queue()
        self.cache = {}
        self.inference = FAQInference(model_config, id_to_label_maps)

    def start(self):
        while True:
            if self.queue.empty():
                continue
            task = self.queue.get_nowait()
            print(f"task: {task.data}" )
            if not task.data.get("status") == "process":
                continue
            content, status_code = task.func(task.kwargs)
            task.set_result(content, status_code)
            self.cache[task.data["id"]] = task

    def add_task(self, task: Task):
        self.queue.put(task)
        self.cache[task.data["id"]] = task



    def get_request_hint(self, question):
        predictions = self.inference.predict(question)
        return predictions


application = MainApplication()


def get_hint_wrapper(kwargs):
    result = application.get_request_hint(kwargs.get("question"))
    if not result:
        return {"error": "error while getting hint"}, 500
    return result, 200




@app.route('/request_hint', methods=['POST'])
def request_hint():
    question = request.json.get("question")
    task = Task("index_in_db", get_hint_wrapper, {
                                                              "question": question})
    application.add_task(task)

@app.route('/task_status', methods=['POST'])
def task_status():
    task_id = request.json.get('task_id')
    if task_id is None or task_id == "" or task_id not in application.cache.keys():
        return jsonify({"result": "error", "error_str": "task_id cant be empty"}), 422
    return jsonify(application.cache[task_id].data["content"]), application.cache[task_id].status_code

run_application()
application.start()