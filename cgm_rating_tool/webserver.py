from flask import Response
from flask import Flask
from flask import render_template
from flask import request
from flask import make_response
from flask import jsonify
from flask import redirect
import threading
import time
import cv2
import uuid

# initialize a flask object
app = Flask(__name__)

def main():
    ip = "0.0.0.0"
    port = "8666"

    # start the flask app
    print("Starting flask-app...")
    app.run(
        host=ip,
        port=port,
        debug=True,
        threaded=True,
        use_reloader=False)


# Goes directly to the main page.
@app.route("/")
def index():

    # Gets a random measure.
    random_measure_id = str(uuid.uuid1())

    # Redirect to the measure.
    return redirect("/{}". format(random_measure_id), code=302)


# Shows a random measure.
@app.route("/<measure_id>")
def index_with_measure(measure_id):
    # return the rendered template
    return render_template("index.html", measure_id=measure_id)


# Returns the image of the measure.
@app.route("/image/<measure_id>")
def image_with_measure(measure_id):

    image = get_image(measure_id)

    return Response(
        image,
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )


# Method for physically fetching the measure.
def get_image(measure_id):

        # TODO use the right id here and return proper picture
        image = cv2.imread("templates/default.jpg")

        (flag, encodedImage) = cv2.imencode(".jpg", image)

        # yield the output frame in the byte format
        return(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


# POST endpoint for the decision.
@app.route("/decide/<measure_id>", methods=["POST"])
def decide(measure_id):

    decision = request.form["decision"]

    if decision == "good":
        print("Measure {} is good.".format(measure_id))
    elif decision == "bad":
        print("Measure {} is bad.".format(measure_id))
    elif decision == "delete":
        print("Measure {} should be deleted.".format(measure_id))

    return redirect("/", code=302)


# Start.
if __name__ == '__main__':

    main()
