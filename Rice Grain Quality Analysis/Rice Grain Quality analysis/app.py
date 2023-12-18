from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

def get_classification(ratio):
    ratio = round(ratio, 1)
    to_ret = ""
    if ratio >= 3:
        to_ret = "Slender"
    elif 2.1 <= ratio < 3:
        to_ret = "Medium"
    elif 1.1 <= ratio < 2.1:
        to_ret = "Bold"
    elif ratio <= 1:
        to_ret = "Round"
    to_ret = "(" + to_ret + ")"
    return to_ret

def calculate_dust_percentage(img):
    # Your dust detection logic goes here
    # For example, you can apply thresholding and contour detection to identify dust particles
    _, binary_dust = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # Adjust the threshold as needed

    # Find contours for dust detection
    contours_dust, _ = cv2.findContours(binary_dust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate dust percentage
    total_pixels = img.size
    dust_pixels = sum(cv2.contourArea(cnt) for cnt in contours_dust)
    dust_percentage = (dust_pixels / total_pixels) * 100

    return dust_percentage

def process_image(img):
    # Convert the processed image to binary
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

    # Apply an averaging filter
    kernel = np.ones((5, 5), np.float32) / 9
    dst = cv2.filter2D(binary, -1, kernel)  # -1: depth of the destination image

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Perform erosion
    erosion = cv2.erode(dst, kernel2, iterations=1)

    # Perform dilation
    dilation = cv2.dilate(erosion, kernel2, iterations=1)

    # Apply edge detection
    edges = cv2.Canny(dilation, 100, 200)

    # Find contours for size detection
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("No. of rice grains=", len(contours))

    total_ar = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio
        print(round(aspect_ratio, 2), get_classification(aspect_ratio))
        total_ar += aspect_ratio

    avg_ar = total_ar / len(contours)
    print("Average Aspect Ratio=", round(avg_ar, 2), get_classification(avg_ar))

    # Calculate dust percentage
    dust_percentage = calculate_dust_percentage(img)
    print("Dust Percentage =", round(dust_percentage, 2), "%")

    # Calculate rice grain percentage
    rice_grain_percentage = 100 - dust_percentage

    # Convert the processed image to base64 for displaying in the browser
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    _, buffer1 = cv2.imencode('.jpg', binary)
    filter_img_str1 = base64.b64encode(buffer1).decode('utf-8')

    _, buffer2 = cv2.imencode('.jpg', dst)
    filter_img_str2 = base64.b64encode(buffer2).decode('utf-8')

    _, buffer3 = cv2.imencode('.jpg', erosion)
    filter_img_str3 = base64.b64encode(buffer3).decode('utf-8')

    _, buffer4 = cv2.imencode('.jpg', dilation)
    filter_img_str4 = base64.b64encode(buffer4).decode('utf-8')

    _, buffer5 = cv2.imencode('.jpg', edges)
    filter_img_str5 = base64.b64encode(buffer5).decode('utf-8')

    # Collect information about rice grains and aspect ratios
    grain_info = []
    grain_counts = {}  # Dictionary to store counts for each grain type

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio

        grain_type = get_classification(aspect_ratio)
        grain_info.append((round(aspect_ratio, 2), grain_type))

        # Increment the count for the grain type
        if grain_type not in grain_counts:
            grain_counts[grain_type] = 1
        else:
            grain_counts[grain_type] += 1

    # Calculate average aspect ratio
    avg_aspect_ratio = round(avg_ar, 2)

    # Data
    labels = ['Dust Percentage', 'Rice Grain']
    sizes = [dust_percentage, rice_grain_percentage]
    colors = ['lightcoral', 'lightskyblue']

# Plotting the Pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

# Save the pie chart as an image
    plt.savefig('pie_chart.png')

# Display the pie chart
    plt.show()

    # Return the base64 strings, grain counts, and dust percentage
    return img_str, filter_img_str1, filter_img_str2, filter_img_str3, filter_img_str4, filter_img_str5, grain_info, avg_aspect_ratio, grain_counts, dust_percentage, rice_grain_percentage



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image
        image = request.files["image"]
        if image:
            # Read the image and convert it to grayscale
            img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

            # Process the image and get the base64 string, grain counts, and dust percentage
            processed_image = process_image(img)

            return render_template("result.html", processed_image=processed_image, get_classification=get_classification)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
