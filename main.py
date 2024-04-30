import cv2

# Load pre-trained model
model_path = "age_gender_model.xml"
age_gender_model = cv2.dnn.readNet(model_path)

# Load image
image_path = "sample_image.jpg"
image = cv2.imread(image_path)

# Convert image to blob
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

# Pass the blob through the network to get predictions
age_gender_model.setInput(blob)
predictions = age_gender_model.forward()

# Get gender and age predictions
gender = "Male" if predictions[0][0] > 0.5 else "Female"
age = predictions[1][0]

# Display the result
print("Gender:", gender)
print("Age:", age)
