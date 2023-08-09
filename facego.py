import cv2
import pickle
from simple_facerec import SimpleFacerec

# Load the saved model
with open("face_recognition_model.pkl", "rb") as f:
    sfr = pickle.load(f)

# Load the input image
input_image = cv2.imread("test.jpg")

# Perform face recognition on the input image
face_locations, face_names = sfr.detect_known_faces(input_image)
DOOR = False
for face_loc, name in zip(face_locations, face_names):
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
    cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 200), 4)
    cv2.putText(input_image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    if name in ["ryan", "robert", "nick","grace","claude","miranda","brayo","mark","leonard","fidel","sam","m.allan",]:
        DOOR = True

# Save the processed image
cv2.imwrite("output_image.jpg", input_image)

# Display the processed image
cv2.imshow("Processed Image", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if the door should be opened or not
if DOOR == True:
    print("Opening the door...")
else:
    print("Access denied.")

