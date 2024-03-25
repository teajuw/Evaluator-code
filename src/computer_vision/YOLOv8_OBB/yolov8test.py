def main():
    # Load violin detection model
    violin_model_path = "best.pt"
    violin_model = load_violin_detection_model(violin_model_path)

    # Existing code...
    video_capture = cv2.VideoCapture(0)  # assuming webcam
    frame_count = 0

    while True:
        # Get next frame
        ret, frame = video_capture.read()
        frame_count += 1
        frame = cv2.flip(frame, 1)

        # Detect violin parts
        violin_prediction = detect_violin_parts(frame, violin_model)

        # Process violin prediction and overlay on the frame
        overlay_violin_prediction(frame, violin_prediction)

        # Display the frame
        cv2.imshow('Violin Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    video_capture.release()
    cv2.destroyAllWindows()

def overlay_violin_prediction(frame, violin_prediction):
    # Process violin prediction and overlay on the frame
    for prediction in violin_prediction:
        x, y, w, h = prediction['box']
        label = prediction['label']
        confidence = prediction['confidence']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put label and confidence
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)