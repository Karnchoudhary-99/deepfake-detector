from deepfake_detector.faces import find_face


def test_skin_face_found_near_expected_box(sample_images):
    box = find_face(sample_images["real_img"])
    assert box.method == "skin"
    # Synthetic face region is (64, 48, 128x160); allow mask/resize slack.
    assert 40 <= box.x <= 90
    assert 25 <= box.y <= 75
    assert 90 <= box.width <= 170
    assert 120 <= box.height <= 200
    assert box.confidence > 0.5


def test_blank_image_uses_labelled_fallback(blank_image):
    box = find_face(blank_image)
    assert box.method == "center-fallback"
    assert box.confidence == 0.3
    assert 0 <= box.x and box.right <= 200
    assert 0 <= box.y and box.bottom <= 200


def test_expanded_box_clamped_to_image(sample_images):
    box = find_face(sample_images["real_img"])
    expanded = box.expanded(sample_images["real_img"].size, margin=1.0)
    assert expanded.right <= 256 and expanded.bottom <= 256
    assert expanded.x >= 0 and expanded.y >= 0
